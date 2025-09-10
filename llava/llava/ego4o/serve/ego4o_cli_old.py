import argparse
import pickle

import torch

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, \
    DEFAULT_IMAGE_PATCH_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.ego4o.constants import DEFAULT_MOTION_TOKEN
from llava.ego4o.dataset.transforms.motion_transforms import ConvertNymeriaToHML, ZUp2YUp, InitAlignIMUMotion, \
    HMLMotionRepresentation, NormalizeHMLMotion, PadMotion, ChangeHMLShape
from llava.ego4o.model.ego4o import Ego4oForCausalLM
from llava.ego4o.train.train_ego4o import tokenizer_motion_token
from llava.utils import disable_torch_init
from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path

from PIL import Image

import requests
from PIL import Image
from io import BytesIO
from transformers import TextStreamer, AutoTokenizer, BitsAndBytesConfig


def load_pretrained_model(model_path, model_base, model_name, load_8bit=False, load_4bit=False, device_map="auto", device="cuda", use_flash_attn=False, **kwargs):
    kwargs = {"device_map": device_map, **kwargs}

    if device != "cuda":
        kwargs['device_map'] = {"": device}

    if load_8bit:
        kwargs['load_in_8bit'] = True
    elif load_4bit:
        kwargs['load_in_4bit'] = True
        kwargs['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4'
        )
    else:
        kwargs['torch_dtype'] = torch.float16

    if use_flash_attn:
        kwargs['attn_implementation'] = 'flash_attention_2'

    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    model = Ego4oForCausalLM.from_pretrained(
        model_path,
        **kwargs
    )

    mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
    mm_use_im_patch_token = getattr(model.config, "mm_use_im_patch_token", True)
    if mm_use_im_patch_token:
        tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
    if mm_use_im_start_end:
        tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
    model.resize_token_embeddings(len(tokenizer))

    vision_tower = model.get_vision_tower()
    if not vision_tower.is_loaded:
        vision_tower.load_model(device_map=device_map)
    if device_map != 'auto':
        vision_tower.to(device=device_map, dtype=torch.float16)
    image_processor = vision_tower.image_processor

    if hasattr(model.config, "max_sequence_length"):
        context_len = model.config.max_sequence_length
    else:
        context_len = 2048

    return tokenizer, model, image_processor, context_len

def load_image(image_file):
    if image_file.startswith('http://') or image_file.startswith('https://'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image

def load_motion(motion_file):
    with open(motion_file, "rb") as f:
        motion_data = pickle.load(f)  # motion data is a dictionary
    return motion_data


def image_parser(args):
    out = args.image_file.split(args.sep)
    return out


def load_images(image_files):
    out = []
    for image_file in image_files:
        image = load_image(image_file)
        out.append(image)
    return out

def main(args):
    # Model
    disable_torch_init()

    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, args.model_base, model_name, args.load_8bit, args.load_4bit, device=args.device)


    if "llama-2" in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "mistral" in model_name.lower():
        conv_mode = "mistral_instruct"
    elif "v1.6-34b" in model_name.lower():
        conv_mode = "chatml_direct"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"

    if args.conv_mode is not None and conv_mode != args.conv_mode:
        print('[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}'.format(conv_mode, args.conv_mode, args.conv_mode))
    else:
        args.conv_mode = conv_mode

    conv = conv_templates[args.conv_mode].copy()
    roles = conv.roles

    motion_path = args.motion_path
    motion_list: dict = load_motion(motion_path)
    # random select one id from motion_list
    motion_id = list(motion_list.keys())[10]
    print(f"motion_id: {motion_id}")
    motion_data_transform_input = {
        'segment_tXYZ': motion_list[motion_id]['segment_tXYZ'],
    }
    human_motion_transforms = [
        ConvertNymeriaToHML(joint_name='segment_tXYZ', out_name=None),
        ZUp2YUp(joint_name='segment_tXYZ'),
        InitAlignIMUMotion(imu_acc_name=None, imu_ori_name=None, joint_name='segment_tXYZ'),
        HMLMotionRepresentation(joint_name='init_aligned_global_smpl_joints'),
        NormalizeHMLMotion(hml_motion_name='motion_hml',
                           hml_mean_path='/CT/EgoMocap/work/EgoOmniMocap/projects/TLControl/demo/info_motion_mean.pt',
                           hml_std_path='/CT/EgoMocap/work/EgoOmniMocap/projects/TLControl/demo/info_motion_std.pt'),
        PadMotion(seq_len=196, pad_name_list=('motion_hml',), resize_input_sequence=True),
        ChangeHMLShape(hml_motion_name='motion_hml'),
    ]
    for transform in human_motion_transforms:
        motion_data_transform_input = transform.transform(motion_data_transform_input)
    motion_hml = torch.asarray(motion_data_transform_input['motion_hml']).unsqueeze(0).float().cuda()

    # Images

    image = load_image(args.image_file)
    image_size = image.size
    # Similar operation in model_worker.py
    image_tensor = process_images([image], image_processor, model.config)
    if type(image_tensor) is list:
        image_tensor = [image.to(model.device, dtype=torch.float16) for image in image_tensor]
    else:
        image_tensor = image_tensor.to(model.device, dtype=torch.float16)



    while True:
        try:
            inp = input("USER: ")
        except EOFError:
            inp = ""
        if not inp:
            print("exit...")
            break

        print("ASSISTANT: ", end="")

        # if image is not None:
        #     # first message
        #     if model.config.mm_use_im_start_end:
        #         inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + inp
        #     else:
        #         inp = DEFAULT_IMAGE_TOKEN + '\n' + inp
        #     image = None

        conv.append_message(conv.roles[0], inp)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

        input_ids = (
            tokenizer_motion_token(prompt, tokenizer, return_tensors="pt")
            .unsqueeze(0)
            .cuda()
        )

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor,
                image_sizes=[image_size],
                motion_hml=motion_hml,
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                max_new_tokens=args.max_new_tokens,
                streamer=streamer,
                use_cache=True)

        outputs = tokenizer.decode(output_ids[0]).strip()
        conv.messages[-1][-1] = outputs

        if args.debug:
            print("\n", {"prompt": prompt, "outputs": outputs}, "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="checkpoints/motion_to_text_finetune")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--motion_path", type=str, required=True)
    parser.add_argument("--image-file", type=str, required=False,
                        default="https://llava-vl.github.io/static/images/view.jpg")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--sep", type=str, default=",")
    args = parser.parse_args()
    main(args)
