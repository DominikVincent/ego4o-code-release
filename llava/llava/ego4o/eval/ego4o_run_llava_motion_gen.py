import argparse
import pickle
import torch.nn.functional as F
import torch
from transformers import AutoTokenizer, BitsAndBytesConfig

from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IMAGE_PLACEHOLDER, DEFAULT_IMAGE_PATCH_TOKEN,
)
from llava.conversation import conv_templates, SeparatorStyle
from llava.ego4o.constants import DEFAULT_MOTION_TOKEN, MOTION_TOKEN_INDEX
from llava.ego4o.dataset.transforms.motion_transforms import ConvertNymeriaToHML, InitAlignIMUMotion, \
    HMLMotionRepresentation, PadMotion, ChangeHMLShape, ZUp2YUp, NormalizeHMLMotion
from llava.ego4o.model.ego4o import Ego4oForCausalLM
from llava.ego4o.train.train_ego4o import tokenizer_motion_token
# from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import (
    process_images,
    tokenizer_image_token,
    get_model_name_from_path,
)

from PIL import Image

import requests
from PIL import Image
from io import BytesIO
import re
import pdb

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


def load_motion(motion_file):
    with open(motion_file, "rb") as f:
        motion_data = pickle.load(f)  # motion data is a dictionary
    return motion_data


def image_parser(args):
    out = args.image_file.split(args.sep)
    return out


def load_image(image_file):
    if image_file.startswith("http") or image_file.startswith("https"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    return image

def load_images(image_files):
    out = []
    for image_file in image_files:
        image = load_image(image_file)
        out.append(image)
    return out

def eval_model(args):
    # Model
    disable_torch_init()

    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        args.model_path, args.model_base, model_name
    )

    human_motion_token_id = tokenizer(DEFAULT_MOTION_TOKEN, add_special_tokens=False).input_ids[0]
    model.motion_token_index = human_motion_token_id

    qs = args.query
    # qs = DEFAULT_MOTION_TOKEN + "\n" + qs

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
        print(
            "[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}".format(
                conv_mode, args.conv_mode, args.conv_mode
            )
        )
    else:
        args.conv_mode = conv_mode

    conv = conv_templates[args.conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()


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

    image_files = image_parser(args)
    images = load_images(image_files)
    image_sizes = [x.size for x in images]
    images_tensor = process_images(
        images,
        image_processor,
        model.config
    ).to(model.device, dtype=torch.float16)

    input_ids = (
        tokenizer_motion_token(prompt, tokenizer, return_tensors="pt")
        .unsqueeze(0)
        .cuda()
    )

    with torch.no_grad():
        output_dict = model.generate(
            input_ids,
            images=images_tensor,
            image_sizes=image_sizes,
            motion_hml=motion_hml,
            do_sample=True if args.temperature > 0 else False,
            temperature=args.temperature,
            top_p=args.top_p,
            num_beams=args.num_beams,
            max_new_tokens=args.max_new_tokens,
            use_cache=True,
            return_dict_in_generate=True,
            output_hidden_states=True,
        )

    # with torch.inference_mode():
    #     output_dict = model(
    #         input_ids,
    #         images=images_tensor,
    #         image_sizes=image_sizes,
    #         motion_hml=motion_hml,
    #         use_cache=True,
    #         output_hidden_states=True,
    #     )



    hidden_state = output_dict.hidden_states[0][-1]
    hidden_state = model.get_model().vq_net_decoder_preprocess(hidden_state)

    hidden_state_motion_part = hidden_state[:, -50:-1, :]
    bs, N_token, emb_dimx6 = hidden_state_motion_part.shape
    vq_net_decoder_reshape = hidden_state_motion_part.reshape(bs, N_token, 6, emb_dimx6 // 6)
    vq_net_decoder_permute = vq_net_decoder_reshape.permute(0, 2, 1, 3)  # [batch_size, 6, N_token, emb_dim]

    codes_pick_gumbel_softmax = F.gumbel_softmax(vq_net_decoder_permute, tau=1, eps=1e-10, hard=True, dim=-1)

    # pdb.set_trace()

    x_quantized_fromIds = model.get_model().vq_net.vqvae.get_x_quantized_from_x_ids(codes_pick_gumbel_softmax.permute(0, 2, 3, 1).contiguous())

    vq_net_decoder_output = model.get_model().vq_net.vqvae.forward_decoder_from_quantized_codes(x_quantized_fromIds)

    # vq_net_decoder_input = hidden_state.permute(0, 2, 1)  # [batch_size, emb_dim, seq_len]
    # vq_net_decoder_output = model.get_model().vq_net.vqvae.decoder(vq_net_decoder_input)

    # save the output to pkl
    vq_net_decoder_output = vq_net_decoder_output.squeeze(-2).permute(0, 2, 1).cpu().detach().numpy()
    with open("/CT/EgoMocap/work/LLaVA/motion_out/output_motion_info.pkl", "wb") as f:
        pickle.dump(vq_net_decoder_output, f)
    print("output motion info saved to /CT/EgoMocap/work/LLaVA/motion_out/output_motion_info.pkl")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="checkpoints/multi_modal_finetun_old/checkpoint-500")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--motion_path", type=str, required=True)
    parser.add_argument("--image-file", type=str, required=False, default="https://llava-vl.github.io/static/images/view.jpg")
    parser.add_argument("--query", type=str, required=True)
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--sep", type=str, default=",")
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    args = parser.parse_args()

    eval_model(args)
