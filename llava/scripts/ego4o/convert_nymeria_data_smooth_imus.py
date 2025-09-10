import pdb
import sys
sys.path.append('../..')

import pickle
from tqdm import tqdm
import numpy as np
import json
import os
from llava.ego4o.constants import MOTION_TO_TEXT_QUESTION_LIST, IMAGE_TO_TEXT_QUESTION_LIST, IMAGE_MOTION_TO_TEXT_QUESTION_LIST


# smooth the free accleration
def smooth_free_accleration(free_accleration):
    pass

def down_sample_motion_data(motion_data, original_fps, target_fps):
    if target_fps > original_fps:
        raise ValueError("Target FPS must be less than or equal to the original FPS.")

    if original_fps % target_fps != 0:
        raise ValueError("Original FPS must be divisible by the target FPS.")

        # Calculate the downsampling factor
    downsampling_factor = original_fps // target_fps

    # Select every nth frame
    downsampled_motion_data = motion_data[::downsampling_factor]

    return downsampled_motion_data

def main():
    nymeria_root_path = '/scratch/inf0/user/jianwang/nymeria'
    # output_path = os.path.join(nymeria_root_path, 'ego4o_input')
    # os.makedirs(output_path, exist_ok=True)

    motion_output_path = os.path.join(nymeria_root_path, 'ego4o_input_motion')
    os.makedirs(motion_output_path, exist_ok=True)
    json_output_path = os.path.join(nymeria_root_path, 'ego4o_input_json_image_motion.jsonl')

    summary_path = os.path.join(nymeria_root_path, 'summary')
    atomic_path = os.path.join(nymeria_root_path, 'automic')

    print('processing automic data', flush=True)

    output_list = []

    # iterate over the atomic path
    for atomic_file in tqdm(os.listdir(atomic_path)):
        # print(f'Processing {atomic_file}', flush=True)
        atomic_file_path = os.path.join(atomic_path, atomic_file)
        seq_name = os.path.splitext(atomic_file)[0]
        # image_file = os.path.join(dummy_image_path, f'{seq_name}.jpg')

        # image_name = f'{seq_name}.jpg'
        with open(atomic_file_path, 'rb') as f:
            atomic_data_list = pickle.load(f)

        motion_output_pkl = {}

        # check if the number of images are same
        image_root_dir = os.path.join('/scratch/inf0/user/jianwang/nymeria/images', seq_name)
        image_count = len(os.listdir(image_root_dir))
        assert image_count == len(atomic_data_list), f'Number of images {image_count} does not match the number of atomic data {len(atomic_data_list)}'
        image_name_list = []
        for i in range(image_count):
            image_dir = f'{i:06d}'
            image_name = os.path.join(image_root_dir, seq_name, image_dir, 'frame_3.jpg')
            if not os.path.exists(image_name):
                image_name = os.path.join(image_root_dir, image_dir, 'frame_1.jpg')
            image_name_list.append(image_name)

        for i, atomic_data in enumerate(atomic_data_list):
            motion_data = atomic_data['motion']
            segment_tXYZ = motion_data['segment_tXYZ']
            sensor_qWXYZ = motion_data['sensor_qWXYZ']
            sensor_freeAcceleration = motion_data['sensor_freeAcceleration']
            segment_tXYZ = down_sample_motion_data(segment_tXYZ, 240, 30)
            sensor_qWXYZ = down_sample_motion_data(sensor_qWXYZ, 240, 30)
            sensor_freeAcceleration = down_sample_motion_data(sensor_freeAcceleration, 240, 30)
            new_id = f'{i}_{seq_name}'
            motion_output_pkl[new_id] = {
                'segment_tXYZ': segment_tXYZ,
                'sensor_qWXYZ': sensor_qWXYZ,
                'sensor_freeAcceleration': sensor_freeAcceleration
            }

            # generate motion to text data
            data_item = {}
            data_item['image'] = image_name_list[i]
            data_item['fps'] = 30
            data_item['id'] = new_id
            data_item['motion_id'] = [new_id]
            data_item['motion_file'] = f'{seq_name}.pkl'
            question_list = IMAGE_MOTION_TO_TEXT_QUESTION_LIST + IMAGE_TO_TEXT_QUESTION_LIST + MOTION_TO_TEXT_QUESTION_LIST
            random_question = np.random.choice(question_list)
            data_item['conversations'] = [
                {
                    'from': 'human',
                    'value': random_question
                },
                {
                    'from': 'gpt',
                    'value': atomic_data['text']['Describe my atomic actions']
                }
            ]
            output_list.append(data_item)

        # save motion data
        motion_output_file = os.path.join(motion_output_path, f'{seq_name}.pkl')
        with open(motion_output_file, 'wb') as f:
            pickle.dump(motion_output_pkl, f)
    # save output list to jsonl file
    with open(json_output_path, 'w') as f:
        for item in output_list:
            f.write(json.dumps(item) + '\n')




if __name__ == '__main__':
    main()
