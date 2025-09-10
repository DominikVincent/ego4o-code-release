#!/bin/bash

# This script is used to test the Nymeria package
python llava/ego4o/eval/test_ego4o_image_imu_batch_wo_motion.py --model_path checkpoints/imu_image_to_text_finetune_lora_bak/checkpoint-9000
