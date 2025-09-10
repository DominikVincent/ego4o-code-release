import json
import pdb
from copy import deepcopy
from typing import Dict, Optional, Union, Tuple, List

import numpy as np
import torch
from mmcv.transforms import BaseTransform
# from mmpose.registry import TRANSFORMS
from mmengine.registry import TRANSFORMS

from mmpose.codecs import *  # noqa: F401, F403

# @TRANSFORMS.register_module()
# class AddGeneratedText(BaseTransform):
#     def __init__(self, text_json_file):
#         # read the text json file
#         self.text_json_file = text_json_file
#         self.text_data = None
#         with open(text_json_file) as f:
#             self.text_data = json.load(f)
#
#
#     def transform(self, results: dict) -> dict:
#         # get the motion id
#         motion_id = results['motion_id']
#         # get the text data
#         text_data = self.text_data[motion_id]['text']
#         motion_file = self.text_data[motion_id]['motion_file']
#         assert motion_file == results['motion_file']
#         # add the text data to the results
#         results['text'] = text_data
#         return results

@TRANSFORMS.register_module()
class AddGeneratedText(BaseTransform):
    def read_text_json_file(self, text_json_file):
        with open(text_json_file) as f:
            data_list = json.load(f)
        for data_item in data_list:
            motion_id = data_item['motion_id']
            self.text_data[motion_id] = data_item

    def __init__(self, text_json_file):
        # read the text json file
        self.text_data = {}
        if type(text_json_file) == list:
            for file in text_json_file:
                self.read_text_json_file(file)
        else:
            self.read_text_json_file(text_json_file)



    def transform(self, results: dict) -> dict:
        # get the motion id
        motion_id = results['motion_id']
        # get the text data
        text_data = self.text_data[motion_id]['pred_text']
        motion_file = self.text_data[motion_id]['motion_file']
        assert motion_file == results['motion_file']
        # add the text data to the results
        results['text'] = text_data
        return results