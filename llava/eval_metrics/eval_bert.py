import os
import numpy as np
import evaluate
import json

bert_metric = evaluate.load("bertscore")

base_path = '/home/jianwang/EgoMocap/work/LLaVA/vis_out/10_28_14_02_55'

pred_path = os.path.join(base_path, 'pred_text_list.json')
gt_path = os.path.join(base_path, 'gt_text_list.json')

# load json list
with open(pred_path, 'r') as f:
    pred_list = json.load(f)
with open(gt_path, 'r') as f:
    gt_list = json.load(f)

bert_result = bert_metric.compute(predictions=pred_list, references=gt_list, lang='en', verbose=False,
                                  idf=True, rescale_with_baseline=True,)

print(bert_result)
print(np.mean(bert_result['f1']))
