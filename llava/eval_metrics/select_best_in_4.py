import os
import numpy as np
import evaluate
import json

bert_metric = evaluate.load("bertscore")

# rougel_metric = evaluate.load("rouge")

base_path = '/home/jianwang/EgoMocap/work/LLaVA/eval_out/important/test_nymeria_all10_29_00_12_29'

result_json_path = os.path.join(base_path, 'result.json')
pred_path = os.path.join(base_path, 'pred_text_list.json')
gt_path = os.path.join(base_path, 'gt_text_list.json')

with open(result_json_path, 'r') as f:
    result_data = json.load(f)

with open(pred_path, 'r') as f:
    pred_list = json.load(f)
with open(gt_path, 'r') as f:
    gt_list = json.load(f)

indices = [i for i in range(len(gt_list)) if len(gt_list[i]) > 150 and "walking" in gt_list[i]]

pred_list = [pred_list[i] for i in indices]
gt_list = [gt_list[i] for i in indices]
result_data = [result_data[i] for i in indices]

print(len(pred_list), len(gt_list), len(result_data))

# calculate the bleu and rouge score for each sample and save them in a list
bert_result = bert_metric.compute(predictions=pred_list, references=gt_list, lang='en', verbose=False,
                                  idf=True, rescale_with_baseline=True,)

# print(bert_result)

# sort the bert_result by the f1 score
bert_result_to_sort = [(bert_result['f1'][i], result_data[i]) for i in range(len(bert_result['f1']))]
bert_result_to_sort_4 = []
for i in range(len(bert_result_to_sort) - 4):
    temp_list = []
    average_val = 0
    for j in range(4):
        average_val += bert_result['f1'][i+j]
        temp_list.append(result_data[i+j])
    average_val /= 4
    bert_result_to_sort_4.append((average_val, temp_list))
bert_result_to_sort_4 = sorted(bert_result_to_sort_4, key=lambda x: x[0], reverse=True)
print(bert_result_to_sort_4[:10])

