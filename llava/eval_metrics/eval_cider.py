import os
import numpy as np

import json

from typing import List
import evaluate
import os
import tempfile
import subprocess

from pycocoevalcap.cider.cider import CiderScorer

base_path = '/home/jianwang/EgoMocap/work/LLaVA/vis_out/10_28_03_15_34'

pred_path = os.path.join(base_path, 'pred_text_list.json')
gt_path = os.path.join(base_path, 'gt_text_list.json')

# load json list
with open(pred_path, 'r') as f:
    pred_list = json.load(f)
with open(gt_path, 'r') as f:
    gt_list = json.load(f)

_URLS = {
    "stanford-corenlp": "https://repo1.maven.org/maven2/edu/stanford/nlp/stanford-corenlp/3.4.1/stanford-corenlp-3.4.1.jar"
}

def tokenize(tokenizer_path: str, predictions: List[str], references: List[List[str]]):
    PUNCTUATIONS = [
        "''",
        "'",
        "``",
        "`",
        "-LRB-",
        "-RRB-",
        "-LCB-",
        "-RCB-",
        ".",
        "?",
        "!",
        ",",
        ":",
        "-",
        "--",
        "...",
        ";",
    ]

    cmd = [
        "java",
        "-cp",
        tokenizer_path,
        "edu.stanford.nlp.process.PTBTokenizer",
        "-preserveLines",
        "-lowerCase",
    ]

    sentences = "\n".join(
        [
            s.replace("\n", " ")
            for s in predictions + [ref for refs in references for ref in refs]
        ]
    )

    with tempfile.NamedTemporaryFile(delete=False) as f:
        f.write(sentences.encode())

    cmd.append(f.name)
    p_tokenizer = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    token_lines = p_tokenizer.communicate(input=sentences.rstrip())[0]
    token_lines = token_lines.decode()
    lines = [
        " ".join([w for w in line.rstrip().split(" ") if w not in PUNCTUATIONS])
        for line in token_lines.split("\n")
    ]

    os.remove(f.name)

    pred_size = len(predictions)
    ref_sizes = [len(ref) for ref in references]

    predictions = lines[:pred_size]
    start = pred_size
    references = []
    for size in ref_sizes:
        references.append(lines[start : start + size])
        start += size

    return predictions, references


class CIDEr(evaluate.Metric):


    def __init__(self):
        self.tokenizer_path = '/home/jianwang/EgoMocap/work/LLaVA/eval_metrics/stanford-corenlp-3.4.1.jar'

    def _compute(self, predictions, references, n=3, sigma=6.0):
        predications, references = tokenize(
            self.tokenizer_path, predictions, references
        )
        scorer = CiderScorer(n=n, sigma=sigma)
        for pred, refs in zip(predications, references):
            scorer += (pred, refs)
        score, scores = scorer.compute_score()
        return {"CIDEr": score}

cider_metric = CIDEr()
cider_result = cider_metric._compute(predictions=pred_list, references=gt_list)
print(cider_result)

