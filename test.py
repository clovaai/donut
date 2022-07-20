"""
Donut
Copyright (c) 2022-present NAVER Corp.
MIT License
"""
import argparse
import json
import os
import re
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm

from donut import DonutModel, JSONParseEvaluator, load_json, save_json


def test(args):
    pretrained_model = DonutModel.from_pretrained(args.pretrained_path)

    if torch.cuda.is_available():
        pretrained_model.half()
        pretrained_model.to("cuda")
    else:
        pretrained_model.encoder.to(torch.bfloat16)

    pretrained_model.eval()

    if args.save_path:
        os.makedirs(os.path.dirname(args.save_path), exist_ok=True)

    output_list = []
    accs = []

    dataset = load_dataset(args.dataset_name_or_path, split=args.split)

    for idx, sample in tqdm(enumerate(dataset), total=len(dataset)):
        ground_truth = json.loads(sample["ground_truth"])

        if args.task_name == "docvqa":
            output = pretrained_model.inference(
                image=sample["image"],
                prompt=f"<s_{args.task_name}><s_question>{ground_truth["gt_parses"][0]['question'].lower()}</s_question><s_answer>",
            )["predictions"][0]
        else:
            output = pretrained_model.inference(image=sample["image"], prompt=f"<s_{args.task_name}>")["predictions"][0]

        if args.task_name == "rvlcdip":
            gt = ground_truth["gt_parse"]
            score = float(output["class"] == gt["class"])
        elif args.task_name == "docvqa":
            score = 0.0  # note: docvqa is evaluated on the official website
        else:
            gt = ground_truth["gt_parse"]
            evaluator = JSONParseEvaluator()
            score = evaluator.cal_acc(output, gt)

        accs.append(score)

        output_list.append(output)

    scores = {"accuracies": accs, "mean_accuracy": np.mean(accs)}
    print(scores, f"length : {len(accs)}")

    if args.save_path:
        scores["predictions"] = output_list
        save_json(args.save_path, scores)

    return output_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_path", type=str)
    parser.add_argument("--dataset_name_or_path", type=str)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--task_name", type=str, default=None)
    parser.add_argument("--save_path", type=str, default=None)
    args, left_argv = parser.parse_known_args()

    if args.task_name is None:
        args.task_name = os.path.basename(args.dataset_name_or_path)

    predicts = test(args)
