"""
Python equivalent of the Kuzushiji competition metric (https://www.kaggle.com/c/kuzushiji-recognition/)
Kaggle's backend uses a C# implementation of the same metric. This version is
provided for convenience only; in the event of any discrepancies the C# implementation
is the master version.

Tested on Python 3.6 with numpy 1.16.4 and pandas 0.24.2.
"""


import argparse
import json
import multiprocessing

import numpy as np
import pandas as pd


def define_console_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sub_path", type=str)
    parser.add_argument("--solution_path", type=str)
    return parser


def score_page(preds, truth, image_id):
    """
    Scores a single page.
    Args:
        preds: prediction string of labels and center points.
        truth: ground truth string of labels and bounding boxes.
    Returns:
        True/false positive and false negative counts for the page
    """
    tp = 0
    fp = 0
    fn = 0

    truth_indices = {"label": 0, "X": 1, "Y": 2, "Width": 3, "Height": 4}
    preds_indices = {"label": 0, "X": 1, "Y": 2}

    is_truth_empty = pd.isna(truth) or truth == ""
    is_preds_empty = pd.isna(preds) or preds == ""
    if is_truth_empty and is_preds_empty:
        return {"tp": tp, "fp": fp, "fn": fn}

    if is_truth_empty:
        fp += len(preds.split(" ")) // len(preds_indices)
        return {"tp": tp, "fp": fp, "fn": fn}

    if is_preds_empty:
        fn += len(truth.split(" ")) // len(truth_indices)
        return {"tp": tp, "fp": fp, "fn": fn}

    truth = truth.split(" ")
    if len(truth) % len(truth_indices) != 0:
        raise ValueError("Malformed solution string")
    truth_label = np.array(truth[truth_indices["label"] :: len(truth_indices)])
    truth_xmin = np.array(truth[truth_indices["X"] :: len(truth_indices)]).astype(float)
    truth_ymin = np.array(truth[truth_indices["Y"] :: len(truth_indices)]).astype(float)
    truth_xmax = truth_xmin + np.array(
        truth[truth_indices["Width"] :: len(truth_indices)]
    ).astype(float)
    truth_ymax = truth_ymin + np.array(
        truth[truth_indices["Height"] :: len(truth_indices)]
    ).astype(float)

    preds = preds.split(" ")
    if len(preds) % len(preds_indices) != 0:
        raise ValueError("Malformed prediction string")
    preds_label = np.array(preds[preds_indices["label"] :: len(preds_indices)])
    preds_x = np.array(preds[preds_indices["X"] :: len(preds_indices)]).astype(float)
    preds_y = np.array(preds[preds_indices["Y"] :: len(preds_indices)]).astype(float)
    preds_unused = np.ones(len(preds_label)).astype(bool)

    matches = dict()
    for truth_idx, (xmin, xmax, ymin, ymax, label) in enumerate(
        zip(truth_xmin, truth_xmax, truth_ymin, truth_ymax, truth_label)
    ):
        # Matching = point inside box & character same & prediction not already used
        matching = (
            (xmin < preds_x)
            & (xmax > preds_x)
            & (ymin < preds_y)
            & (ymax > preds_y)
            & (preds_label == label)
            & preds_unused
        )
        if matching.sum() == 0:
            fn += 1
        else:
            tp += 1
            pred_idx = np.argmax(matching)
            preds_unused[pred_idx] = False
            matches[truth_idx] = pred_idx
    fp += preds_unused.sum()
    return {"tp": tp, "fp": fp, "fn": fn, "image_id": image_id, "matches": matches}


def calc_f1_from_counts(tps, fps, fns):
    tp = sum(tps)
    fp = sum(fps)
    fn = sum(fns)
    if (tp + fp) == 0 or (tp + fn) == 0:
        f1 = 0.0
    else:
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        if precision > 0 and recall > 0:
            f1 = (2 * precision * recall) / (precision + recall)
        else:
            f1 = 0.0
    return f1


def kuzushiji_f1(sub, solution):
    """
    Calculates the competition metric.
    Args:
        sub: submissions, as a Pandas dataframe
        solution: solution, as a Pandas dataframe
    Returns:
        f1 score
    """
    if not all(sub["image_id"].values == solution["image_id"].values):
        raise ValueError("Submission image id codes don't match solution")

    pool = multiprocessing.Pool()
    results = pool.starmap(
        score_page,
        zip(sub["labels"].values, solution["labels"].values, sub["image_id"].values),
    )
    pool.close()
    pool.join()

    tps = [x["tp"] for x in results]
    fps = [x["fp"] for x in results]
    fns = [x["fn"] for x in results]
    f1 = calc_f1_from_counts(tps, fps, fns)
    matches = {x["image_id"]: x["matches"] for x in results if "matches" in x}

    return f1, matches


def load_prediction(cleval_preds=None, sub_path=None):
    if cleval_preds is None:
        with open(sub_path) as f:
            cleval_preds = json.load(f)
    rows = []
    for image_id, preds in cleval_preds.items():
        _preds = []
        for pred in preds:
            char, xys, *_ = pred
            center_xy = np.mean(xys, axis=0)
            _preds.append(char)
            _preds.extend([str(i) for i in center_xy.astype(int).tolist()])
        rows.append({"image_id": image_id, "labels": " ".join(_preds)})
    return pd.DataFrame(rows)


def load_solution(solution_path="cleval_valid/kuzushiji_validation_gt.json"):
    with open(solution_path) as f:
        cleval_gt = json.load(f)
    rows = []
    for image_id, gts in cleval_gt.items():
        _gts = []
        for pred in gts:
            char, xys = pred
            xyxy = [*xys[0], *xys[2]]
            _gts.append(char.encode("unicode-escape").decode("utf-8"))
            _gts.extend([str(i) for i in xyxy])
        rows.append({"image_id": image_id, "labels": " ".join(_gts)})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    parser = define_console_parser()
    shell_args = parser.parse_args()
    sub = pd.read_csv(shell_args.sub_path)
    solution = pd.read_csv(shell_args.solution_path)
    sub = sub.rename(columns={"rowId": "image_id", "PredictionString": "labels"})
    solution = solution.rename(
        columns={"rowId": "image_id", "PredictionString": "labels"}
    )
    score, matches = kuzushiji_f1(sub, solution)
    print("F1 score of: {0}".format(score))
