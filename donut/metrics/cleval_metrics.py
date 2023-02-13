import argparse
import codecs
import json
from collections import defaultdict

import numpy as np
from cleval import CLEvalMetric


def calc_kuzushiji_cleval(pred_json, gt_json, ap_constraint=0.5):
    with open(pred_json, "r") as jsonFile:
        preds = json.load(jsonFile)
    with open(gt_json, "r") as jsonFile:
        gts = json.load(jsonFile)

    metircs = CLEvalMetric(ap_constraint=ap_constraint)
    for img_name in gts:
        pred_texts = []
        pred_coords = []
        gt_texts = []
        gt_coords = []
        for pred_text, pred_coord, *_ in preds[img_name]:
            if pred_text.startswith("\\"):
                pred_text = codecs.decode(pred_text, "unicode_escape")
            pred_texts.append(pred_text)
            pred_coords.append(pred_coord)
        for gt_text, gt_coord in gts[img_name]:
            gt_texts.append(gt_text)
            gt_coords.append(gt_coord)
        pred_coords = np.array(pred_coords).astype(np.int64)
        gt_coords = np.array(gt_coords).astype(np.int64)
        metircs(
            pred_coords.reshape(-1, 8), gt_coords.reshape(-1, 8), pred_texts, gt_texts
        )
    res = metircs.compute()
    det_r, det_p, det_h, e2e_r, e2e_p, e2e_h = res
    result = {
        "det r": det_r.item(),
        "det p": det_p.item(),
        "det h": det_h.item(),
        "e2e r": e2e_r.item(),
        "e2e p": e2e_p.item(),
        "e2e h": e2e_h.item(),
    }
    return result


def calc_cleval(preds: list, gts: list, ap_constraint: float = 0.5) -> dict:
    """Calculate CLEval

    Args:
        preds (list): Predicted results.
        gts (list): Ground truth.
        ap_constraint (float): Area precision threshold (default: 0.5).

    Returns:
        result (dict): detection and end2end results.

    Note:
        0. Please check `notebook/calc_cleval.ipynb` to understad the practical usage.
        1. preds and gts format
            ```
            [
                [
                    {
                        "text": text1(str),
                        "quad": [x1, y1, x2, y2, x3, y3, x4, y4]
                    },
                    {
                        "text": text2(str),
                        "quad": [x1, y1, x2, y2, x3, y3, x4, y4]
                    },
                    ...
                ],
                ...
            ]
            ```

        2. output results format
            ```
            results = {
                    "det r": float,
                    "det p": float,
                    "det h": float,
                    "e2e r": float,
                    "e2e p": float,
                    "e2e h": float
                }
            ```
    """
    assert len(preds) == len(gts)
    assert ap_constraint >= 0.0 and ap_constraint <= 1.0

    metrics = CLEvalMetric(ap_constraint=ap_constraint)
    for pred_results, gt_annos in zip(preds, gts):
        gt_text_list, gt_quad_list = list(), list()
        pred_text_list, pred_quad_list = list(), list()

        # GT
        for anno in gt_annos:
            text = anno["text"]
            quad = anno["quad"]
            gt_text_list.append(text)
            gt_quad_list.append(quad)

        # Pred
        for pred in pred_results:
            text = pred["text"]
            # for kuzushiji
            if text.startswith("\\"):
                text = codecs.decode(text, "unicode_escape")

            quad = pred["quad"]
            pred_text_list.append(text)
            pred_quad_list.append(quad)

        pred_quads = np.array(pred_quad_list).astype(np.int64)
        gt_quads = np.array(gt_quad_list).astype(np.int64)
        metrics(
            pred_quads.reshape(-1, 8),
            gt_quads.reshape(-1, 8),
            pred_text_list,
            gt_text_list,
        )

    res = metrics.compute()
    det_r, det_p, det_h, e2e_r, e2e_p, e2e_h = res
    result = {
        "det r": float(det_r),
        "det p": float(det_p),
        "det h": float(det_h),
        "e2e r": float(e2e_r),
        "e2e p": float(e2e_p),
        "e2e h": float(e2e_h),
    }
    return result


def calc_category_aware_cleval(
    preds: list,
    gts: list,
    ap_constraint: float = 0.5,
    avg_denominator: str = "char_number",
) -> dict:
    """Calculate category-aware CLEval

    Args:
        preds (list): Predicted results.
        gts (list): Ground truth.
        ap_constraint (float): Area precision threshold (default: 0.5).
        avg_denominator (str): Reduction method. Allowed "char_number" or "sample_number".

    Returns:
        results (dict): The results. We explain its format in Note.

    Note:
        0. Please check `notebook/calc_category_cleval.ipynb` to understad the practical usage.
        1. preds and gts format
            ```
            [
                {
                    "menu.nm": [
                        {
                            "text": text1(str),
                            "quad": [x1, y1, x2, y2, x3, y3, x4, y4]
                        },
                        {
                            "text": text2(str),
                            "quad": [x1, y1, x2, y2, x3, y3, x4, y4]
                        }
                    ],
                    "menu.num": [
                        {
                            "text": text1(str),
                            "quad": [x1, y1, x2, y2, x3, y3, x4, y4]
                        }
                    ],
                    ...
                },
                ...
            ]
            ```
            Note that field(word)-level text results in field(word)-level CLEval.

        2. output results format
            2.a. avg_denominator == "char_number"
            ```
            results = {
                "avg": {
                    "det r": float,
                    ...
                    "e2e h": float
                },
                ...,
                "total.total_price": {
                    "det r": float,
                    ...
                    "e2e h": float
                }
            }
            ```
            2.b. avg_denominator == "sample_number"
            ```
            results = {
                "avg": {
                    "det r": float,
                    ...
                    "e2e h": float
                }
            }
            ```
    """
    assert len(preds) == len(gts)
    assert ap_constraint >= 0.0 and ap_constraint <= 1.0
    assert avg_denominator in ("char_number", "sample_number")
    div_by_char_num = avg_denominator == "char_number"

    # NOTE: When div_by_char_num is True, the denominator accumulation is done by CLEvalMetric class.
    # If False, we keep each sample results and accumulate them at last.
    if div_by_char_num:
        per_category_metric = {}
    else:
        per_sample_result = defaultdict(list)

    for pred_result, gt_anno in zip(preds, gts):
        gt_text_list = defaultdict(list)
        gt_quad_list = defaultdict(list)
        pred_text_list = defaultdict(list)
        pred_quad_list = defaultdict(list)

        # GT
        for field, annos in gt_anno.items():
            for anno in annos:
                text = anno["text"]
                quad = anno["quad"]
                gt_text_list[field].append(text)
                gt_quad_list[field].append(quad)

        # Pred
        for field, preds in pred_result.items():
            for pred in preds:
                text = pred["text"]
                quad = pred["quad"]
                pred_text_list[field].append(text)
                pred_quad_list[field].append(quad)

        all_categories = set(list(gt_quad_list.keys()) + list(pred_quad_list.keys()))

        for cat in all_categories:
            if len(pred_quad_list[cat]) == 0:
                pred_quad_list[cat] = [-1] * 8  # set invalid cordinate value
                pred_text_list[cat] = "xxxxxxxxxxxxxxxxxxx"  # set invalid text
            if len(gt_quad_list[cat]) == 0:
                gt_quad_list[cat] = [-1] * 8
                gt_text_list[cat] = "xxxxxxxxxxxxxxxxxxx"

        pred_quads = {
            cat: np.array(pred_quad_list[cat]).astype(np.int64)
            for cat in all_categories
        }
        gt_quads = {
            cat: np.array(gt_quad_list[cat]).astype(np.int64) for cat in all_categories
        }

        # NOTE: When div_by_char_num is False, we calculate CLEval score for each sample.
        if not div_by_char_num:
            per_category_metric = {}

        for cat in all_categories:
            if per_category_metric.get(cat) is None:
                per_category_metric[cat] = CLEvalMetric(ap_constraint=ap_constraint)

        sample_scores = defaultdict(list)
        for cat in all_categories:
            per_category_metric[cat](
                pred_quads[cat].reshape(-1, 8),
                gt_quads[cat].reshape(-1, 8),
                pred_text_list[cat],
                gt_text_list[cat],
            )

            if not div_by_char_num:
                res = per_category_metric[cat].compute()
                det_r, det_p, det_h, e2e_r, e2e_p, e2e_h = res
                sample_scores["det r"].append(float(det_r))
                sample_scores["det p"].append(float(det_p))
                sample_scores["det h"].append(float(det_h))
                sample_scores["e2e r"].append(float(e2e_r))
                sample_scores["e2e p"].append(float(e2e_p))
                sample_scores["e2e h"].append(float(e2e_h))

        if not div_by_char_num:
            for metric, scores in sample_scores.items():
                mean_score = np.mean(np.array(scores))
                per_sample_result[metric].append(mean_score)

    if div_by_char_num:
        results = defaultdict(dict)
        avg_result = defaultdict(list)
        for cat in all_categories:
            res = per_category_metric[cat].compute()
            det_r, det_p, det_h, e2e_r, e2e_p, e2e_h = res
            result = {
                "det r": float(det_r),
                "det p": float(det_p),
                "det h": float(det_h),
                "e2e r": float(e2e_r),
                "e2e p": float(e2e_p),
                "e2e h": float(e2e_h),
            }
            results[field] = result

            avg_result["det r"].append(float(det_r))
            avg_result["det p"].append(float(det_p))
            avg_result["det h"].append(float(det_h))
            avg_result["e2e r"].append(float(e2e_r))
            avg_result["e2e p"].append(float(e2e_p))
            avg_result["e2e h"].append(float(e2e_h))

        avg_dict = {}
        for metric, scores in avg_result.items():
            avg_dict[metric] = np.mean(np.array(scores))
        results["avg"] = avg_dict
    else:
        avg_dict = {}
        for metric, scores in per_sample_result.items():
            avg_dict[metric] = np.mean(np.array(scores))
        results = {"avg": avg_dict}

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--preds_json", type=str)
    parser.add_argument("--gts_json", type=str)
    parser.add_argument("--ap_constraint", type=float, default=0.5)

    args = parser.parse_args()

    result = calc_kuzushiji_cleval(args.preds_json, args.gts_json, args.ap_constraint)
    for key in result:
        print(f"{key}: {result[key]:.4f}")
