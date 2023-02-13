from collections import defaultdict
from typing import List, Literal, NamedTuple, Optional, Tuple, Union

import cv2
import numpy as np
import torch
import torch.nn.functional as F


class FieldMetadata(NamedTuple):
    key: Optional[str] = None
    value: Optional[str] = None
    start_index: Optional[int] = None
    end_index: Optional[int] = None
    att_indices: Optional[int] = None


class DetectionResult(NamedTuple):
    boxes: Optional[List[List[int]]] = None  # [[x1, y1, x2, y2], ...]
    quads: Optional[List[List[List[int]]]] = None  # [[[x1, y1], [x2, y2], ...], ...]
    points: Optional[List[List[int]]] = None  # [[x, y], ...]]
    scores: Optional[List[float]] = None
    preds: Optional[Union[List[str], List[FieldMetadata]]] = None
    attributes: Optional[List[int]] = None  # 0:textblob, 1:attblob

    def to_format_for_map_calculation(self, label_mapper=ord) -> dict:
        d = {
            "boxes": torch.tensor(self.boxes),
            "scores": torch.tensor(self.scores),
            "labels": torch.tensor([label_mapper(p) for p in self.preds]),
        }
        assert len(d["boxes"]) == len(d["scores"]) == len(d["labels"])
        return d

    def to_format_for_fcleval_calculation(self) -> dict:
        gt = defaultdict(list)
        if self.quads is not None:
            for quad, pred in zip(self.quads, self.preds):
                gt[pred.key].append(
                    {
                        "text": pred.value,
                        "boundingBoxes": quad,
                    }
                )
        else:
            for box, pred in zip(self.boxes, self.preds):
                x1, y1, x2, y2 = box
                quad = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
                gt[pred.key].append(
                    {
                        "text": pred.value,
                        # "quad": [x1, y1, x2, y1, x2, y2, x1, y2],
                        "boundingBoxes": quad,
                    }
                )
        return gt

    def to_list_of_dict(self) -> dict:
        ret = list()
        for k in range(len(self.boxes)):
            ret.append(
                {
                    "box": self.boxes[k],
                    "quad": self.quads[k],
                    "key": self.preds[k].key,
                    "value": self.preds[k].value,
                    "attribute": self.attributes[k],
                }
            )
        return ret

    def __len__(self):
        return (
            len(self.boxes) or len(self.points) or len(self.scores) or len(self.preds)
        )


class Detector:
    EPS = 1e-6
    CONST_FOR_NUMERIC_STABILITY = 1000

    def __init__(
        self,
        input_size: Tuple[int, int],
        layer_fuse: Literal["max", "mean", "final"] = "final",
        head_fuse: Literal["max", "mean", "weighted-mean"] = "weighted-mean",
        quality_metric: Literal["std", "max"] = "std",
        topk: int = -1,
        discard_threshold: float = 0.01,
        alpha: float = 8.0,
    ):
        """
        Detect the location of the detected tokens in the original image

        Args:
            cross_attentions: (num_tokens, num_layers, num_heads, 1, num_patches)
            confidences: (num_tokens,)
            pred: predicted token sequence
            orig_image_height: original image height
            orig_image_width: original image width
            layer_fuse: method to fuse the attention maps from different layers
            head_fuse: method to fuse the attention maps from different heads
            quality_metric: method to measure the quality of the attention maps
            topk: number of topk attention maps to use
            discard_threshold: discard the attention maps with quality lower than the threshold
        """
        self.input_size = input_size
        self.layer_fuse = layer_fuse
        self.head_fuse = head_fuse
        self.quality_metric = quality_metric
        self.topk = topk
        self.discard_threshold = discard_threshold
        self.alpha = alpha

    def __call__(
        self,
        cross_attentions: Optional[
            torch.FloatTensor
        ] = None,  # (num_tokens, num_layers, num_heads, num_patches)
        confidences: Optional[torch.FloatTensor] = None,  # (num_tokens,)
        fields_metadata: Optional[List[FieldMetadata]] = None,
        orig_image_height: Optional[int] = None,
        orig_image_width: Optional[int] = None,
    ) -> DetectionResult:

        cross_attentions = cross_attentions.clone()
        cross_attentions = self._fuse_layers(
            cross_attentions
        )  # (num_tokens - 1, num_heads, num_patches)
        heatmaps, stride = self._reshape2d_and_resize(cross_attentions)
        heatmaps[heatmaps < self.discard_threshold] = 0

        pred_bboxes = []
        pred_points = []
        pred_scores = []
        pred_fields = []

        for field in fields_metadata:
            fwd = heatmaps[field.start_index : field.end_index].mean(
                dim=0, keepdim=True
            )
            heatmap = self._fuse_heads(fwd).squeeze(0)
            heatmap = (heatmap * 255.0).numpy().astype(np.uint8)

            bbox, point, det_conf = self._detect_bbox_and_point(heatmap)
            if bbox is None:
                continue
            score = (
                confidences[field.start_index : field.end_index].mean() * det_conf
            ).item()
            bbox, point = self._back_to_orig_coordinate(
                bbox=bbox,
                point=point,
                orig_image_height=orig_image_height,
                orig_image_width=orig_image_width,
                stride=stride,
            )

            x1, y1, x2, y2 = bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]
            pred_bboxes.append([x1, y1, x2, y2])
            pred_points.append(point)
            pred_scores.append(score)
            pred_fields.append(field)

        return DetectionResult(
            boxes=pred_bboxes,
            points=pred_points,
            scores=pred_scores,
            preds=pred_fields,
        )

    def output_original_sized_fused_heatmaps(
        self,
        cross_attentions: torch.FloatTensor,  # (num_tokens, num_layers, num_heads, num_patches)
        orig_image_height: int,
        orig_image_width: int,
    ):
        cross_attentions = self._fuse_layers(
            cross_attentions
        )  # (num_tokens - 1, num_heads, num_patches)
        heatmaps, stride = self._reshape2d_and_resize(cross_attentions)

        heatmaps = self._fuse_heads(heatmaps).unsqueeze(0)

        # back to original image size
        # 1. back to input_size
        heatmaps = F.interpolate(
            heatmaps,
            size=self.input_size,
            mode="bilinear",
            align_corners=False,
        )
        # 2. unpad
        padding = self._calc_padding(orig_image_height, orig_image_width)
        heatmaps = heatmaps[
            ...,
            padding[1] : heatmaps.shape[-2] - padding[3],
            padding[0] : heatmaps.shape[-1] - padding[2],
        ]
        # 3. back to original image size
        heatmaps = F.interpolate(
            heatmaps,
            size=(orig_image_height, orig_image_width),
            mode="bilinear",
            align_corners=False,
        )
        heatmaps = heatmaps.squeeze(0)

        return heatmaps, stride

    def _fuse_layers(self, attentions):
        # final token is for the end of sentence token, so we discard it
        if self.layer_fuse == "max":
            attentions = attentions[:-1].max(dim=1).values
        elif self.layer_fuse == "mean":
            attentions = attentions[:-1].mean(dim=1)
        elif self.layer_fuse == "final":
            attentions = attentions[:-1, -1]
        return attentions

    def _fuse_heads(self, attentions):
        if self.topk == -1:
            topk_attentions = attentions
        else:
            topk_attentions = []
            for attn in attentions:
                l = attn.shape[0]
                if self.quality_metric == "std":
                    qualities = (attn + self.EPS).reshape(l, -1).std(dim=-1)
                elif self.quality_metric == "max":
                    qualities = attn.reshape(l, -1).max(dim=-1).values
                else:
                    raise NotImplementedError

                topk_attentions.append(
                    attn[qualities.topk(self.topk, largest=True).indices].unsqueeze(0)
                )
            topk_attentions = torch.cat(topk_attentions, dim=0)

        if self.head_fuse == "mean":
            fused_attentions = topk_attentions.mean(dim=1)
        elif self.head_fuse == "max":
            fused_attentions = topk_attentions.max(dim=1).values
        elif self.head_fuse == "weighted-mean":
            t, l = topk_attentions.shape[:2]
            if self.quality_metric == "std":
                qualities = (topk_attentions + self.EPS).reshape(t, l, -1).std(dim=-1)
            else:
                qualities = topk_attentions.reshape(t, l, -1).max(dim=-1).values
            qualities = (qualities * self.CONST_FOR_NUMERIC_STABILITY) ** self.alpha
            fused_attentions = (
                topk_attentions * qualities.unsqueeze(-1).unsqueeze(-1)
            ).sum(dim=1) / qualities.sum(dim=1).unsqueeze(-1).unsqueeze(-1)
        else:
            fused_attentions = attentions[:, int(self.head_fuse), :]

        return fused_attentions

    def _detect_bbox_and_point(self, hmap):
        _, _, stats, centroids = cv2.connectedComponentsWithStats(hmap)
        max_bbox = None
        max_centroids = None
        max_conf = -1
        for stat, centroid in zip(stats[1:], centroids[1:]):
            x, y, w, h, _ = stat
            conf = hmap[y : y + h, x : x + w].sum() / 255.0
            if conf > max_conf:
                max_conf = conf
                max_bbox = [x, y, w, h]
                max_centroids = centroid
        return max_bbox, max_centroids, max_conf

    def _back_to_orig_coordinate(
        self,
        bbox: Optional[List[int]] = None,  # [x, y, w, h]
        point: Optional[List[int]] = None,  # [x, y]
        quad: Optional[
            List[List[int]]
        ] = None,  # [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
        orig_image_height: Optional[int] = None,
        orig_image_width: Optional[int] = None,
        stride: Optional[int] = None,
    ):
        padding = self._calc_padding(orig_image_height, orig_image_width)
        input_height, input_width = self.input_size

        w_ratio = orig_image_width / (input_width - padding[0] - padding[2])
        h_ratio = orig_image_height / (input_height - padding[1] - padding[3])

        outputs = ()
        if bbox is not None:
            bbox = np.array(bbox) * stride
            bbox[0] = max(bbox[0] - padding[0], 0)
            bbox[1] = max(bbox[1] - padding[1], 0)
            bbox[[0, 2]] = bbox[[0, 2]] * w_ratio
            bbox[[1, 3]] = bbox[[1, 3]] * h_ratio
            bbox = np.round(bbox).astype(int).tolist()
            outputs += (bbox,)

        if point is not None:
            point = np.array(point) * stride
            point[0] = max(point[0] - padding[0], 0)
            point[1] = max(point[1] - padding[1], 0)
            point[0] = point[0] * w_ratio
            point[1] = point[1] * h_ratio
            point = np.round(point).astype(int).tolist()
            outputs += (point,)

        if quad is not None:
            # quad = np.array(quad) * stride
            for xy in quad:
                xy[0] = round(max(xy[0] * stride - padding[0], 0) * w_ratio)
                xy[1] = round(max(xy[1] * stride - padding[1], 0) * h_ratio)
            outputs += (quad,)

        return outputs

    def _calc_padding(self, orig_image_height, orig_image_width):
        input_height, input_width = self.input_size

        hratio = orig_image_height / input_height
        wratio = orig_image_width / input_width

        if hratio > wratio:
            delta_width = input_width - round(orig_image_width / hratio)
            delta_height = 0
        else:
            delta_height = input_height - int(orig_image_height / wratio)
            delta_width = 0

        pad_width = delta_width // 2
        pad_height = delta_height // 2
        padding = (
            pad_width,
            pad_height,
            delta_width - pad_width,
            delta_height - pad_height,
        )
        return padding

    def _reshape2d_and_resize(self, cross_attentions):
        input_height, input_width = self.input_size
        aspect_ratio = input_height / input_width
        num_patches = cross_attentions.shape[-1]
        tw = int((num_patches / aspect_ratio) ** 0.5)
        th = int(tw * aspect_ratio)
        stride = int(input_width / tw)

        all_heatmaps = cross_attentions.reshape(
            cross_attentions.shape[0], -1, th, tw
        )  # (num_tokens - 1, th, tw)

        return all_heatmaps, stride
