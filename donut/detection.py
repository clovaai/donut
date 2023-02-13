from collections import defaultdict
from typing import List, Literal, NamedTuple, Optional, Tuple, Union

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import pairwise_distances


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
    def __init__(
        self,
        input_size: Tuple[int, int],
        layer_fuse: Literal["max", "mean", "final"] = "final",
        head_fuse: Literal["max", "mean", "weighted-mean"] = "weighted-mean",
        quality_metric: Literal["std", "max"] = "std",
        topk: int = -1,
        discard_threshold: float = 0.01,
        alpha: float = 8.0,
        eps: float = 1e-6,
        const_for_numeric_stability: float = 1000.0,
        bin_th: float = 100,
        area_precision_th: float = 0.0001,
        iterations1: int = 7,
        iterations2: int = 5,
        iterations3: int = 2,
        aspct_ratio_th: float = 20.0,
        visualize: bool = False,
    ):
        """
        Detect the location of the detected tokens in the original image

        Args:
            input_size: size of the input image
            layer_fuse: method to fuse the attention maps from different layers
            head_fuse: method to fuse the attention maps from different heads
            quality_metric: method to measure the quality of the attention maps
            topk: number of topk attention maps to use
            discard_threshold: discard the attention maps with quality lower than the threshold
            alpha: parameter for the quality metric
            eps: epsilon value for numerical stability
            const_for_numeric_stability: constant for numerical stability
            bin_th: threshold for binarization
            area_precision_th: threshold for area precision
            iterations1: # iterations for the first horizontal dilation for watershed labeling
            iterations2: # iterations for the dilation to separate the foreground and background for watershed labeling
            iterations3: # iterations for the heatmap dilation
            aspct_ratio_th: threshold for aspect ratio, which removes the weird long boxes
            visualize: whether to visualize the attention maps and the detected results
        """
        self.input_size = input_size
        self.layer_fuse = layer_fuse
        self.head_fuse = head_fuse
        self.quality_metric = quality_metric
        self.topk = topk
        self.discard_threshold = discard_threshold
        self.alpha = alpha
        self.eps = eps
        self.const_for_numeric_stability = const_for_numeric_stability
        self.bin_th = bin_th
        self.area_precision_th = area_precision_th
        self.iterations1 = iterations1
        self.iterations2 = iterations2
        self.iterations3 = iterations3
        self.aspct_ratio_th = aspct_ratio_th
        self.visualize = visualize

    def __call__(
        self,
        img: Image.Image,
        cross_attentions: Optional[
            torch.FloatTensor
        ] = None,  # (num_tokens, num_layers, num_heads, num_patches)
        fields_metadata: Optional[List[FieldMetadata]] = None,
    ):
        img_array = np.array(img)
        orig_image_height, orig_image_width = img_array.shape[:2]
        longer_length = max(self.input_size)
        scale_factor = float(longer_length) / max(orig_image_width, orig_image_height)
        scaled_height, scaled_width = int(scale_factor * orig_image_height), int(
            scale_factor * orig_image_width
        )
        resized_img = cv2.resize(img_array, (scaled_width, scaled_height))

        markers, num_marker, cnt_markers, blob_centroids = self._watershed_labeling(
            resized_img,
            orig_image_width=orig_image_width,
            orig_image_height=orig_image_height,
        )
        marker_height, marker_width = markers.shape

        heatmaps, stride = self._output_original_sized_fused_heatmaps(
            cross_attentions, marker_height, marker_width
        )

        # Normalize heatmaps
        heatmaps = (
            heatmaps
            / heatmaps.max(dim=1, keepdim=True).values.max(dim=2, keepdim=True).values
        )

        # Init status variable
        overlaped_pixel_matrix = np.zeros(
            (len(fields_metadata), num_marker), dtype=np.uint32
        )

        # 1st step : calc overlap region for each field
        num_fields = len(fields_metadata)
        hmap_peak_coords = []
        bin_heatmaps = [[] for _ in range(num_fields)]
        vis_heatmaps = []
        for field_index, field in enumerate(fields_metadata):
            heatmap = heatmaps[field.att_indices].max(dim=0).values
            heatmap = (heatmap * 255.0).cpu().numpy().astype(np.uint8)

            hmap_argmax = np.unravel_index(np.argmax(heatmap), heatmap.shape)
            hmap_peak_coords.append([hmap_argmax[1], hmap_argmax[0]])

            if self.visualize:
                vis_heatmaps.append(np.copy(heatmap))

            _, bin_heatmap = cv2.threshold(heatmap, self.bin_th, 255, cv2.THRESH_BINARY)
            bin_heatmap = cv2.dilate(
                bin_heatmap, np.ones((3, 3), np.uint8), iterations=self.iterations3
            )

            # Only use dominant component within field blob
            hor_kernel = np.reshape(
                np.array([0, 0, 0, 1, 1, 1, 0, 0, 0], dtype=np.uint8), (3, 3)
            )
            dom_heatmap = cv2.dilate(bin_heatmap, hor_kernel, iterations=50)
            _, labels, stats, _ = cv2.connectedComponentsWithStats(dom_heatmap)
            sorted_label = np.argsort(stats[1:, cv2.CC_STAT_AREA])  # zero is background
            dominant_label = sorted_label[-1] + 1
            bin_heatmap[labels != dominant_label] = 0

            bin_heatmaps[field_index] = np.copy(bin_heatmap)

            # count matched text blobs
            ca_region_index_y, ca_region_index_x = np.where(bin_heatmap > 0)
            found_blob_num = list()
            for ind in zip(ca_region_index_y, ca_region_index_x):
                blob_num = markers[ind]
                if blob_num < 2:
                    continue
                overlaped_pixel_matrix[field_index, blob_num] += 1
                if not blob_num in found_blob_num:
                    found_blob_num.append(blob_num)

            # remove match count if area precision is under the threshold
            for blob_num in found_blob_num:
                if (
                    overlaped_pixel_matrix[field_index, blob_num]
                    / cnt_markers[blob_num]
                    < self.area_precision_th
                ):
                    overlaped_pixel_matrix[field_index, blob_num] = 0

        # 2nd step : argmax of text blobs
        selected_blob = [[] for _ in range(num_fields)]
        for k in range(num_marker):
            if np.max(overlaped_pixel_matrix[:, k]) > 0:
                ca_ind = np.argmax(overlaped_pixel_matrix[:, k])
                selected_blob[ca_ind].append(k)

        # 3rd step : making det boxes
        pred_bboxes = []
        pred_quads = []
        attributes = []
        field_blobs = []
        for field_index in range(num_fields):
            field_blob = np.zeros((marker_height, marker_width), dtype=np.uint8)
            attribute = 0  # 0:TextBlob, 1:AttBlob, 2:NearestTextBlob

            att_blob = np.copy(bin_heatmaps[field_index])
            num_att_blob = np.sum(att_blob > 0)

            if len(selected_blob[field_index]) == 0:  # No matched text blob
                dists = pairwise_distances(
                    [hmap_peak_coords[field_index]], blob_centroids
                )[0]
                nearest_blob = np.argsort(dists) + 1
                matched_blob = -1
                for k in range(min(3, len(nearest_blob))):
                    blob_num = nearest_blob[k]
                    overlaped_pixel_matrix[:, blob_num]
                    if np.sum(overlaped_pixel_matrix[:, blob_num]) == 0:
                        matched_blob = blob_num
                        break

                if matched_blob >= 0:
                    field_blob[markers == blob_num] = 255
                    attribute = 2
                else:
                    field_blob[att_blob == 255] = 255
                    attribute = 1
            else:  # Found text blobs
                for blob_ind in selected_blob[field_index]:
                    field_blob[markers == blob_ind] = 255

            if self.visualize:
                field_blobs.append(field_blob)

            blob_pts = np.where(field_blob == 255)
            np_contours = np.fliplr(np.array(blob_pts).transpose().reshape(-1, 2))
            rectangle = cv2.minAreaRect(np_contours)

            blob_w, blob_h = rectangle[1]
            blob_area = blob_w * blob_h

            if (
                float(min(num_att_blob, blob_area))
                / float(max(num_att_blob, blob_area) + 1e-7)
                < 0.25
            ):  # Use attention map for quad
                blob_pts = np.where(att_blob == 255)
                np_contours = np.fliplr(np.array(blob_pts).transpose().reshape(-1, 2))
                rectangle = cv2.minAreaRect(np_contours)
                attribute = 1

            quad = cv2.boxPoints(rectangle)
            quad = np.round(quad).astype(np.int32)
            x1, y1 = quad.min(axis=0).tolist()
            x2, y2 = quad.max(axis=0).tolist()

            bbox = (
                (np.array([x1, y1, x2 - x1, y2 - y1]) / scale_factor)
                .astype(np.int32)
                .tolist()
            )
            quad = (quad / scale_factor).astype(np.int32).tolist()
            x1, y1, w, h = bbox
            pred_bboxes.append([x1, y1, x1 + w, y1 + h])
            pred_quads.append(quad)
            attributes.append(attribute)

        dets = DetectionResult(
            boxes=pred_bboxes,
            quads=pred_quads,
            scores=[1] * len(fields_metadata),
            preds=fields_metadata,
            attributes=attributes,
        )
        return dets, field_blobs, markers, vis_heatmaps, bin_heatmaps

    def simple_detect(
        self,
        cross_attentions: Optional[
            torch.FloatTensor
        ] = None,  # (num_tokens, num_layers, num_heads, num_patches)
        confidences: Optional[torch.FloatTensor] = None,  # (num_tokens,)
        fields_metadata: Optional[List[FieldMetadata]] = None,
        orig_image_height: Optional[int] = None,
        orig_image_width: Optional[int] = None,
    ) -> DetectionResult:
        """Purely detect the fields in the image without any post-processing."""

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

    def _output_original_sized_fused_heatmaps(
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

    def _watershed_labeling(self, img, orig_image_width, orig_image_height):
        # binarization
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_gray = cv2.medianBlur(img_gray, 5)
        img_bin = cv2.adaptiveThreshold(
            img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 10
        )
        img_bin = cv2.bitwise_not(img_bin)

        # morphology
        hor_kernel = np.reshape(
            np.array([0, 0, 0, 1, 1, 1, 0, 0, 0], dtype=np.uint8), (3, 3)
        )
        img_morph = cv2.dilate(img_bin, hor_kernel, iterations=self.iterations1)

        # find sure_bg, sure_fg
        kernel = np.ones((3, 3), np.uint8)
        sure_bg = cv2.dilate(img_morph, kernel, iterations=self.iterations2)
        sure_fg = img_morph
        unknown = cv2.subtract(sure_bg, sure_fg)

        _, markers, stats, blob_centroids = cv2.connectedComponentsWithStats(sure_fg)
        # remove separator if the blob has a extreme aspect ratio
        for k, stat in enumerate(stats):
            x, y, w, h, area = stat

            if (
                (w / h) > self.aspct_ratio_th
                or (h / w) > self.aspct_ratio_th
                or (w * h) > (orig_image_width * orig_image_height * 0.5)
                or h > (orig_image_height * 0.5)
            ):
                markers[markers == k] = 0
                blob_centroids[k] = [-9999, -9999]

        markers = markers + 1
        markers[unknown == 255] = 0
        markers = cv2.watershed(img, markers)
        markers[markers == -1] = 0  # remove boundary marker
        num_marker = np.max(markers) + 1

        # count marker points of each blob
        cnt_markers = np.zeros((num_marker), dtype=np.uint32)
        for i in range(num_marker):
            cnt_markers[i] = np.sum(markers == i)

        if self.visualize:
            from matplotlib import pyplot as plt

            fig, ax = plt.subplots(2, 2, figsize=(5 * 2, 8 * 2))
            plt.subplot(2, 2, 1), plt.imshow(img)
            plt.subplot(2, 2, 2), plt.imshow(img_morph, "gray")
            plt.subplot(2, 2, 3), plt.imshow(sure_bg)
            plt.subplot(2, 2, 4), plt.imshow(markers)

        return markers, num_marker, cnt_markers, blob_centroids

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
                    qualities = (attn + self.eps).reshape(l, -1).std(dim=-1)
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
                qualities = (topk_attentions + self.eps).reshape(t, l, -1).std(dim=-1)
            else:
                qualities = topk_attentions.reshape(t, l, -1).max(dim=-1).values
            qualities = (qualities * self.const_for_numeric_stability) ** self.alpha
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
