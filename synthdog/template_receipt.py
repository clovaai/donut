"""
Donut / SynthDoG-VN
MIT License

Template synthtiger sinh ảnh hoá đơn bán lẻ Việt Nam.

    synthtiger -o ./outputs/VNReceipt -c 100 -w 4 -v \
        template_receipt.py SynthVNReceipt config_vi_receipt.yaml

Khác template SynthDoG gốc:
  * nội dung có cấu trúc -> nhãn `gt_parse` lồng nhau (trích xuất thông tin),
    hoặc `text_sequence` (đọc trơn) tuỳ `label_format`;
  * mỗi trường vẽ bằng một TextLayer thay vì từng ký tự -> nhanh hơn nhiều;
  * có `CurlWarp` làm cong giấy mà vẫn map lại được toạ độ, nên metadata kèm
    theo polygon 4 điểm cho từng trường.
"""
import json
import os
import re
from typing import List

import numpy as np
from PIL import Image
from synthtiger import components, layers, templates

from elements import Background, Paper
from elements.receipt import Receipt
from elements.warp import CurlWarp


class SynthVNReceipt(templates.Template):
    def __init__(self, config=None, split_ratio: List[float] = [0.8, 0.1, 0.1]):
        super().__init__(config)
        config = config or {}

        self.quality = config.get("quality", [50, 95])
        self.short_size = config.get("short_size", [720, 1024])
        self.canvas_fill = config.get("canvas_fill", [0.55, 0.95])
        self.canvas_aspect = config.get("canvas_aspect", [1.0, 1.9])
        self.label_format = config.get("label_format", "parse")

        self.background = Background(config.get("background", {}))
        self.paper = Paper(config.get("paper", {}))
        self.receipt = Receipt(config.get("receipt", {}))
        self.curl = CurlWarp(config.get("curl", {}))

        self.doc_effect = components.Iterator(
            [
                components.Switch(components.ElasticDistortion()),
                components.Switch(components.AdditiveGaussianNoise()),
                components.Switch(components.Perspective()),
            ],
            **config.get("doc_effect", {}),
        )
        self.effect = components.Iterator(
            [
                components.Switch(components.RGB()),
                components.Switch(components.Shadow()),
                components.Switch(components.Contrast()),
                components.Switch(components.Brightness()),
                components.Switch(components.MotionBlur()),
                components.Switch(components.GaussianBlur()),
            ],
            **config.get("effect", {}),
        )

        self.splits = ["train", "validation", "test"]
        self.split_indexes = np.random.choice(3, size=10000, p=split_ratio)

    # ------------------------------------------------------------------

    def generate(self):
        out = self.receipt.generate()
        text_layers, fields = out["text_layers"], out["fields"]
        width, height = out["size"]

        paper_layer = self.paper.generate((width, height))
        self.doc_effect.apply([*text_layers, paper_layer])

        # gộp giấy + chữ thành một ảnh, đổi quad về hệ toạ độ của ảnh đó
        doc_group = layers.Group([*text_layers, paper_layer])
        origin = doc_group.topleft
        quads = np.array([layer.quad for layer in text_layers], dtype=np.float32) - origin
        doc_image = doc_group.output()

        # cong giấy (ảnh được pad, quad đã cộng offset pad)
        doc_image, quads = self.curl.apply(doc_image, quads)
        doc_layer = layers.Layer(doc_image)
        dw, dh = doc_layer.size

        # khung ảnh bao quanh tờ hoá đơn
        fill = np.random.uniform(*self.canvas_fill)
        aspect = np.random.uniform(*self.canvas_aspect)
        canvas_h = int(dh / fill)
        canvas_w = int(max(dw / fill, canvas_h / aspect))
        canvas = (canvas_w, canvas_h)

        bg_layer = self.background.generate(canvas)

        left = np.random.randint(max(canvas_w - int(dw), 0) + 1)
        top = np.random.randint(max(canvas_h - int(dh), 0) + 1)
        doc_layer.left, doc_layer.top = left, top
        quads = quads + (left, top)

        merged = layers.Group([doc_layer, bg_layer]).merge()
        self.effect.apply([merged])
        image = merged.output(bbox=[0, 0, *canvas])

        # thu nhỏ về kích thước mục tiêu (luôn là downscale -> chữ vẫn nét)
        short = np.random.randint(self.short_size[0], self.short_size[1] + 1)
        scale = short / min(canvas)
        if scale < 1.0:
            new_size = (max(int(canvas_w * scale), 1), max(int(canvas_h * scale), 1))
            image = np.array(
                Image.fromarray(image[..., :3].astype(np.uint8)).resize(new_size, Image.LANCZOS),
                dtype=np.float32,
            )
            quads = quads * scale
        else:
            image = image[..., :3]

        boxes = [
            {"kind": f["kind"], "text": f["text"], "quad": np.round(q, 1).tolist()}
            for f, q in zip(fields, quads)
            if f["kind"] != "sep"
        ]

        return {
            "image": image,
            "gt_parse": Receipt.to_gt_parse(out["data"]),
            "text_sequence": re.sub(r"\s+", " ", Receipt.to_text_sequence(fields)).strip(),
            "boxes": boxes,
            "quality": int(np.random.randint(self.quality[0], self.quality[1] + 1)),
        }

    # ------------------------------------------------------------------

    def init_save(self, root):
        os.makedirs(root, exist_ok=True)

    def save(self, root, data, idx):
        split_idx = self.split_indexes[idx % len(self.split_indexes)]
        output_dirpath = os.path.join(root, self.splits[split_idx])
        os.makedirs(output_dirpath, exist_ok=True)

        image_filename = f"image_{idx}.jpg"
        Image.fromarray(data["image"].astype(np.uint8)).save(
            os.path.join(output_dirpath, image_filename), quality=data["quality"]
        )

        if self.label_format == "text":
            gt_parse = {"text_sequence": data["text_sequence"]}
        else:
            gt_parse = data["gt_parse"]

        metadata = {
            "file_name": image_filename,
            "ground_truth": json.dumps({"gt_parse": gt_parse}, ensure_ascii=False),
            # Donut bỏ qua các khoá lạ trong ground_truth, nên box để riêng ở đây
            "boxes": data["boxes"],
        }
        with open(os.path.join(output_dirpath, "metadata.jsonl"), "a", encoding="utf-8") as fp:
            json.dump(metadata, fp, ensure_ascii=False)
            fp.write("\n")

    def end_save(self, root):
        pass
