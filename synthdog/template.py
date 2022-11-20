"""
Donut
Copyright (c) 2022-present NAVER Corp.
MIT License
"""
import json
import os
import re
from typing import Any, List

import numpy as np
from elements import Background, Document
from PIL import Image
from synthtiger import components, layers, templates


class SynthDoG(templates.Template):
    def __init__(self, config=None, split_ratio: List[float] = [0.8, 0.1, 0.1]):
        super().__init__(config)
        if config is None:
            config = {}

        self.quality = config.get("quality", [50, 95])
        self.landscape = config.get("landscape", 0.5)
        self.short_size = config.get("short_size", [720, 1024])
        self.aspect_ratio = config.get("aspect_ratio", [1, 2])
        self.background = Background(config.get("background", {}))
        self.document = Document(config.get("document", {}))
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

        # config for splits
        self.splits = ["train", "validation", "test"]
        self.split_ratio = split_ratio
        self.split_indexes = np.random.choice(3, size=10000, p=split_ratio)

    def generate(self):
        landscape = np.random.rand() < self.landscape
        short_size = np.random.randint(self.short_size[0], self.short_size[1] + 1)
        aspect_ratio = np.random.uniform(self.aspect_ratio[0], self.aspect_ratio[1])
        long_size = int(short_size * aspect_ratio)
        size = (long_size, short_size) if landscape else (short_size, long_size)

        bg_layer = self.background.generate(size)
        paper_layer, text_layers, texts = self.document.generate(size)

        document_group = layers.Group([*text_layers, paper_layer])
        document_space = np.clip(size - document_group.size, 0, None)
        document_group.left = np.random.randint(document_space[0] + 1)
        document_group.top = np.random.randint(document_space[1] + 1)
        roi = np.array(paper_layer.quad, dtype=int)

        layer = layers.Group([*document_group.layers, bg_layer]).merge()
        self.effect.apply([layer])

        image = layer.output(bbox=[0, 0, *size])
        label = " ".join(texts)
        label = label.strip()
        label = re.sub(r"\s+", " ", label)
        quality = np.random.randint(self.quality[0], self.quality[1] + 1)

        data = {
            "image": image,
            "label": label,
            "quality": quality,
            "roi": roi,
        }

        return data

    def init_save(self, root):
        if not os.path.exists(root):
            os.makedirs(root, exist_ok=True)

    def save(self, root, data, idx):
        image = data["image"]
        label = data["label"]
        quality = data["quality"]
        roi = data["roi"]

        # split
        split_idx = self.split_indexes[idx % len(self.split_indexes)]
        output_dirpath = os.path.join(root, self.splits[split_idx])

        # save image
        image_filename = f"image_{idx}.jpg"
        image_filepath = os.path.join(output_dirpath, image_filename)
        os.makedirs(os.path.dirname(image_filepath), exist_ok=True)
        image = Image.fromarray(image[..., :3].astype(np.uint8))
        image.save(image_filepath, quality=quality)

        # save metadata (gt_json)
        metadata_filename = "metadata.jsonl"
        metadata_filepath = os.path.join(output_dirpath, metadata_filename)
        os.makedirs(os.path.dirname(metadata_filepath), exist_ok=True)

        metadata = self.format_metadata(image_filename=image_filename, keys=["text_sequence"], values=[label])
        with open(metadata_filepath, "a") as fp:
            json.dump(metadata, fp, ensure_ascii=False)
            fp.write("\n")

    def end_save(self, root):
        pass

    def format_metadata(self, image_filename: str, keys: List[str], values: List[Any]):
        """
        Fit gt_parse contents to huggingface dataset's format
        keys and values, whose lengths are equal, are used to constrcut 'gt_parse' field in 'ground_truth' field
        Args:
            keys: List of task_name
            values: List of actual gt data corresponding to each task_name
        """
        assert len(keys) == len(values), "Length does not match: keys({}), values({})".format(len(keys), len(values))

        _gt_parse_v = dict()
        for k, v in zip(keys, values):
            _gt_parse_v[k] = v
        gt_parse = {"gt_parse": _gt_parse_v}
        gt_parse_str = json.dumps(gt_parse, ensure_ascii=False)
        metadata = {"file_name": image_filename, "ground_truth": gt_parse_str}
        return metadata
