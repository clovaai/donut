"""
Donut
Copyright (c) 2022-present NAVER Corp.
MIT License
"""
import numpy as np
from synthtiger import layers


class TextBox:
    def __init__(self, config):
        self.fill = config.get("fill", [1, 1])

    def generate(self, size, text, font):
        width, height = size

        char_layers, chars = [], []
        fill = np.random.uniform(self.fill[0], self.fill[1])
        width = np.clip(width * fill, height, width)
        font = {**font, "size": int(height)}
        left, top = 0, 0

        for char in text:
            if char in "\r\n":
                continue

            char_layer = layers.TextLayer(char, **font)
            char_scale = height / char_layer.height
            char_layer.bbox = [left, top, *(char_layer.size * char_scale)]
            if char_layer.right > width:
                break

            char_layers.append(char_layer)
            chars.append(char)
            left = char_layer.right

        text = "".join(chars).strip()
        if len(char_layers) == 0 or len(text) == 0:
            return None, None

        text_layer = layers.Group(char_layers).merge()

        return text_layer, text
