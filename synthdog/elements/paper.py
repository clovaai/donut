"""
Donut
Copyright (c) 2022-present NAVER Corp.
MIT License
"""
from synthtiger import components, layers


class Paper:
    def __init__(self, config):
        self.image = components.BaseTexture(**config.get("image", {}))

    def generate(self, size):
        paper_layer = layers.RectLayer(size, (255, 255, 255, 255))
        self.image.apply([paper_layer])

        return paper_layer
