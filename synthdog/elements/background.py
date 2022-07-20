"""
Donut
Copyright (c) 2022-present NAVER Corp.
MIT License
"""
from synthtiger import components, layers


class Background:
    def __init__(self, config):
        self.image = components.BaseTexture(**config.get("image", {}))
        self.effect = components.Iterator(
            [
                components.Switch(components.GaussianBlur()),
            ],
            **config.get("effect", {})
        )

    def generate(self, size):
        bg_layer = layers.RectLayer(size, (255, 255, 255, 255))
        self.image.apply([bg_layer])
        self.effect.apply([bg_layer])

        return bg_layer
