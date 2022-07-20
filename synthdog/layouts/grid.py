"""
Donut
Copyright (c) 2022-present NAVER Corp.
MIT License
"""
import numpy as np


class Grid:
    def __init__(self, config):
        self.text_scale = config.get("text_scale", [0.05, 0.1])
        self.max_row = config.get("max_row", 5)
        self.max_col = config.get("max_col", 3)
        self.fill = config.get("fill", [0, 1])
        self.full = config.get("full", 0)
        self.align = config.get("align", ["left", "right", "center"])

    def generate(self, bbox):
        left, top, width, height = bbox

        text_scale = np.random.uniform(self.text_scale[0], self.text_scale[1])
        text_size = min(width, height) * text_scale
        grids = np.random.permutation(self.max_row * self.max_col)

        for grid in grids:
            row = grid // self.max_col + 1
            col = grid % self.max_col + 1
            if text_size * (col * 2 - 1) <= width and text_size * row <= height:
                break
        else:
            return None

        bound = max(1 - text_size / width * (col - 1), 0)
        full = np.random.rand() < self.full
        fill = np.random.uniform(self.fill[0], self.fill[1])
        fill = 1 if full else fill
        fill = np.clip(fill, 0, bound)

        padding = np.random.randint(4) if col > 1 else np.random.randint(1, 4)
        padding = (bool(padding // 2), bool(padding % 2))

        weights = np.zeros(col * 2 + 1)
        weights[1:-1] = text_size / width
        probs = 1 - np.random.rand(col * 2 + 1)
        probs[0] = 0 if not padding[0] else probs[0]
        probs[-1] = 0 if not padding[-1] else probs[-1]
        probs[1::2] *= max(fill - sum(weights[1::2]), 0) / sum(probs[1::2])
        probs[::2] *= max(1 - fill - sum(weights[::2]), 0) / sum(probs[::2])
        weights += probs

        widths = [width * weights[c] for c in range(col * 2 + 1)]
        heights = [text_size for _ in range(row)]

        xs = np.cumsum([0] + widths)
        ys = np.cumsum([0] + heights)

        layout = []

        for c in range(col):
            align = self.align[np.random.randint(len(self.align))]

            for r in range(row):
                x, y = xs[c * 2 + 1], ys[r]
                w, h = xs[c * 2 + 2] - x, ys[r + 1] - y
                bbox = [left + x, top + y, w, h]
                layout.append((bbox, align))

        return layout
