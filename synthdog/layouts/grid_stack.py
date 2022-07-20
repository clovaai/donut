"""
Donut
Copyright (c) 2022-present NAVER Corp.
MIT License
"""
import numpy as np

from layouts import Grid


class GridStack:
    def __init__(self, config):
        self.text_scale = config.get("text_scale", [0.05, 0.1])
        self.max_row = config.get("max_row", 5)
        self.max_col = config.get("max_col", 3)
        self.fill = config.get("fill", [0, 1])
        self.full = config.get("full", 0)
        self.align = config.get("align", ["left", "right", "center"])
        self.stack_spacing = config.get("stack_spacing", [0, 0.05])
        self.stack_fill = config.get("stack_fill", [1, 1])
        self.stack_full = config.get("stack_full", 0)
        self._grid = Grid(
            {
                "text_scale": self.text_scale,
                "max_row": self.max_row,
                "max_col": self.max_col,
                "align": self.align,
            }
        )

    def generate(self, bbox):
        left, top, width, height = bbox

        stack_spacing = np.random.uniform(self.stack_spacing[0], self.stack_spacing[1])
        stack_spacing *= min(width, height)

        stack_full = np.random.rand() < self.stack_full
        stack_fill = np.random.uniform(self.stack_fill[0], self.stack_fill[1])
        stack_fill = 1 if stack_full else stack_fill

        full = np.random.rand() < self.full
        fill = np.random.uniform(self.fill[0], self.fill[1])
        fill = 1 if full else fill
        self._grid.fill = [fill, fill]

        layouts = []
        line = 0

        while True:
            grid_size = (width, height * stack_fill - line)
            text_scale = np.random.uniform(self.text_scale[0], self.text_scale[1])
            text_size = min(width, height) * text_scale
            text_scale = text_size / min(grid_size)
            self._grid.text_scale = [text_scale, text_scale]

            layout = self._grid.generate([left, top + line, *grid_size])
            if layout is None:
                break

            line = max(y + h - top for (_, y, _, h), _ in layout) + stack_spacing
            layouts.append(layout)

        line = max(line - stack_spacing, 0)
        space = max(height - line, 0)
        spaces = np.random.rand(len(layouts) + 1)
        spaces *= space / sum(spaces) if sum(spaces) > 0 else 0
        spaces = np.cumsum(spaces)

        for layout, space in zip(layouts, spaces):
            for bbox, _ in layout:
                x, y, w, h = bbox
                bbox[:] = [x, y + space, w, h]

        return layouts
