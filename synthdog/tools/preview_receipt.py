"""
Donut / SynthDoG-VN
MIT License

Xem trước hoá đơn và kiểm tra box có bám đúng chữ sau khi giấy bị cong hay không.

    python tools/preview_receipt.py --count 4 --out /tmp/preview --boxes

Chạy từ thư mục `synthdog/` (đường dẫn trong config là tương đối).
"""
import argparse
import os
import sys

import numpy as np
import yaml
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from template_receipt import SynthVNReceipt  # noqa: E402

COLORS = {
    "store": (220, 40, 40),
    "menu": (40, 160, 60),
    "total": (40, 90, 230),
    "meta": (200, 140, 0),
    "footer": (150, 60, 200),
}


def color_of(kind):
    return COLORS.get(kind.split(".")[0], (120, 120, 120))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config_vi_receipt.yaml")
    ap.add_argument("--count", type=int, default=4)
    ap.add_argument("--out", default="/tmp/preview")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--boxes", action="store_true", help="vẽ polygon của từng trường")
    ap.add_argument("--clean", action="store_true", help="tắt hết hiệu ứng ảnh")
    args = ap.parse_args()

    config = yaml.safe_load(open(args.config, encoding="utf-8"))
    if args.clean:
        for key in ("doc_effect", "effect"):
            for arg in config.get(key, {}).get("args", []):
                arg["prob"] = 0
        config["curl"]["prob"] = 0

    np.random.seed(args.seed)
    template = SynthVNReceipt(config)
    os.makedirs(args.out, exist_ok=True)

    for i in range(args.count):
        data = template.generate()
        image = Image.fromarray(data["image"].astype(np.uint8))
        if args.boxes:
            draw = ImageDraw.Draw(image)
            for box in data["boxes"]:
                quad = [tuple(p) for p in box["quad"]]
                draw.polygon(quad, outline=color_of(box["kind"]))
        path = os.path.join(args.out, f"preview_{i}.jpg")
        image.save(path, quality=92)
        print(f"{path}  {image.size}  {len(data['boxes'])} boxes")


if __name__ == "__main__":
    main()
