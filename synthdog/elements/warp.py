"""
Donut / SynthDoG-VN
MIT License

Cong giấy phi tuyến (page curl) cho hoá đơn nhiệt.

Khác với `components.ElasticDistortion` của synthtiger — vốn chỉ warp pixel và
KHÔNG cập nhật toạ độ — biến dạng ở đây được định nghĩa bằng công thức giải
tích nên toạ độ của từng ô chữ map lại được chính xác. Nhờ vậy nhãn bounding
box vẫn đúng sau khi giấy đã cong.

Mô hình biến dạng gồm 2 lượt, mỗi lượt khả nghịch:
    lượt 1 (theo hàng y): x' = a(y) * (x - cx) + cx + b(y)
    lượt 2 (theo cột x'): y' = y + c(x')
với a, b, c là các hàm sin/cos trơn theo một trục duy nhất.
"""
import cv2
import numpy as np


class CurlWarp:
    def __init__(self, config=None):
        config = config or {}
        self.prob = config.get("prob", 1.0)
        # biên độ lệch ngang, theo tỉ lệ chiều rộng
        self.shift = config.get("shift", [0.0, 0.03])
        # độ bóp ngang (giấy cuộn vào trong), theo tỉ lệ
        self.squeeze = config.get("squeeze", [0.0, 0.08])
        # biên độ gợn dọc, theo tỉ lệ chiều cao
        self.wave = config.get("wave", [0.0, 0.010])
        # số chu kỳ sóng trên toàn chiều dài / chiều rộng
        self.periods_y = config.get("periods_y", [0.4, 2.0])
        self.periods_x = config.get("periods_x", [0.3, 0.8])

    def sample(self):
        u = np.random.uniform
        return {
            "state": np.random.rand() < self.prob,
            "shift": u(*self.shift),
            "squeeze": u(*self.squeeze),
            "wave": u(*self.wave),
            "periods_y": u(*self.periods_y),
            "periods_x": u(*self.periods_x),
            "phase_y": u(0, 2 * np.pi),
            "phase_x": u(0, 2 * np.pi),
        }

    def apply(self, image, quads, meta=None):
        """Warp `image` (HxWx4 float32) và map theo danh sách quad.

        quads: array (N, 4, 2) toạ độ trong hệ của `image`.
        Trả về (image_mới, quads_mới). Ảnh được pad trước để không bị cắt mất,
        và quad đã cộng sẵn offset của phần pad.
        """
        meta = self.sample() if meta is None else meta
        quads = np.array(quads, dtype=np.float32).reshape(-1, 4, 2)
        if not meta["state"]:
            return image, quads

        h, w = image.shape[:2]
        pad = int(np.ceil(max(meta["shift"] * w, meta["wave"] * h, meta["squeeze"] * w) + 2))
        image = np.pad(image, ((pad, pad), (pad, pad), (0, 0)))
        quads = quads + pad
        ph, pw = image.shape[:2]
        cx = pw / 2.0

        def a_of(y):  # hệ số bóp ngang theo hàng
            t = 2 * np.pi * meta["periods_y"] * y / max(ph, 1) + meta["phase_y"]
            return 1.0 - meta["squeeze"] * (1.0 - np.cos(t)) / 2.0

        def b_of(y):  # lệch ngang theo hàng
            t = 2 * np.pi * meta["periods_y"] * y / max(ph, 1) + meta["phase_y"]
            return meta["shift"] * pw * np.sin(t)

        def c_of(x):  # lệch dọc theo cột
            t = 2 * np.pi * meta["periods_x"] * x / max(pw, 1) + meta["phase_x"]
            return meta["wave"] * ph * np.sin(t)

        # --- ảnh: cần ánh xạ ngược (dst -> src) cho cv2.remap ---
        xx, yy = np.meshgrid(np.arange(pw, dtype=np.float32), np.arange(ph, dtype=np.float32))
        y1 = yy - c_of(xx)                       # nghịch lượt 2
        map_x = (xx - cx - b_of(y1)) / a_of(y1) + cx   # nghịch lượt 1
        map_y = y1
        image = cv2.remap(
            image,
            map_x.astype(np.float32),
            map_y.astype(np.float32),
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0, 0),
        )

        # --- toạ độ: dùng ánh xạ xuôi (src -> dst) ---
        xs, ys = quads[..., 0], quads[..., 1]
        nx = a_of(ys) * (xs - cx) + cx + b_of(ys)
        ny = ys + c_of(nx)
        quads = np.stack([nx, ny], axis=-1).astype(np.float32)

        return image, quads
