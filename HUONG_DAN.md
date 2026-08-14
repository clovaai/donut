# Hướng dẫn cài đặt và chạy

Tài liệu này hướng dẫn cài môi trường và chạy hai thứ trong repo:

1. **SynthDoG-VN** 🧾 — sinh ảnh **hoá đơn bán lẻ Việt Nam** kèm nhãn có cấu trúc
   *(phần mới thêm — tài liệu chi tiết: [`synthdog/README_vi_receipt.md`](synthdog/README_vi_receipt.md))*
2. **SynthDoG gốc** 🐶 — sinh ảnh tài liệu chữ ngẫu nhiên (EN/KO/JA/ZH)
3. **Donut** 🍩 — train model trên dữ liệu đã sinh

![Mẫu hoá đơn sinh ra](synthdog/docs/samples/receipts.jpg)

> ⚠️ **Quan trọng**: dùng **hai môi trường ảo tách biệt**. Phần sinh dữ liệu cần
> `pillow<10` + `numpy<2`, còn Donut cần bản mới hơn. Cài chung sẽ xung đột.

---

## Phần 1 — Sinh dữ liệu

### 1.1. Yêu cầu

| Thành phần | Yêu cầu |
|---|---|
| Python | 3.8 – 3.12 (đã kiểm thử **3.11**) |
| Hệ điều hành | Linux / macOS / WSL |
| RAM | ~1 GB mỗi worker |
| GPU | **Không cần** |
| Đĩa | ~50 KB mỗi ảnh |

Thư viện (đã ghim sẵn trong `synthdog/requirements.txt`):

```
synthtiger==1.2.1
pillow<10           # synthtiger gọi ImageFont.getsize(), bị xoá ở Pillow 10
numpy<2             # imgaug dùng np.sctypes, bị xoá ở NumPy 2
opencv-python<5     # bản 5 đòi numpy>=2, xung đột dòng trên
fonttools>=4.0
```

Cả ba mốc chặn trên đều là lỗi thật khi chạy, không phải đề phòng thừa.

### 1.2. Cài đặt

```bash
git clone https://github.com/LinhPhuong14/synthdog.git
cd synthdog/synthdog

python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate

pip install -U pip setuptools wheel  # BẮT BUỘC — xem mục Sự cố thường gặp
pip install -r requirements.txt
```

macOS cần thêm:

```bash
export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES
```

Kiểm tra:

```bash
python -c "import synthtiger, PIL, numpy, cv2; print(synthtiger.__version__, PIL.__version__, numpy.__version__, cv2.__version__)"
# 1.2.1 9.5.0 1.26.4 4.11.0
```

### 1.3. Chạy — hoá đơn Việt Nam

```bash
# đứng ở thư mục synthdog/synthdog
synthtiger -o ./outputs/VNReceipt -c 1000 -w 4 -v \
    template_receipt.py SynthVNReceipt config_vi_receipt.yaml
```

### 1.4. Chạy — SynthDoG gốc (tài liệu chữ ngẫu nhiên)

```bash
synthtiger -o ./outputs/SynthDoG_en -c 1000 -w 4 -v template.py SynthDoG config_en.yaml
# đổi config_en.yaml -> config_ko / config_ja / config_zh cho ngôn ngữ khác
```

### 1.5. Tham số dòng lệnh

| Tham số | Ý nghĩa |
|---|---|
| `-o` | Thư mục xuất dữ liệu |
| `-c` | Số ảnh cần sinh |
| `-w` | Số worker — đặt bằng số nhân CPU |
| `-s` | Seed — **cùng seed cho ra cùng dataset**, bất kể `-w` bao nhiêu |
| `-v` | In traceback |

> ⚠️ **Luôn bật `-v` khi debug.** synthtiger nuốt exception rồi tự retry vô hạn, nên
> template lỗi biểu hiện thành **treo im lặng** chứ không báo gì.

### 1.6. Kết quả

```
outputs/VNReceipt/
├── train/         (80%)   image_0.jpg, image_3.jpg, ..., metadata.jsonl
├── validation/    (10%)
└── test/          (10%)
```

Mỗi dòng trong `metadata.jsonl`:

```json
{
  "file_name": "image_0.jpg",
  "ground_truth": "{\"gt_parse\": {\"store\": {...}, \"menu\": [...], \"total\": {...}}}",
  "boxes": [{"kind": "menu.nm", "text": "BUN BO HUE", "quad": [[x,y],[x,y],[x,y],[x,y]]}]
}
```

### 1.7. Xem trước và kiểm tra (chỉ có ở template hoá đơn)

```bash
# lưới 8 mẫu, đủ hiệu ứng
python tools/preview_receipt.py --count 8 --grid 4 --seed 2026 --out /tmp/preview

# tắt hiệu ứng + vẽ bounding box từng trường
python tools/preview_receipt.py --count 2 --grid 2 --seed 3 --clean --boxes --out /tmp/preview

# kiểm tra font đủ dấu tiếng Việt (chạy TRƯỚC khi thêm font mới)
python tools/check_fonts.py resources/font/vi
```

![Bố cục và bounding box](synthdog/docs/samples/layout_boxes.jpg)

---

## Phần 2 — Train Donut

Dùng **môi trường ảo khác** với phần sinh dữ liệu.

### 2.1. Yêu cầu

| Thành phần | Yêu cầu |
|---|---|
| Python | 3.7+ |
| GPU | **Có** — fine-tune CORD chạy trên 1×A100 theo README gốc |
| Thư viện | `torch`, `pytorch-lightning`, `transformers`, `timm`, `sconf`… |

### 2.2. Cài đặt

```bash
cd /đường/dẫn/tới/synthdog        # thư mục gốc repo
python -m venv .venv-train
source .venv-train/bin/activate
pip install -U pip
pip install .                      # cài theo setup.py
```

### 2.3. Cấu hình

Tạo `config/train_vi_receipt.yaml`:

```yaml
resume_from_checkpoint_path: null
result_path: "./result"
pretrained_model_name_or_path: "naver-clova-ix/donut-base"
dataset_name_or_paths: ["./synthdog/outputs/VNReceipt"]
sort_json_key: False
train_batch_sizes: [4]
val_batch_sizes: [1]
input_size: [1280, 960]
max_length: 768
align_long_axis: False
num_nodes: 1
seed: 2022
lr: 3e-5
warmup_steps: 300
num_training_samples_per_epoch: 800
max_epochs: 30
max_steps: -1
num_workers: 8
val_check_interval: 1.0
check_val_every_n_epoch: 3
gradient_clip_val: 1.0
verbose: True
```

### 2.4. Chạy

```bash
python train.py --config config/train_vi_receipt.yaml --exp_version "vi_receipt_v1"

# đánh giá
python test.py --dataset_name_or_path ./synthdog/outputs/VNReceipt \
               --pretrained_model_name_or_path ./result/train_vi_receipt/vi_receipt_v1 \
               --save_path ./result/output.json
```

---

## Sự cố thường gặp

| Triệu chứng | Nguyên nhân & cách xử lý |
|---|---|
| `ERROR: Failed building wheel for pytweening` | setuptools bản vá của Debian/Ubuntu. Cài trong **venv** rồi `pip install -U setuptools` trước. Đừng cài vào python hệ thống. |
| Chạy mãi không ra ảnh, không báo lỗi | synthtiger nuốt exception rồi retry vô hạn. Chạy lại kèm `-v`. |
| `AttributeError: 'FreeTypeFont' object has no attribute 'getsize'` | Pillow ≥ 10 → `pip install "pillow<10"` |
| `AttributeError: np.sctypes was removed` | NumPy ≥ 2 → `pip install "numpy<2"` |
| `FileNotFoundError: resources/...` | Phải chạy từ thư mục `synthdog/synthdog` — đường dẫn trong YAML là tương đối. |
| Chữ tiếng Việt hiện ô vuông ▯▯▯ | Font thiếu glyph. Chạy `python tools/check_fonts.py resources/font/vi`. **Nhãn vẫn ghi đúng chữ nên lỗi này không tự báo ra.** |
| macOS treo khi dùng nhiều worker | `export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES` |

---

## Hiệu năng đo được

Máy 4 vCPU, 4 worker:

| Template | Thời gian | CPU | Nội dung |
|---|---:|---:|---|
| Hoá đơn VN | **0.44 s/ảnh** | 1.78 CPU-s | ~40 dòng chữ |
| SynthDoG gốc | 0.60 s/ảnh | 2.34 CPU-s | ~370 ký tự |

Ước tính: 10.000 ảnh hoá đơn ≈ **75 phút** trên 4 nhân, hoặc ~20 phút trên 16 nhân.

---

## Đọc thêm

- [`synthdog/README_vi_receipt.md`](synthdog/README_vi_receipt.md) — tài liệu đầy đủ về
  template hoá đơn: kiến trúc, các trục ngẫu nhiên hoá, cách tuỳ chỉnh, hạn chế
- [`synthdog/README.md`](synthdog/README.md) — SynthDoG gốc
- [`README.md`](README.md) — Donut gốc (tiếng Anh)
