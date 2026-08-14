# SynthDoG-VN 🧾 — Sinh ảnh hoá đơn bán lẻ Việt Nam

Template [synthtiger](https://github.com/clovaai/synthtiger) sinh ảnh hoá đơn kiểu máy in
nhiệt (quán ăn, nhà hàng, tạp hoá) **kèm nhãn có cấu trúc**, dùng để train/fine-tune
[Donut](https://github.com/clovaai/donut) cho bài toán trích xuất thông tin từ hoá đơn.

![Mẫu hoá đơn sinh ra](docs/samples/receipts.jpg)

*8 mẫu sinh bằng config mặc định — có dấu/không dấu, chữ hoa/thường, 1 dòng/2 dòng mỗi
mặt hàng, có/không VAT, giảm giá, tiền thối; giấy nghiêng, cong và nhoè khác nhau.*

---

## 1. Yêu cầu môi trường

| Thành phần | Yêu cầu |
|---|---|
| Python | 3.8 – 3.12 (đã kiểm thử trên **3.11**) |
| Hệ điều hành | Linux / macOS / WSL |
| RAM | ~1 GB mỗi worker |
| GPU | **Không cần** — toàn bộ chạy trên CPU |
| Đĩa | ~50 KB mỗi ảnh sinh ra |

Thư viện đã ghim sẵn trong `requirements.txt`. **Ba mốc chặn trên là do lỗi thật, đừng gỡ:**

- `pillow<10` — synthtiger 1.2.1 gọi `ImageFont.getsize()`, API này bị xoá ở Pillow 10.
- `numpy<2` — `imgaug` dùng `np.sctypes`, bị xoá ở NumPy 2.
- `opencv-python<5` — bản 5 yêu cầu `numpy>=2`, xung đột với dòng trên.

---

## 2. Cài đặt

```bash
git clone https://github.com/LinhPhuong14/synthdog.git
cd synthdog/synthdog

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -U pip setuptools wheel   # BẮT BUỘC — xem mục Sự cố thường gặp
pip install -r requirements.txt
```

> **macOS** cần thêm: `export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES`

Kiểm tra cài đặt:

```bash
python -c "import synthtiger, PIL, numpy, cv2; print(synthtiger.__version__, PIL.__version__, numpy.__version__, cv2.__version__)"
# 1.2.1 9.5.0 1.26.4 4.11.0
```

---

## 3. Sinh dữ liệu

```bash
# chạy từ thư mục synthdog/synthdog
synthtiger -o ./outputs/VNReceipt -c 1000 -w 4 -v \
    template_receipt.py SynthVNReceipt config_vi_receipt.yaml
```

| Tham số | Ý nghĩa |
|---|---|
| `-o` | Thư mục xuất dữ liệu |
| `-c` | Số ảnh cần sinh |
| `-w` | Số worker (đặt bằng số nhân CPU) |
| `-s` | Seed — **cùng seed cho ra cùng dataset**, bất kể `-w` bao nhiêu |
| `-v` | In traceback |

> ⚠️ **Luôn bật `-v` khi debug.** synthtiger nuốt exception rồi retry vô hạn; template
> hỏng sẽ treo im lặng không báo gì.

Kết quả:

```
outputs/VNReceipt/
├── train/        (80%)  image_0.jpg, image_3.jpg, ..., metadata.jsonl
├── validation/   (10%)
└── test/         (10%)
```

---

## 4. Xem trước và kiểm tra

```bash
# lưới 8 mẫu, có đủ hiệu ứng
python tools/preview_receipt.py --count 8 --grid 4 --seed 2026 --out /tmp/preview

# tắt hiệu ứng + vẽ box từng trường — để soi bố cục
python tools/preview_receipt.py --count 2 --grid 2 --seed 3 --clean --boxes --out /tmp/preview

# kiểm tra font có đủ dấu tiếng Việt không (chạy TRƯỚC khi thêm font mới)
python tools/check_fonts.py resources/font/vi
```

**Bố cục sạch, không hiệu ứng** — mỗi màu là một nhóm trường
(🔴 cửa hàng · 🟠 thông tin phiếu · 🟢 mặt hàng · 🔵 tổng tiền · 🟣 chân hoá đơn):

![Bố cục và bounding box](docs/samples/layout_boxes.jpg)

**Sau khi giấy đã nghiêng và cong** — box vẫn bám sát từng dòng chữ:

![Box sau khi giấy cong](docs/samples/curl_boxes.jpg)

---

## 5. Nhãn xuất ra

`metadata.jsonl` — mỗi dòng một ảnh, đúng định dạng `DonutDataset` đọc được:

```json
{
  "file_name": "image_0.jpg",
  "ground_truth": "{\"gt_parse\": {\"store\": {\"name\": \"QUAN AN CHO LON\", \"address\": \"40-71 TON DAN HAI PHONG\", \"phone\": \"DT: 044695122\"}, \"menu\": [{\"nm\": \"BUN MOC\", \"cnt\": \"2\", \"price\": \"114,000 VND\"}], \"total\": {\"total_price\": \"1,677,000 VND\", \"cashprice\": \"1,700,000 VND\", \"changeprice\": \"23,000 VND\"}}}",
  "boxes": [
    {"kind": "menu.nm", "text": "BUN MOC", "quad": [[x,y],[x,y],[x,y],[x,y]]}
  ]
}
```

- **`gt_parse`** — cấu trúc lồng nhau kiểu CORD:
  - `store.{name, address, phone}`
  - `menu[].{nm, cnt, price, unitprice}`
  - `total.{subtotal_price, discount_price, tax_price, total_price, cashprice, changeprice}`
- **`boxes`** — polygon 4 điểm cho từng trường. Donut **bỏ qua** khoá lạ trong
  `ground_truth` nên `boxes` để riêng bên ngoài; dùng cho detection hoặc để kiểm tra nhãn.
- Đổi `label_format: text` trong YAML để xuất `{"text_sequence": "..."}` thay cho
  `gt_parse` — dùng cho bài pre-training đọc trơn.

**Số học trong nhãn luôn nhất quán**: đơn giá × số lượng = thành tiền, tổng các dòng =
tạm tính, tiền khách đưa − tổng = tiền thối. Đã kiểm tra tự động trên 60 mẫu, 0 lỗi.

---

## 6. Dùng để train Donut

Tạo `config/train_vi_receipt.yaml` ở thư mục gốc repo:

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

```bash
cd ..                 # về thư mục gốc repo
python train.py --config config/train_vi_receipt.yaml --exp_version "vi_receipt_v1"
```

Lưu ý `train.py` cần thêm `torch`, `pytorch-lightning`, `transformers` — cài theo
`setup.py` ở thư mục gốc (`pip install .`), **tách riêng khỏi venv sinh dữ liệu** vì
Donut cần Pillow/NumPy mới hơn mức synthtiger cho phép.

---

## 7. Thành phần

| File | Vai trò |
|---|---|
| `template_receipt.py` | Template `SynthVNReceipt` — điều phối và lưu dữ liệu |
| `elements/receipt.py` | `ReceiptSampler` (sinh nội dung) + `ReceiptLayout` (bố cục) + `Receipt` (render) |
| `elements/warp.py` | `CurlWarp` — cong giấy phi tuyến, **có map lại toạ độ** |
| `config_vi_receipt.yaml` | Toàn bộ tham số ngẫu nhiên |
| `requirements.txt` | Thư viện đã ghim phiên bản |
| `resources/corpus/vi/` | `items.txt` (78 món + khoảng giá), `shops.txt`, `streets.txt`, `footers.txt` |
| `resources/font/vi/` | Liberation Mono (SIL OFL 1.1) — phủ đủ dấu tiếng Việt |
| `tools/preview_receipt.py` | Xem trước, ghép lưới, vẽ box |
| `tools/check_fonts.py` | Kiểm tra font có đủ glyph tiếng Việt |

---

## 8. Khác gì SynthDoG gốc

**1. Nội dung có cấu trúc, không phải chữ ngẫu nhiên.**
SynthDoG cắt ký tự liên tục từ Wikipedia. Ở đây hoá đơn sinh từ một mô hình dữ liệu thật
(cửa hàng → mặt hàng → tổng tiền), nên nhãn xuất được dạng `gt_parse` lồng nhau và các
con số khớp nhau.

**2. Vẽ theo trường, không theo ký tự.**
`elements/textbox.py` của SynthDoG tạo một `TextLayer` cho **mỗi ký tự** — đo được
~2.7 ms/ký tự. Hoá đơn dày chữ nên cách đó không dùng được. Ở đây mỗi trường là một
`TextLayer`.

| | Hoá đơn ~40 dòng | SynthDoG gốc (~370 ký tự) |
|---|---:|---:|
| 4 worker (máy 4 nhân) | **0.44 s/ảnh** | 0.60 s/ảnh |
| CPU tiêu tốn | **1.78 CPU-s/ảnh** | 2.34 CPU-s/ảnh |

Nhiều chữ hơn hẳn mà vẫn nhanh hơn.

**3. Giấy cong mà nhãn vẫn đúng.**
`components.ElasticDistortion` của synthtiger chỉ warp pixel, **không** cập nhật toạ độ —
méo mạnh là box lệch khỏi chữ. `CurlWarp` định nghĩa biến dạng bằng công thức giải tích,
tách thành 2 lượt, mỗi lượt khả nghịch trên một trục:

```
lượt 1 (theo hàng y):  x' = a(y)·(x − cx) + cx + b(y)
lượt 2 (theo cột x'):  y' = y + c(x')
```

Nhờ vậy vừa dựng được ánh xạ ngược cho `cv2.remap` (ảnh), vừa map xuôi được 4 góc của
từng trường (nhãn). Đó là lý do box trong ảnh mục 4 vẫn bám sát chữ dù giấy đã uốn.

---

## 9. Các trục ngẫu nhiên hoá

- **Có dấu / không dấu** (`ascii_fold`, mặc định 60% bỏ dấu) — máy in nhiệt đời cũ chỉ in
  ASCII, đúng như hoá đơn thật ("QUAN AN THIEN TAN").
- **Chữ hoa/thường** (`uppercase`), **đậm/thường** (`font.bold`), độ đậm mực (`ink`).
- **Bề rộng giấy** tính theo số ký tự (`ncols: [32, 48]` — đúng khổ giấy nhiệt thật).
- **Bố cục mặt hàng**: một dòng (`SL | tên | thành tiền`) hoặc hai dòng
  (`tên` / `SL x đơn giá | thành tiền`).
- **Khối tổng tiền**: có/không tạm tính, VAT 8%, giảm giá, tiền khách đưa, tiền thối.
- **Định dạng tiền**: `537,000` / `537.000` / `537.000đ` / `537,000 VND`.
- **Ký tự phân cách**: `*`, `-`, `=`, `.`, `~`, `_`.
- **Biến dạng ảnh**: cong giấy, elastic, perspective, bóng đổ, nhoè chuyển động, nén JPEG.

---

## 10. Muốn chỉnh gì thì sửa ở đâu

| Muốn | Sửa |
|---|---|
| Thêm món / đổi giá | `resources/corpus/vi/items.txt` — `tên<TAB>giá_min<TAB>giá_max` |
| Thêm tên quán / đường / dòng chân | `shops.txt` / `streets.txt` / `footers.txt` |
| Thêm font | Bỏ vào `resources/font/vi`, **chạy `tools/check_fonts.py` trước** |
| Giấy cong nhiều/ít | `curl.shift`, `curl.squeeze`, `curl.wave` |
| Hoá đơn dài/ngắn | `receipt.content.num_items` |
| Giấy rộng/hẹp | `receipt.layout.ncols` |
| Tỉ lệ bỏ dấu | `receipt.content.ascii_fold` |
| Thêm hiệu ứng ảnh | Thêm component vào `Iterator` trong `template_receipt.py` **và** thêm khối `args` **đúng thứ tự** trong YAML — synthtiger ghép theo index, sai thứ tự **không báo lỗi** |

---

## 11. Sự cố thường gặp

| Triệu chứng | Nguyên nhân & cách xử lý |
|---|---|
| `ERROR: Failed building wheel for pytweening` | setuptools bản vá của Debian/Ubuntu. Cài trong **venv** và chạy `pip install -U setuptools` trước. Đừng cài vào python hệ thống. |
| Chạy mãi không ra ảnh, không báo lỗi | synthtiger nuốt exception rồi retry vô hạn. Chạy lại với `-v` để thấy traceback. |
| `AttributeError: 'FreeTypeFont' object has no attribute 'getsize'` | Pillow ≥ 10. Chạy `pip install "pillow<10"`. |
| `AttributeError: np.sctypes was removed` | NumPy ≥ 2. Chạy `pip install "numpy<2"`. |
| Chữ hiện ô vuông ▯▯▯ | Font thiếu glyph tiếng Việt. Chạy `python tools/check_fonts.py resources/font/vi`. **Nhãn vẫn ghi đúng chữ nên lỗi này không tự báo ra** — phải chủ động kiểm tra. |
| `FileNotFoundError: resources/...` | Phải chạy từ thư mục `synthdog/synthdog`, đường dẫn trong YAML là tương đối. |

---

## 12. Hạn chế đã biết

- **Ảnh nền** vẫn là 9 ảnh của SynthDoG gốc (mặt trăng, hạt cà phê, bánh donut...), không
  giống bối cảnh hoá đơn thật. Bỏ vài chục ảnh chụp mặt bàn / tay cầm hoá đơn vào
  `resources/background` sẽ cải thiện realism **nhiều hơn bất kỳ thay đổi code nào** —
  đây là việc đáng làm đầu tiên.
- **Giấy** dùng texture giấy A4 của SynthDoG, chưa có vân giấy nhiệt (bóng, hơi ngả vàng).
- Chỉ có 2 font (Liberation Mono thường/đậm). Hoá đơn thật dùng nhiều font kim/nhiệt khác.
- `CurlWarp` mô hình hoá giấy cong theo sóng trơn, chưa có nếp gấp gãy góc.
- Chưa sinh mã vạch / QR / logo cửa hàng.
- Giá tiền lấy theo khoảng cố định trong `items.txt`, chưa mô phỏng lạm phát theo năm in
  trên hoá đơn.

---

## Giấy phép

Code theo MIT (kế thừa từ Donut). Font Liberation Mono theo SIL Open Font License 1.1.
