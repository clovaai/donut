# SynthDoG-VN 🧾 — Sinh ảnh hoá đơn bán lẻ Việt Nam

Template synthtiger sinh ảnh hoá đơn kiểu máy in nhiệt (quán ăn, nhà hàng, tạp hoá)
kèm nhãn có cấu trúc, dùng để train/fine-tune Donut cho bài toán trích xuất thông tin.

## Chạy

```bash
cd synthdog
synthtiger -o ./outputs/VNReceipt -c 1000 -w 4 -v \
    template_receipt.py SynthVNReceipt config_vi_receipt.yaml
```

Tham số giống SynthDoG gốc: `-o` thư mục ra, `-c` số ảnh, `-w` số worker,
`-s` seed, `-v` in traceback (**luôn bật khi debug** — synthtiger nuốt lỗi và
retry vô hạn, template hỏng sẽ treo im lặng).

Xem trước và soi bounding box:

```bash
python tools/preview_receipt.py --count 4 --out /tmp/preview --boxes          # có hiệu ứng
python tools/preview_receipt.py --count 4 --out /tmp/preview --boxes --clean  # tắt hiệu ứng
```

Kiểm tra font trước khi thêm vào `resources/font/vi`:

```bash
python tools/check_fonts.py resources/font/vi
```

## Thành phần

| File | Vai trò |
|---|---|
| `template_receipt.py` | Template `SynthVNReceipt` — điều phối và lưu dữ liệu |
| `elements/receipt.py` | `ReceiptSampler` (sinh nội dung) + `ReceiptLayout` (bố cục) + `Receipt` (render) |
| `elements/warp.py` | `CurlWarp` — cong giấy phi tuyến, **có map lại toạ độ** |
| `config_vi_receipt.yaml` | Toàn bộ tham số ngẫu nhiên |
| `resources/corpus/vi/` | `items.txt` (tên món + khoảng giá), `shops.txt`, `streets.txt`, `footers.txt` |
| `resources/font/vi/` | Liberation Mono (SIL OFL 1.1) — phủ đủ dấu tiếng Việt |
| `tools/preview_receipt.py` | Xem trước + vẽ box để kiểm tra |
| `tools/check_fonts.py` | Kiểm tra font có đủ glyph tiếng Việt |

## Khác gì SynthDoG gốc

**1. Nội dung có cấu trúc, không phải chữ ngẫu nhiên.**
SynthDoG cắt ký tự liên tục từ Wikipedia; ở đây hoá đơn được sinh từ mô hình dữ liệu
thật (cửa hàng → mặt hàng → tổng tiền), nên số học **nhất quán**: đơn giá × số lượng
= thành tiền, tổng các dòng = tạm tính, tiền khách đưa − tổng = tiền thối.

**2. Vẽ theo trường, không theo ký tự.**
`elements/textbox.py` của SynthDoG tạo một `TextLayer` cho **mỗi ký tự** (~2.7 ms/ký tự).
Hoá đơn dày chữ nên cách đó không dùng được. Ở đây mỗi trường là một `TextLayer`.
Đo trên máy 4 nhân: **0.44 s/ảnh** (4 worker) cho hoá đơn ~40 dòng, so với 0.60 s/ảnh
của SynthDoG chỉ với ~370 ký tự.

**3. Giấy cong mà nhãn vẫn đúng.**
`components.ElasticDistortion` của synthtiger chỉ warp pixel, **không** cập nhật toạ độ —
méo mạnh là box lệch. `CurlWarp` định nghĩa biến dạng bằng công thức giải tích:

```
lượt 1 (theo hàng y):  x' = a(y)·(x − cx) + cx + b(y)
lượt 2 (theo cột x'):  y' = y + c(x')
```

Mỗi lượt khả nghịch trên một trục, nên vừa dựng được `cv2.remap` (ánh xạ ngược) cho
ảnh, vừa map xuôi được toạ độ 4 góc của từng trường. Kết quả: giấy uốn sóng nhưng
polygon vẫn bám sát chữ.

## Nhãn xuất ra

`metadata.jsonl` theo đúng định dạng `DonutDataset` đọc được:

```json
{
  "file_name": "image_0.jpg",
  "ground_truth": "{\"gt_parse\": {\"store\": {...}, \"menu\": [...], \"total\": {...}}}",
  "boxes": [{"kind": "menu.nm", "text": "BUN BO HUE", "quad": [[x,y],[x,y],[x,y],[x,y]]}]
}
```

- `gt_parse` — cấu trúc lồng nhau kiểu CORD: `store.{name,address,phone}`,
  `menu[].{nm,cnt,price,unitprice}`, `total.{subtotal_price,discount_price,tax_price,
  total_price,cashprice,changeprice}`.
- `boxes` — polygon 4 điểm cho từng trường. Donut **bỏ qua** khoá lạ trong
  `ground_truth` nên `boxes` để riêng ngoài; dùng cho detection hoặc để kiểm tra.
- Đổi `label_format: text` trong YAML để xuất `{"text_sequence": "..."}` thay vì
  `gt_parse` — dùng cho bài pre-training đọc trơn.

## Các trục ngẫu nhiên hoá

- **Có dấu / không dấu** (`ascii_fold`) — máy in nhiệt đời cũ chỉ in ASCII, đúng như
  ảnh hoá đơn thật ("QUAN AN THIEN TAN"). Mặc định 60% bỏ dấu.
- **Chữ hoa / thường** (`uppercase`), **đậm / thường** (`font.bold`), độ đậm mực (`ink`).
- **Bề rộng giấy** tính theo số ký tự (`ncols: [32, 48]` — đúng khổ giấy nhiệt thật).
- **Bố cục mặt hàng**: một dòng (`SL | tên | thành tiền`) hoặc hai dòng
  (`tên` / `SL x đơn giá | thành tiền`).
- **Khối tổng tiền**: có/không tạm tính, VAT 8%, giảm giá, tiền khách đưa, tiền thối.
- **Định dạng tiền**: `537,000` / `537.000` / `537.000đ` / `537,000 VND`.
- **Ký tự phân cách**: `*`, `-`, `=`, `.`, `~`, `_`.
- **Biến dạng**: cong giấy, elastic, perspective, bóng, nhoè chuyển động, nén JPEG.

## Muốn chỉnh gì thì sửa ở đâu

| Muốn | Sửa |
|---|---|
| Thêm món / đổi giá | `resources/corpus/vi/items.txt` (`tên<TAB>giá_min<TAB>giá_max`) |
| Thêm font | Bỏ vào `resources/font/vi`, **chạy `tools/check_fonts.py` trước** |
| Giấy cong nhiều/ít hơn | `curl.shift`, `curl.squeeze`, `curl.wave` trong YAML |
| Hoá đơn dài/ngắn | `receipt.content.num_items` |
| Giấy rộng/hẹp | `receipt.layout.ncols` |
| Thêm hiệu ứng ảnh | Thêm component vào `Iterator` trong `template_receipt.py` **và** thêm khối `args` **đúng thứ tự** trong YAML (synthtiger ghép theo index, sai thứ tự không báo lỗi) |

## Hạn chế đã biết

- **Ảnh nền** vẫn dùng 9 ảnh của SynthDoG gốc (mặt trăng, hạt cà phê, phòng ngủ...),
  không giống bối cảnh hoá đơn thật. Bỏ ảnh chụp mặt bàn / tay cầm hoá đơn vào
  `resources/background` sẽ cải thiện rõ nhất — đây là thay đổi đáng làm đầu tiên.
- **Giấy** dùng texture giấy A4 của SynthDoG, chưa có vân giấy nhiệt (bóng, hơi ngả vàng).
- Chỉ có 2 font (Liberation Mono thường/đậm). Hoá đơn thật dùng nhiều font kim/nhiệt khác nhau.
- `CurlWarp` mô hình hoá giấy cong theo sóng trơn, chưa có nếp gấp gãy góc.
- Chưa sinh mã vạch / QR / logo cửa hàng.
