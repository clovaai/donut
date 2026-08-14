"""
Donut / SynthDoG-VN
MIT License

Sinh nội dung + bố cục hoá đơn bán lẻ Việt Nam (kiểu máy in nhiệt).

Khác `elements/content.py` của SynthDoG gốc ở hai điểm:
  1. Nội dung có CẤU TRÚC (cửa hàng / mặt hàng / tổng tiền) chứ không phải
     ký tự ngẫu nhiên cắt từ Wikipedia, nên nhãn xuất ra được dạng `gt_parse`
     lồng nhau dùng cho bài toán trích xuất thông tin.
  2. Mỗi trường được vẽ bằng MỘT `TextLayer` cho cả chuỗi, thay vì một
     TextLayer cho từng ký tự. Cách của SynthDoG tốn ~2.7 ms/ký tự; hoá đơn
     dày chữ nên vẽ theo ký tự là không dùng được.
"""
import os
import unicodedata

import numpy as np
from synthtiger import layers

# ký tự phân cách hay gặp trên hoá đơn nhiệt
SEPARATORS = ["*", "-", "=", ".", "~", "_"]
CITIES = [
    "TPHCM", "TP.HCM", "Hà Nội", "Đà Nẵng", "Hải Phòng", "Cần Thơ",
    "Biên Hoà", "Nha Trang", "Huế", "Vũng Tàu", "Bình Dương",
]
SHOP_PREFIXES = [
    "Quán Ăn", "Nhà Hàng", "Cửa Hàng", "Quán", "Cafe", "Quán Nhậu",
    "Bếp", "Tiệm Ăn", "Nhà Hàng - Karaoke", "Siêu Thị Mini",
]
PAY_LABELS = ["Tiền mặt", "Tổng cộng", "Thành tiền", "Tổng thanh toán", "Tổng tiền"]


def ascii_fold(text):
    """Bỏ dấu tiếng Việt — máy in nhiệt đời cũ chỉ in được ASCII."""
    text = text.replace("Đ", "D").replace("đ", "d")
    text = unicodedata.normalize("NFD", text)
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    return unicodedata.normalize("NFC", text)


def read_list(path):
    """Đọc file corpus, bỏ dòng trống và dòng bắt đầu bằng '#'."""
    with open(path, "r", encoding="utf-8") as fp:
        lines = [ln.strip() for ln in fp]
    return [ln for ln in lines if ln and not ln.startswith("#")]


def read_items(path):
    """items.txt: tên<TAB>giá_min<TAB>giá_max"""
    out = []
    for line in read_list(path):
        parts = line.split("\t")
        if len(parts) != 3:
            continue
        out.append((parts[0], int(parts[1]), int(parts[2])))
    return out


class ReceiptSampler:
    """Bốc ra nội dung một hoá đơn, chưa liên quan gì tới ảnh."""

    def __init__(self, config=None):
        config = config or {}
        root = config.get("corpus", "resources/corpus/vi")
        self.items = read_items(os.path.join(root, "items.txt"))
        self.shops = read_list(os.path.join(root, "shops.txt"))
        self.streets = read_list(os.path.join(root, "streets.txt"))
        self.footers = read_list(os.path.join(root, "footers.txt"))

        self.num_items = config.get("num_items", [2, 14])
        self.ascii_fold = config.get("ascii_fold", 0.6)
        self.uppercase = config.get("uppercase", 0.7)
        self.prob_address = config.get("prob_address", 0.9)
        self.prob_phone = config.get("prob_phone", 0.8)
        self.prob_table = config.get("prob_table", 0.5)
        self.prob_meta = config.get("prob_meta", 0.85)
        self.prob_subtotal = config.get("prob_subtotal", 0.35)
        self.prob_vat = config.get("prob_vat", 0.3)
        self.prob_discount = config.get("prob_discount", 0.15)
        self.prob_cash = config.get("prob_cash", 0.45)
        self.prob_unit_price = config.get("prob_unit_price", 0.5)
        self.num_footers = config.get("num_footers", [0, 3])

    # ---------- tiện ích ----------

    def _money(self, value, style):
        sep = "," if style["thousand"] == "," else "."
        s = f"{int(value):,}".replace(",", sep)
        return s + style["suffix"]

    def _case(self, text, style):
        if style["fold"]:
            text = ascii_fold(text)
        if style["upper"]:
            text = text.upper()
        return text

    # ---------- sinh ----------

    def sample(self):
        rand = np.random.rand
        randint = np.random.randint

        style = {
            "fold": rand() < self.ascii_fold,
            "upper": rand() < self.uppercase,
            "thousand": "," if rand() < 0.6 else ".",
            "suffix": ["", "", "", "đ", " VND"][randint(5)],
        }
        # bỏ dấu thì hậu tố 'đ' cũng phải bỏ dấu
        if style["fold"] and style["suffix"] == "đ":
            style["suffix"] = "d"

        C = lambda t: self._case(t, style)
        M = lambda v: self._money(v, style)

        # --- cửa hàng ---
        name = f"{SHOP_PREFIXES[randint(len(SHOP_PREFIXES))]} {self.shops[randint(len(self.shops))]}"
        store = {"name": C(name)}
        if rand() < self.prob_address:
            num = f"{randint(1, 300)}" if rand() < 0.7 else f"{randint(1, 60)}-{randint(61, 200)}"
            street = self.streets[randint(len(self.streets))]
            city = CITIES[randint(len(CITIES))]
            ward = f" P{randint(1, 20)}Q{randint(1, 13)}" if rand() < 0.5 else ""
            store["address"] = C(f"{num} {street}{ward} {city}")
        if rand() < self.prob_phone:
            label = "DT" if style["fold"] else "ĐT"
            phone = f"0{randint(2, 10)}{randint(1000000, 99999999)}"[:11]
            if rand() < 0.4:
                phone = f"{phone[:7]}-{randint(1000000, 9999999)}"
            store["phone"] = C(f"{label}: {phone}")

        # --- dòng thông tin phiếu ---
        day, month, year = randint(1, 29), randint(1, 13), randint(2018, 2027)
        hour, minute = randint(6, 24), randint(0, 60)
        date = f"{day:02d}-{month:02d}-{year}" if rand() < 0.5 else f"{day:02d}/{month:02d}/{year}"
        meta = []
        if rand() < self.prob_meta:
            meta.append(("REG", f"{date} {hour:02d}:{minute:02d}"))
            meta.append((f"CA {randint(1, 4)}", f"MC #{randint(1, 10):02d}   {randint(1, 999999):06d}"))
        else:
            meta.append((C("Ngày"), f"{date} {hour:02d}:{minute:02d}"))
        table = C(f"Bàn số:{randint(1, 60)}") if rand() < self.prob_table else None

        # --- mặt hàng ---
        n = randint(self.num_items[0], self.num_items[1] + 1)
        show_unit = rand() < self.prob_unit_price
        items, subtotal = [], 0
        for _ in range(n):
            nm, lo, hi = self.items[randint(len(self.items))]
            unit = int(round(np.random.uniform(lo, hi) / 1000.0)) * 1000
            cnt = int(randint(1, 11)) if rand() < 0.25 else int(randint(1, 4))
            price = unit * cnt
            subtotal += price
            item = {"nm": C(nm), "cnt": str(cnt), "price": M(price)}
            if show_unit:
                item["unitprice"] = M(unit)
            items.append(item)

        # --- tổng tiền ---
        total = {}
        if rand() < self.prob_subtotal:
            total["subtotal_price"] = M(subtotal)
        grand = subtotal
        if rand() < self.prob_discount:
            disc = int(round(subtotal * np.random.uniform(0.03, 0.2) / 1000)) * 1000
            total["discount_price"] = M(disc)
            grand -= disc
        if rand() < self.prob_vat:
            vat = int(round(grand * 0.08 / 1000)) * 1000
            total["tax_price"] = M(vat)
            grand += vat
        total["total_price"] = M(grand)
        pay_label = C(PAY_LABELS[randint(len(PAY_LABELS))])
        if rand() < self.prob_cash:
            cash = int(np.ceil(grand / 50000.0)) * 50000
            total["cashprice"] = M(cash)
            total["changeprice"] = M(cash - grand)

        # --- chân hoá đơn ---
        k = randint(self.num_footers[0], self.num_footers[1] + 1)
        idx = np.random.permutation(len(self.footers))[:k]
        footer = [C(self.footers[i]) for i in idx]

        return {
            "style": style,
            "store": store,
            "meta": meta,
            "table": table,
            "items": items,
            "total": total,
            "pay_label": pay_label,
            "footer": footer,
        }


class ReceiptLayout:
    """Xếp nội dung thành các dòng trên lưới ký tự cố định (kiểu máy in nhiệt).

    Trả về danh sách "ô": (text, kind, row, col_start, col_end, align, scale)
    `kind` là đường dẫn khoá trong gt_parse, ví dụ 'menu.nm' hay 'total.total_price'.
    """

    def __init__(self, config=None):
        config = config or {}
        self.ncols = config.get("ncols", [32, 48])
        self.wrap_name = config.get("wrap_name", 0.5)
        self.two_line_item = config.get("two_line_item", 0.35)
        self.big_total = config.get("big_total", 0.7)
        self.big_header = config.get("big_header", 0.8)

    @staticmethod
    def _split_name(item, name_w, wrap):
        """Tên hàng dài hơn cột: xuống dòng ở khoảng trắng, hoặc cắt bớt.

        Cắt bớt thì phải sửa luôn `item['nm']` — nếu không `gt_parse` sẽ ghi
        tên đầy đủ trong khi ảnh chỉ hiện phần đã cắt.
        """
        name = item["nm"]
        if len(name) <= name_w:
            return name, None
        if not wrap:
            item["nm"] = name[:name_w].strip()
            return item["nm"], None
        cut = name.rfind(" ", 0, name_w + 1)
        if cut < name_w // 2:  # không có chỗ ngắt hợp lý thì cắt cứng
            cut = name_w
        return name[:cut].strip(), name[cut:].strip()

    def generate(self, data):
        ncols = int(np.random.randint(self.ncols[0], self.ncols[1] + 1))
        rand = np.random.rand
        cells = []
        row = [0]  # dùng list để đóng gói mutable cho closure

        def put(text, kind, align="center", col0=0, col1=None, scale=1.0):
            if not text:
                return
            col1 = ncols if col1 is None else col1
            cells.append({
                "text": str(text), "kind": kind, "row": row[0],
                "col0": col0, "col1": col1, "align": align, "scale": scale,
            })

        def newline(n=1):
            row[0] += n

        def rule():
            ch = SEPARATORS[np.random.randint(len(SEPARATORS))]
            width = ncols if rand() < 0.7 else int(ncols * np.random.uniform(0.3, 0.8))
            put(ch * width, "sep", "center")
            newline()

        store, total = data["store"], data["total"]
        hdr_scale = float(np.random.uniform(1.3, 1.9)) if rand() < self.big_header else 1.0

        # ----- đầu hoá đơn -----
        put(store["name"], "store.name", "center", scale=hdr_scale)
        newline(2 if hdr_scale > 1.2 else 1)
        if "address" in store:
            put(store["address"], "store.address", "center")
            newline()
        if "phone" in store:
            put(store["phone"], "store.phone", "center")
            newline()
        rule()

        # ----- thông tin phiếu -----
        for key, value in data["meta"]:
            put(key, "meta", "left", 0, ncols // 2)
            put(value, "meta", "right", ncols // 2, ncols)
            newline()
        if data["table"]:
            put(data["table"], "meta", "left", scale=float(np.random.uniform(1.0, 1.4)))
            newline()
        if rand() < 0.6:
            newline()

        # ----- các mặt hàng -----
        two_line = rand() < self.two_line_item
        wrap = rand() < self.wrap_name
        # cột tiền phải đủ rộng cho chuỗi giá DÀI NHẤT, nếu không tên hàng sẽ
        # đâm vào số tiền (hậu tố ' VND' làm chuỗi giá dài tới 12-13 ký tự)
        price_w = max([len(item["price"]) for item in data["items"]] + [1]) + 1
        name_end = max(ncols - price_w, 12)
        name_w = max(name_end - 4, 6)
        for item in data["items"]:
            if two_line:
                if len(item["nm"]) > ncols:
                    item["nm"] = item["nm"][:ncols].strip()
                put(item["nm"], "menu.nm", "left", 0, ncols)
                newline()
                unit = item.get("unitprice")
                qty = f"{item['cnt']} x {unit}" if unit else item["cnt"]
                if data["style"]["upper"]:
                    qty = qty.upper()
                put(qty, "menu.cnt", "left", 1, name_end)
                put(item["price"], "menu.price", "right", name_end, ncols)
                newline()
            else:
                head, tail = self._split_name(item, name_w, wrap)
                put(item["cnt"], "menu.cnt", "right", 0, 3)
                put(head, "menu.nm", "left", 4, name_end)
                put(item["price"], "menu.price", "right", name_end, ncols)
                newline()
                if tail:
                    put(tail, "menu.nm", "left", 4, ncols)
                    newline()
        rule()

        # ----- tổng tiền -----
        labels = {
            "subtotal_price": "Tam tinh" if data["style"]["fold"] else "Tạm tính",
            "discount_price": "Giam gia" if data["style"]["fold"] else "Giảm giá",
            "tax_price": "VAT 8%",
            "cashprice": "Tien khach dua" if data["style"]["fold"] else "Tiền khách đưa",
            "changeprice": "Tien thoi lai" if data["style"]["fold"] else "Tiền thối lại",
        }
        for key in ("subtotal_price", "discount_price", "tax_price"):
            if key in total:
                label = labels[key]
                put(data["style"]["upper"] and label.upper() or label,
                    f"total.{key}_label", "left", 0, ncols // 2)
                put(total[key], f"total.{key}", "right", ncols // 2, ncols)
                newline()

        tot_scale = float(np.random.uniform(1.3, 2.0)) if rand() < self.big_total else 1.0
        if tot_scale > 1.2 and rand() < 0.5:
            # nhãn một dòng, số tiền một dòng — như trong ảnh mẫu
            put(data["pay_label"], "total.label", "center", scale=tot_scale)
            newline(2)
            put(total["total_price"], "total.total_price", "right", 0, ncols, scale=tot_scale)
            newline(2)
        else:
            put(data["pay_label"], "total.label", "left", 0, ncols // 2, scale=tot_scale)
            put(total["total_price"], "total.total_price", "right", ncols // 2, ncols, scale=tot_scale)
            newline(2 if tot_scale > 1.2 else 1)

        for key in ("cashprice", "changeprice"):
            if key in total:
                label = labels[key]
                put(data["style"]["upper"] and label.upper() or label,
                    f"total.{key}_label", "left", 0, ncols // 2)
                put(total[key], f"total.{key}", "right", ncols // 2, ncols)
                newline()

        # ----- chân hoá đơn -----
        if data["footer"]:
            newline()
            if rand() < 0.4:
                rule()
            for line in data["footer"]:
                put(line, "footer", "center", scale=float(np.random.uniform(0.9, 1.3)))
                newline()

        return cells, ncols, row[0] + 1


class Receipt:
    """Gộp sampler + layout + render thành các TextLayer."""

    def __init__(self, config=None):
        config = config or {}
        self.sampler = ReceiptSampler(config.get("content", {}))
        self.layout = ReceiptLayout(config.get("layout", {}))
        self.font_paths = config.get("font", {}).get("paths", ["resources/font/vi"])
        self.font_size = config.get("font", {}).get("size", [16, 30])
        self.bold = config.get("font", {}).get("bold", 0.25)
        self.line_spacing = config.get("line_spacing", [1.05, 1.5])
        self.margin = config.get("margin", [0.03, 0.1])
        self.ink = config.get("ink", [0, 90])
        self._fonts = self._collect_fonts()

    def _collect_fonts(self):
        found = []
        for path in self.font_paths:
            if os.path.isdir(path):
                for name in sorted(os.listdir(path)):
                    if name.lower().endswith((".ttf", ".otf")):
                        found.append(os.path.join(path, name))
            elif os.path.exists(path):
                found.append(path)
        if not found:
            raise RuntimeError(f"Không tìm thấy font nào trong {self.font_paths}")
        return found

    def generate(self):
        data = self.sampler.sample()
        cells, ncols, nrows = self.layout.generate(data)

        font_path = self._fonts[np.random.randint(len(self._fonts))]
        size = int(np.random.randint(self.font_size[0], self.font_size[1] + 1))
        bold = bool(np.random.rand() < self.bold)
        spacing = float(np.random.uniform(*self.line_spacing))
        gray = int(np.random.randint(self.ink[0], self.ink[1] + 1))
        color = (gray, gray, gray, 255)
        font = {"path": font_path, "size": size, "bold": bold}

        # bề rộng một ô ký tự — font mono nên mọi ký tự bằng nhau
        probe = layers.TextLayer("0" * 10, **font, color=color)
        char_w = probe.width / 10.0
        line_h = probe.height * spacing

        margin = float(np.random.uniform(*self.margin))
        pad_x = ncols * char_w * margin
        pad_y = line_h * np.random.uniform(0.5, 2.0)
        width = int(ncols * char_w + pad_x * 2)
        height = int(nrows * line_h + pad_y * 2)

        text_layers, fields = [], []
        for cell in cells:
            layer = layers.TextLayer(cell["text"], **font, color=color)
            if cell["scale"] != 1.0:
                layer.size = layer.size * cell["scale"]
            y = pad_y + cell["row"] * line_h
            x0 = pad_x + cell["col0"] * char_w
            x1 = pad_x + cell["col1"] * char_w
            # chữ phóng to (tên quán, dòng tổng tiền) không được tràn khỏi cột
            span = max(x1 - x0, 1.0)
            if layer.width > span:
                layer.size = layer.size * (span / layer.width)
            if cell["align"] == "left":
                layer.left = x0
            elif cell["align"] == "right":
                layer.right = x1
            else:
                layer.centerx = (x0 + x1) / 2
            layer.top = y
            text_layers.append(layer)
            fields.append({"text": cell["text"], "kind": cell["kind"]})

        return {
            "size": (width, height),
            "text_layers": text_layers,
            "fields": fields,
            "data": data,
        }

    @staticmethod
    def to_gt_parse(data):
        """Đổi dữ liệu đã sinh sang gt_parse lồng nhau kiểu CORD."""
        parse = {"store": dict(data["store"])}
        parse["menu"] = [dict(item) for item in data["items"]]
        parse["total"] = dict(data["total"])
        return parse

    @staticmethod
    def to_text_sequence(fields):
        """Nhãn đọc-trơn cho bài pre-training text reading."""
        parts = [f["text"] for f in fields if f["kind"] != "sep"]
        return " ".join(parts)
