"""Generate small synthetic images and PDFs with known text.

Used by the test suite and by ``scripts/generate_samples.py``. Text is rendered with
Pillow's bundled font, so output is reproducible without system fonts or downloads.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Sequence

import fitz  # PyMuPDF
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    return ImageFont.load_default(size=size)


def render_text_image(
    lines: Sequence[str], *, font_size: int = 36, padding: int = 48, line_spacing: float = 1.6, width: int | None = None
) -> Image.Image:
    """Render black text on white, one entry per line."""
    font = _font(font_size)
    probe = ImageDraw.Draw(Image.new("L", (1, 1)))
    text_width = max((probe.textlength(line, font=font) for line in lines), default=0)
    line_height = int(font_size * line_spacing)
    img_w = width or int(text_width + 2 * padding)
    img_h = int(len(lines) * line_height + 2 * padding)
    img = Image.new("RGB", (max(img_w, 16), max(img_h, 16)), "white")
    draw = ImageDraw.Draw(img)
    for i, line in enumerate(lines):
        draw.text((padding, padding + i * line_height), line, fill="black", font=font)
    return img


def render_table_image(rows: Sequence[Sequence[str]], *, font_size: int = 32, cell_padding: int = 20, ruled: bool = True) -> Image.Image:
    """Render a grid table (with or without ruling lines)."""
    font = _font(font_size)
    probe = ImageDraw.Draw(Image.new("L", (1, 1)))
    n_cols = max(len(r) for r in rows)
    col_widths = [
        int(max(probe.textlength(r[c] if c < len(r) else "", font=font) for r in rows)) + 2 * cell_padding + (0 if ruled else 40)
        for c in range(n_cols)
    ]
    row_h = font_size + 2 * cell_padding
    margin = 40
    img = Image.new("RGB", (sum(col_widths) + 2 * margin, row_h * len(rows) + 2 * margin), "white")
    draw = ImageDraw.Draw(img)
    for r, row in enumerate(rows):
        x = margin
        for c in range(n_cols):
            text = row[c] if c < len(row) else ""
            draw.text((x + cell_padding, margin + r * row_h + cell_padding), text, fill="black", font=font)
            x += col_widths[c]
    if ruled:
        right, bottom = margin + sum(col_widths), margin + row_h * len(rows)
        for r in range(len(rows) + 1):
            draw.line([(margin, margin + r * row_h), (right, margin + r * row_h)], fill="black", width=3)
        x = margin
        for c in range(n_cols + 1):
            draw.line([(x, margin), (x, bottom)], fill="black", width=3)
            x += col_widths[c] if c < n_cols else 0
    return img


def degrade(img: Image.Image, *, rotate: float = 0.0, noise: float = 0.0, blur: float = 0.0, scale: float = 1.0, seed: int = 0) -> Image.Image:
    """Apply controlled degradations: small rotation, Gaussian noise, blur and downscaling."""
    out = img.convert("L")
    if rotate:
        out = out.rotate(rotate, resample=Image.BICUBIC, expand=True, fillcolor=255)
    if blur:
        out = out.filter(ImageFilter.GaussianBlur(blur))
    if scale != 1.0:
        out = out.resize((max(1, int(out.width * scale)), max(1, int(out.height * scale))), Image.LANCZOS)
    if noise:
        rng = np.random.default_rng(seed)
        arr = np.asarray(out, dtype=np.float32) + rng.normal(0, noise, (out.height, out.width))
        out = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    return out


def image_bytes(img: Image.Image, fmt: str = "PNG") -> bytes:
    buffer = io.BytesIO()
    save_img = img.convert("RGB") if fmt.upper() in ("JPEG", "BMP") and img.mode not in ("RGB", "L") else img
    save_img.save(buffer, format=fmt)
    return buffer.getvalue()


def make_digital_pdf(pages: Sequence[str]) -> bytes:
    """PDF whose pages contain real (embedded) text."""
    doc = fitz.open()
    for text in pages:
        page = doc.new_page(width=595, height=842)
        page.insert_textbox(fitz.Rect(56, 56, 540, 790), text, fontsize=12, fontname="helv")
    data = doc.tobytes()
    doc.close()
    return data


def make_scanned_pdf(images: Sequence[Image.Image]) -> bytes:
    """PDF whose pages are images only (no embedded text), like a scan."""
    doc = fitz.open()
    for img in images:
        page = doc.new_page(width=595, height=842)
        page.insert_image(fitz.Rect(0, 0, 595, 842), stream=image_bytes(img.convert("RGB"), "PNG"), keep_proportion=True)
    data = doc.tobytes()
    doc.close()
    return data


def make_mixed_pdf(digital_text: str, scanned_image: Image.Image) -> bytes:
    """Page 1 has embedded text; page 2 is an image-only page."""
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    page.insert_textbox(fitz.Rect(56, 56, 540, 790), digital_text, fontsize=12, fontname="helv")
    page = doc.new_page(width=595, height=842)
    page.insert_image(fitz.Rect(0, 0, 595, 842), stream=image_bytes(scanned_image.convert("RGB")), keep_proportion=True)
    data = doc.tobytes()
    doc.close()
    return data


def make_table_pdf(rows: Sequence[Sequence[str]]) -> bytes:
    """Digital PDF with a ruled table drawn from vector lines and text."""
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    n_cols = max(len(r) for r in rows)
    x0, y0, col_w, row_h = 60, 80, 150, 28
    for r, row in enumerate(rows):
        for c in range(n_cols):
            rect = fitz.Rect(x0 + c * col_w, y0 + r * row_h, x0 + (c + 1) * col_w, y0 + (r + 1) * row_h)
            page.draw_rect(rect, color=(0, 0, 0), width=0.8)
            if c < len(row) and row[c]:
                page.insert_text((rect.x0 + 6, rect.y1 - 9), row[c], fontsize=11, fontname="helv")
    data = doc.tobytes()
    doc.close()
    return data


BENCHMARK_SAMPLES: dict[str, dict] = {
    "clean_paragraph": {"lines": ["The quick brown fox jumps over the lazy dog.", "Invoice number 48213 was paid on 12 March 2024.", "Contact: support@example.com"]},
    "small_text": {"lines": ["Small text must be upscaled before OCR.", "Reference ABC-1234"], "font_size": 14},
    "rotated": {"lines": ["This page was scanned at a slight angle.", "Deskewing should straighten the lines."], "rotate": 4.0},
    "noisy": {"lines": ["Noise reduction helps with grainy scans.", "Total amount: 1,250.00"], "noise": 40.0, "seed": 7},
    "incident_report": {
        "lines": [
            "INCIDENT REPORT",
            "Caller Name: Priya Sharma",
            "Phone Number: +91 98450 12345",
            "Incident Location: 14 MG Road, Bengaluru",
            "Incident Type: Road accident",
            "Incident Date: 2024-05-17",
            "Description: Two-wheeler collided with a parked car.",
        ]
    },
}


def render_scene(lines: Sequence[str], *, font_size: int = 32, text_color: tuple[int, int, int] = (20, 20, 20),
                 background: tuple[int, int, int] = (255, 255, 255), padding: int = 56) -> Image.Image:
    """Render coloured text on a coloured background (for screenshot / paper simulations)."""
    font = _font(font_size)
    probe = ImageDraw.Draw(Image.new("L", (1, 1)))
    width = int(max(probe.textlength(line, font=font) for line in lines) + 2 * padding)
    line_h = int(font_size * 1.6)
    img = Image.new("RGB", (width, len(lines) * line_h + 2 * padding), background)
    draw = ImageDraw.Draw(img)
    for i, line in enumerate(lines):
        draw.text((padding, padding + i * line_h), line, fill=text_color, font=font)
    return img


def photo_effects(img: Image.Image, *, shadow: float = 0.0, contrast: float = 1.0, blur: float = 0.0,
                  noise: float = 0.0, jpeg_quality: int | None = None, rotate: float = 0.0,
                  border: int = 0, seed: int = 0) -> Image.Image:
    """Camera/scanner-like degradations on an RGB image.

    ``shadow`` (0-1) darkens one side with a smooth gradient (uneven lighting); ``contrast`` < 1
    pulls text and background towards mid-grey; ``border`` adds a dark scanner frame.
    """
    rng = np.random.default_rng(seed)
    arr = np.asarray(img.convert("RGB"), dtype=np.float32)
    if contrast != 1.0:
        arr = 128 + (arr - 128) * contrast
    if shadow:
        h, w = arr.shape[:2]
        gx = np.linspace(1.0, 1.0 - shadow, w)[None, :]
        gy = np.linspace(1.0, 1.0 - shadow * 0.5, h)[:, None]
        arr = arr * (gx * gy)[:, :, None]
    if noise:
        arr = arr + rng.normal(0, noise, arr.shape)
    out = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    if blur:
        out = out.filter(ImageFilter.GaussianBlur(blur))
    if rotate:
        out = out.rotate(rotate, resample=Image.BICUBIC, expand=True, fillcolor=(255, 255, 255))
    if border:
        framed = Image.new("RGB", (out.width + 2 * border, out.height + 2 * border), (25, 25, 25))
        framed.paste(out, (border, border))
        out = framed
    if jpeg_quality:
        buffer = io.BytesIO()
        out.save(buffer, format="JPEG", quality=jpeg_quality)
        out = Image.open(io.BytesIO(buffer.getvalue())).convert("RGB")
    return out


FORM_LINES = ["Incident Report No. INC-1024", "Location: MG Road, Bengaluru", "Incident Type: Fire",
              "Priority: High    Status: Open", "Reported on 17 March 2024 at 14:35"]
NOTE_LINES = ["Please submit the signed copy by Friday.", "Contact the help desk on 080 2345 6789",
              "or write to support@example.com for help."]

# Harder, "real-world-like" synthetic conditions. Still synthetic: they test that preprocessing
# reacts to each condition, not real-world accuracy (add real scans for that, see EVALUATION.md).
HARD_SAMPLES: dict[str, dict] = {
    "shadow_photo": {"lines": FORM_LINES, "effects": {"shadow": 0.75, "noise": 6, "seed": 1}},
    "low_contrast": {"lines": NOTE_LINES, "scene": {"text_color": (130, 130, 130), "background": (175, 175, 175)}},
    "blur_jpeg": {"lines": FORM_LINES, "effects": {"blur": 1.3, "jpeg_quality": 18, "seed": 2}},
    "dark_mode_screenshot": {"lines": NOTE_LINES, "scene": {"text_color": (215, 215, 215), "background": (30, 30, 36)}},
    "small_noisy_print": {"lines": NOTE_LINES, "scene": {"font_size": 13}, "effects": {"noise": 14, "seed": 3}},
    "sideways_scan": {"lines": FORM_LINES, "effects": {"rotate": 90}},
    "colored_paper": {"lines": FORM_LINES, "scene": {"text_color": (30, 50, 140), "background": (250, 236, 170)}},
    "scanner_border_skew": {"lines": NOTE_LINES, "effects": {"rotate": 3.5, "border": 60, "noise": 5, "seed": 4}},
    "phone_photo": {"lines": FORM_LINES, "effects": {"shadow": 0.5, "rotate": 2.5, "blur": 0.8, "noise": 8,
                                                     "jpeg_quality": 40, "contrast": 0.8, "seed": 5}},
    "clean_form": {"lines": FORM_LINES, "effects": {}},
}


HOLDOUT_TEXTS = [
    ["Delivery Note DN-55871", "Customer: Rao Logistics Pvt Ltd", "Ship to: 22 Residency Road, Chennai",
     "Items: 14 cartons, 3 pallets", "Received by: K. Menon"],
    ["Meeting minutes, 4 June 2025", "Attendees: Priya, Arjun, Meera", "Budget approved: 4,75,000 INR",
     "Next review on 18 June 2025"],
    ["Patient ID: PT-20931", "Blood pressure: 128/84 mmHg", "Prescribed: Amoxicillin 500 mg",
     "Follow-up in two weeks"],
    ["Warranty card", "Model: XR-7 Pro, Serial 7731-QZ", "Purchased: 02/11/2024", "Valid for 24 months"],
]


def holdout_specs(count: int = 12, seed: int = 2026) -> dict[str, dict]:
    """Randomised combinations of degradations with unseen text, for a held-out evaluation.

    These were never used to tune the smart preprocessing thresholds.
    """
    rng = np.random.default_rng(seed)
    specs: dict[str, dict] = {}
    for i in range(count):
        lines = HOLDOUT_TEXTS[i % len(HOLDOUT_TEXTS)]
        effects: dict = {"seed": int(rng.integers(0, 10_000))}
        scene: dict = {"font_size": int(rng.choice([14, 18, 24, 30, 36]))}
        if rng.random() < 0.5:
            effects["shadow"] = float(rng.uniform(0.3, 0.8))
        if rng.random() < 0.5:
            effects["rotate"] = float(rng.choice([rng.uniform(-5, 5), 90, 180, 270]))
        if rng.random() < 0.4:
            effects["blur"] = float(rng.uniform(0.5, 1.5))
        if rng.random() < 0.5:
            effects["noise"] = float(rng.uniform(3, 15))
        if rng.random() < 0.4:
            effects["jpeg_quality"] = int(rng.integers(15, 60))
        if rng.random() < 0.3:
            effects["border"] = int(rng.integers(20, 80))
        if rng.random() < 0.3:
            effects["contrast"] = float(rng.uniform(0.5, 0.9))
        if rng.random() < 0.2:
            scene.update(text_color=(220, 220, 220), background=(28, 30, 36))
        specs[f"holdout_{i:02d}"] = {"lines": lines, "scene": scene, "effects": effects}
    return specs


def render_hard_sample(spec: dict) -> Image.Image:
    img = render_scene(spec["lines"], **spec.get("scene", {}))
    return photo_effects(img, **spec.get("effects", {}))


# Typed documents for the document-intelligence benchmark (expected type, text, rendering).
DOCUMENT_SAMPLES: dict[str, dict] = {
    "invoice_clean": {"type": "invoice", "lines": [
        "TAX INVOICE", "Acme Traders Pvt Ltd, 12 MG Road, Bengaluru", "GSTIN: 29AAGCB7383J1Z4",
        "Invoice No: INV-2024-0042    Invoice Date: 17/05/2024", "Bill To: Ravi Kumar, Phone: +91 98450 12345",
        "Steel bolts   Qty 250   Rate 12.50   Amount 3,125.00", "CGST 9%: 281.25    SGST 9%: 281.25",
        "Grand Total: Rs. 3,687.50", "Email: accounts@acme.example.com"], "effects": {}},
    "invoice_photo": {"type": "invoice", "lines": [
        "TAX INVOICE", "Sri Balaji Stores, Mysuru", "GSTIN: 27AAPFU0939F1ZV",
        "Invoice No: SBS/1187    Date: 03 March 2025", "Rice 25 kg   Qty 2   Amount Rs. 2,400.00",
        "Oil 5 L   Qty 1   Amount Rs. 890.00", "Grand Total: Rs. 3,290.00", "Phone: 0821 245 6789"],
        "effects": {"shadow": 0.5, "rotate": 2.0, "blur": 0.7, "noise": 7, "jpeg_quality": 45, "seed": 11}},
    "receipt": {"type": "receipt", "lines": [
        "PAYMENT RECEIPT", "Receipt No: R-55120", "Received with thanks from Meera Iyer",
        "Amount Paid: Rs. 1,500/-", "Payment Mode: UPI   Txn ID: 412233987711", "Date: 2025-01-09  Time: 4:15 PM"],
        "effects": {"noise": 10, "seed": 12}},
    "payslip": {"type": "payslip", "lines": [
        "SALARY SLIP - JUNE 2025", "Employee Name: Anita Rao    Employee ID: EMP0042",
        "Basic: 40,000    HRA: 16,000", "Provident Fund: 4,800    Professional Tax: 200",
        "Gross Salary: 56,000    Net Pay: 51,000", "Bank A/c No: 50100234567890   IFSC: HDFC0001234"],
        "effects": {"shadow": 0.6, "seed": 13}},
    "bank_statement": {"type": "bank_statement", "lines": [
        "STATEMENT OF ACCOUNT", "Account Number: 012345678901    IFSC: SBIN0004567",
        "Opening Balance: 25,000.00", "05/06/2025  ATM Withdrawal   2,000.00", "12/06/2025  Salary Credit  51,000.00",
        "Closing Balance: 74,000.00"], "effects": {"jpeg_quality": 30, "seed": 14}},
    "letter": {"type": "letter", "lines": [
        "Dear Sir,", "Subject: Request for leave", "I request leave from 3 June 2025 to 7 June 2025.",
        "I can be reached on 98861 23456 or ramesh.n@example.org.", "Yours faithfully,", "Ramesh N"],
        "effects": {"rotate": 1.5, "seed": 15}},
    "incident": {"type": "incident_report", "lines": [
        "INCIDENT REPORT", "Caller Name: Priya Sharma", "Phone Number: +91 98450 12345",
        "Location: Hosur Road, Bengaluru", "Incident Type: Road accident   Priority: High",
        "Vehicle No: KA 05 MH 2231", "Reported on 17 March 2025 at 14:35"], "effects": {"blur": 1.0, "seed": 16}},
    "resume": {"type": "resume", "lines": [
        "CURRICULUM VITAE", "Arjun Mehta", "Email: arjun.mehta@example.com   Mobile: 98450 67890",
        "Education: B.E. Computer Science, 2021", "Work Experience: 3 years, Python developer",
        "Skills: Python, SQL, OCR", "Projects: Document scanner app"], "effects": {"noise": 6, "seed": 17}},
    "utility_bill": {"type": "utility_bill", "lines": [
        "ELECTRICITY BILL", "Consumer No: 7712045531", "Billing Period: 01/05/2025 to 31/05/2025",
        "Units Consumed: 312", "Bill Amount: Rs. 2,184.00", "Due Date: 2025-06-15"],
        "scene": {"text_color": (110, 110, 110), "background": (190, 190, 190)}, "effects": {"seed": 18}},
    "delivery_challan": {"type": "delivery_note", "lines": [
        "DELIVERY CHALLAN", "Challan No: DC-7781    Date: 21/04/2025", "Consignee: Rao Logistics Pvt Ltd",
        "Transporter: VRL   Vehicle No: KA 01 AB 1234", "No of packages: 14", "Received by: K. Menon"],
        "effects": {"shadow": 0.4, "noise": 5, "seed": 19}},
    "pan_card": {"type": "identity_document", "lines": [
        "INCOME TAX DEPARTMENT", "GOVT. OF INDIA", "Permanent Account Number Card", "ABCPK1234F",
        "Name: SURESH KUMAR", "Father's Name: RAMESH KUMAR", "Date of Birth: 14/08/1986"],
        "scene": {"text_color": (20, 40, 90), "background": (225, 235, 245)}, "effects": {"seed": 20}},
    "prescription": {"type": "medical", "lines": [
        "PRESCRIPTION", "City Clinic, Jayanagar", "Patient: Lakshmi Devi   Age: 54",
        "Diagnosis: Hypertension", "Tab. Amlodipine 5 mg - once daily", "Follow up after 2 weeks",
        "Doctor: Dr. S. Rao, Phone: 080 2663 4410"], "effects": {"rotate": -2.0, "noise": 6, "seed": 21}},
}


def write_benchmark_dataset(directory: Path, suite: str = "basic") -> list[Path]:
    """Write ``<name>.png`` + ``<name>.gt.txt`` pairs and return the image paths.

    ``suite`` is ``"basic"`` (clean renders with single degradations), ``"hard"`` (combined
    photo/scan/screenshot conditions), ``"documents"`` (typed documents; also writes
    ``<name>.type.txt``) or one of the held-out suites.
    """
    directory.mkdir(parents=True, exist_ok=True)
    written = []
    samples = {"basic": BENCHMARK_SAMPLES, "hard": HARD_SAMPLES, "documents": DOCUMENT_SAMPLES}.get(suite)
    if samples is None:
        # "holdout" (seed 2026, looked at once during development) or "holdout2" (seed 2027, fresh check)
        seeds = {"holdout": 2026, "holdout2": 2027, "holdout3": 2028}
        samples = holdout_specs(seed=seeds.get(suite, 2026))
    for name, spec in samples.items():
        if suite == "basic":
            img = render_text_image(spec["lines"], font_size=spec.get("font_size", 36))
            img = degrade(img, rotate=spec.get("rotate", 0.0), noise=spec.get("noise", 0.0), seed=spec.get("seed", 0))
        else:
            img = render_hard_sample(spec)
        path = directory / f"{name}.png"
        img.save(path)
        (directory / f"{name}.gt.txt").write_bytes(("\n".join(spec["lines"]) + "\n").encode("utf-8"))
        if "type" in spec:
            (directory / f"{name}.type.txt").write_bytes((spec["type"] + "\n").encode("utf-8"))
        written.append(path)
    return written
