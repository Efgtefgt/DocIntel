"""Render the app's SVG icon to a multi-size Windows .ico (build step; needs PyMuPDF and Pillow)."""
import io
import sys

import fitz  # PyMuPDF
from PIL import Image

svg_path, ico_path = sys.argv[1], sys.argv[2]
page = fitz.open(svg_path)[0]
pix = page.get_pixmap(matrix=fitz.Matrix(256 / page.rect.width, 256 / page.rect.height), alpha=True)
image = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGBA")
image.save(ico_path, sizes=[(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)])
print(f"Wrote {ico_path}")
