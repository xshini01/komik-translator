import os
import img2pdf
from natsort import natsorted
from pathlib import Path

SAVE_DIR = "save_images"
OUTPUT_DIR = "Hasil-PDF"
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}


def compress_toPDF() -> str:
    """
    Compress all translated images in SAVE_DIR into a single PDF.
    Returns:
        str: Path to the generated PDF file.
    Raises:
        RuntimeError: If no images are found.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    existing = sum(1 for e in os.scandir(OUTPUT_DIR) if e.is_file())
    pdf_path = os.path.join(OUTPUT_DIR, f"translated_{existing + 1:04d}.pdf")

    images = natsorted([
        os.path.join(SAVE_DIR, f)
        for f in os.listdir(SAVE_DIR)
        if Path(f).suffix.lower() in IMAGE_EXTENSIONS
    ])

    if not images:
        raise RuntimeError(f"No images found in '{SAVE_DIR}' to compress.")
    with open(pdf_path, "wb") as f:
        f.write(img2pdf.convert(images))

    print(f"[compress] PDF created: {pdf_path} ({len(images)} pages)")
    return pdf_path