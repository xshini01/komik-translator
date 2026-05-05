import os
import io
import img2pdf
from natsort import natsorted
from pathlib import Path
from PIL import Image

SAVE_DIR = "save_images"
OUTPUT_DIR = "Hasil-PDF"
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}


def compress_toPDF(jpeg_quality: int = 85) -> str:
    """
    Compress all translated images in SAVE_DIR into a single PDF.

    PNG images are converted to JPEG before embedding so the PDF size stays
    close to (or smaller than) the original input images.

    Args:
        jpeg_quality (int): JPEG quality (1-95).
                            85 = good balance of quality vs size.
                            70-75 = smaller file, 90-95 = higher quality.

    Returns:
        str: Path to the generated PDF file.

    Raises:
        RuntimeError: If no images are found.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    existing = sum(1 for e in os.scandir(OUTPUT_DIR) if e.is_file())
    pdf_path = os.path.join(OUTPUT_DIR, f"translated_{existing + 1:04d}.pdf")

    image_paths = natsorted([
        os.path.join(SAVE_DIR, f)
        for f in os.listdir(SAVE_DIR)
        if Path(f).suffix.lower() in IMAGE_EXTENSIONS
    ])

    if not image_paths:
        raise RuntimeError(f"No images found in '{SAVE_DIR}' to compress.")

    # Convert every image to JPEG in-memory before feeding to img2pdf.
    # img2pdf embeds PNG as-is (no compression), which results in PDFs
    # 3-5x larger than the original JPEG inputs. Converting to JPEG first
    # fixes this.
    jpeg_buffers = []
    for path in image_paths:
        buf = io.BytesIO()
        with Image.open(path) as img:
            # JPEG does not support alpha channel
            if img.mode in ("RGBA", "P", "LA"):
                img = img.convert("RGB")
            elif img.mode != "RGB":
                img = img.convert("RGB")
            img.save(buf, format="JPEG", quality=jpeg_quality, optimize=True)
        buf.seek(0)
        jpeg_buffers.append(buf)

    with open(pdf_path, "wb") as f:
        f.write(img2pdf.convert(jpeg_buffers))

    size_mb = os.path.getsize(pdf_path) / (1024 * 1024)
    print(f"[compress] PDF created: {pdf_path} ({len(image_paths)} pages, {size_mb:.1f} MB)")
    return pdf_path
