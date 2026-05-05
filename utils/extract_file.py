import os
import zipfile
import rarfile
import shutil
from pathlib import Path
from pdf2image import convert_from_path
from PIL import Image

EXTRACT_DIR = "folder_ekstrak"
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}


def _is_image_file(path: str) -> bool:
    """Check if a file is a valid image by extension."""
    return Path(path).suffix.lower() in IMAGE_EXTENSIONS


def _safe_copy(src: str, dest_dir: str) -> None:
    """Copy a file to dest_dir, avoiding filename collisions."""
    basename = os.path.basename(src)
    dest = os.path.join(dest_dir, basename)
    if os.path.exists(dest):
        stem, ext = os.path.splitext(basename)
        i = 1
        while os.path.exists(dest):
            dest = os.path.join(dest_dir, f"{stem}_{i}{ext}")
            i += 1
    shutil.copy2(src, dest)


def extract_file(files) -> None:
    """
    Extract or copy input files into the extraction folder.

    Supports: .zip, .cbz, .rar, .cbr, .pdf, and direct image files.
    CBZ = ZIP-based comic archive; CBR = RAR-based comic archive.

    Args:
        files: A single file path (str) or a list of file paths.
    """
    if isinstance(files, str):
        files = [files]

    os.makedirs(EXTRACT_DIR, exist_ok=True)

    for file in files:
        if not file or not os.path.exists(file):
            print(f"[extract] Skipping missing file: {file}")
            continue

        suffix = Path(file).suffix.lower()

        if suffix in (".zip", ".cbz"):
            _extract_zip(file)

        elif suffix in (".rar", ".cbr"):
            _extract_rar(file)

        elif suffix == ".pdf":
            _extract_pdf(file)

        elif suffix in IMAGE_EXTENSIONS:
            _copy_image(file)

        else:
            print(f"[extract] Unsupported file type: {file}")


def _extract_zip(file: str) -> None:
    try:
        with zipfile.ZipFile(file, "r") as z:
            for member in z.namelist():
                if Path(member).suffix.lower() in IMAGE_EXTENSIONS:
                    target = os.path.join(EXTRACT_DIR, os.path.basename(member))
                    # Avoid path traversal
                    if ".." in member:
                        continue
                    with z.open(member) as src, open(target, "wb") as dst:
                        shutil.copyfileobj(src, dst)
        print(f"[extract] Extracted ZIP: {file}")
    except zipfile.BadZipFile as e:
        print(f"[extract] Bad ZIP file {file}: {e}")


def _extract_rar(file: str) -> None:
    try:
        with rarfile.RarFile(file, "r") as r:
            for member in r.namelist():
                if Path(member).suffix.lower() in IMAGE_EXTENSIONS:
                    target = os.path.join(EXTRACT_DIR, os.path.basename(member))
                    with r.open(member) as src, open(target, "wb") as dst:
                        shutil.copyfileobj(src, dst)
        print(f"[extract] Extracted RAR: {file}")
    except rarfile.Error as e:
        print(f"[extract] Failed to extract RAR {file}: {e}")


def _extract_pdf(file: str) -> None:
    try:
        images = convert_from_path(file, dpi=150)
        for i, img in enumerate(images):
            out_path = os.path.join(EXTRACT_DIR, f"page_{i + 1:04d}.png")
            img.save(out_path, "PNG")
        print(f"[extract] Extracted PDF: {file} ({len(images)} pages)")
    except Exception as e:
        print(f"[extract] Failed to extract PDF {file}: {e}")


def _copy_image(file: str) -> None:
    try:
        with Image.open(file) as img:
            img.verify()
        _safe_copy(file, EXTRACT_DIR)
    except Exception as e:
        print(f"[extract] Invalid image file {file}: {e}")
