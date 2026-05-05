import os
import subprocess
import shutil
from pathlib import Path

EXTRACT_DIR = "folder_ekstrak"
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}


def mangadex_download(link: str) -> None:
    """
    Download a manga/chapter from MangaDex and extract images to EXTRACT_DIR.

    Requires `mangadex-dl` CLI tool to be installed.

    Args:
        link (str): MangaDex chapter or manga URL.

    Raises:
        RuntimeError: If the download fails or no images are found.
    """
    download_dir = "manga_downloader"
    shutil.rmtree(download_dir, ignore_errors=True)
    os.makedirs(download_dir, exist_ok=True)
    os.makedirs(EXTRACT_DIR, exist_ok=True)

    result = subprocess.run(
        ["mangadex-dl", link],
        cwd=download_dir,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"mangadex-dl failed (code {result.returncode}).\n"
            f"stderr: {result.stderr[:500]}"
        )

    # Collect all image files recursively
    all_images: list[str] = []
    for root, _, files in os.walk(download_dir):
        for f in files:
            if Path(f).suffix.lower() in IMAGE_EXTENSIONS:
                all_images.append(os.path.join(root, f))

    if not all_images:
        shutil.rmtree(download_dir, ignore_errors=True)
        raise RuntimeError("Download succeeded but no images found.")

    # Move images to extract dir with collision handling
    for src in sorted(all_images):
        basename = os.path.basename(src)
        dest = os.path.join(EXTRACT_DIR, basename)
        if os.path.exists(dest):
            stem, ext = os.path.splitext(basename)
            i = 1
            while os.path.exists(dest):
                dest = os.path.join(EXTRACT_DIR, f"{stem}_{i}{ext}")
                i += 1
        shutil.move(src, dest)

    shutil.rmtree(download_dir, ignore_errors=True)
    print(f"[mangadex] Downloaded {len(all_images)} images to '{EXTRACT_DIR}'")
