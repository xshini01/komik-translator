from PIL import Image, ImageDraw, ImageFont
import numpy as np
import textwrap
import cv2
from typing import Tuple


def add_text(
    image: np.ndarray,
    text: str,
    font_path: str,
    bubble_contour: np.ndarray,
    min_font_size: int = 10,
    max_font_size: int = 52,
) -> np.ndarray:
    """
    Render translated text inside a speech bubble.

    Uses a binary search over font sizes for efficient fitting, then
    centers the text block within the bubble's bounding rectangle.

    Args:
        image (np.ndarray): BGR image of the bubble crop (already whitened).
        text (str): Translated text to render.
        font_path (str): Path to a .ttf font file.
        bubble_contour (np.ndarray): Contour of the bubble interior.
        min_font_size (int): Minimum allowed font size.
        max_font_size (int): Maximum allowed font size.

    Returns:
        np.ndarray: Image with text rendered.
    """
    if not text or not text.strip():
        return image

    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_image)

    x, y, w, h = cv2.boundingRect(bubble_contour)

    # Leave a small margin inside the bubble
    padding = max(4, int(min(w, h) * 0.06))
    avail_w = w - 2 * padding
    avail_h = h - 2 * padding

    if avail_w <= 0 or avail_h <= 0:
        return image

    best_font_size, best_lines = _find_best_font(
        draw, text, font_path, avail_w, avail_h,
        min_font_size, max_font_size
    )

    font = ImageFont.truetype(font_path, size=best_font_size)
    line_height = int(best_font_size * 1.25)
    total_text_height = len(best_lines) * line_height

    # Vertically center text block
    text_y = y + padding + max(0, (avail_h - total_text_height) // 2)

    for line in best_lines:
        line_w = draw.textlength(line, font=font)
        text_x = x + padding + max(0, (avail_w - line_w) // 2)
        draw.text((text_x, text_y), line, font=font, fill=(0, 0, 0))
        text_y += line_height

    result = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    image[:, :, :] = result
    return image


def _find_best_font(
    draw: ImageDraw.ImageDraw,
    text: str,
    font_path: str,
    avail_w: int,
    avail_h: int,
    min_size: int,
    max_size: int,
) -> Tuple[int, list[str]]:
    """
    Binary-search for the largest font size where the wrapped text fits
    within (avail_w x avail_h).

    Returns:
        (font_size, wrapped_lines)
    """
    best_size = min_size
    best_lines: list[str] = [text]

    lo, hi = min_size, max_size
    while lo <= hi:
        mid = (lo + hi) // 2
        font = ImageFont.truetype(font_path, size=mid)
        line_height = int(mid * 1.25)

        # Estimate chars per line from average char width
        avg_char_w = max(1, draw.textlength("x", font=font))
        chars_per_line = max(3, int(avail_w / avg_char_w))
        wrapped = textwrap.wrap(text, width=chars_per_line, break_long_words=True)
        if not wrapped:
            wrapped = [text]

        total_h = len(wrapped) * line_height
        fits_h = total_h <= avail_h
        fits_w = all(draw.textlength(l, font=font) <= avail_w for l in wrapped)

        if fits_h and fits_w:
            best_size = mid
            best_lines = wrapped
            lo = mid + 1
        else:
            hi = mid - 1

    return best_size, best_lines
