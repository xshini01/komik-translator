import cv2
import numpy as np
from typing import Tuple


def process_bubble(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Processes a speech bubble crop: finds the bubble contour and
    fills its interior with white (erasing original text).

    Handles both light and dark-background bubbles using adaptive
    thresholding as a fallback.

    Parameters:
        image (np.ndarray): BGR image crop of a detected speech bubble.

    Returns:
        image (np.ndarray): Image with the speech bubble interior set to white.
        largest_contour (np.ndarray): Contour of the detected speech bubble.
                                      Falls back to a full-frame rectangle if none found.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # --- Try simple binary threshold (works for white/light bubbles) ---
    _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # --- Fallback: adaptive threshold (works for darker or colored bubbles) ---
    if not contours:
        thresh = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11, 2
        )
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # --- Second fallback: treat entire crop as bubble area ---
    if not contours:
        h, w = image.shape[:2]
        largest_contour = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.int32).reshape(-1, 1, 2)
        image[:, :] = (255, 255, 255)
        return image, largest_contour

    largest_contour = max(contours, key=cv2.contourArea)

    mask = np.zeros_like(gray)
    cv2.drawContours(mask, [largest_contour], -1, 255, cv2.FILLED)
    image[mask == 255] = (255, 255, 255)

    return image, largest_contour
