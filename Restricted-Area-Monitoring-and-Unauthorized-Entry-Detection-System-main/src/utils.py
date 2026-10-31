"""
Utility Functions
=================
Provides helper functions used across all modules in the
Restricted Area Monitoring & Unauthorized Entry Detection System.

Functions:
    - load_config: Load YAML configuration file
    - draw_bbox: Draw a bounding box with label on a frame
    - draw_detection_info: Draw an info text panel on a frame
    - get_color_for_class: Get BGR color for a detection class
    - create_output_directory: Create output directory if it doesn't exist
    - format_timestamp: Convert seconds to HH:MM:SS string
    - resize_with_aspect_ratio: Resize image while preserving aspect ratio
"""

import os
from pathlib import Path
from typing import Optional, Tuple, Dict

import cv2
import numpy as np
import yaml


# ============================================================
# Configuration Loading
# ============================================================

def load_config(config_path: str) -> dict:
    """
    Load a YAML configuration file and return its contents as a dictionary.

    Args:
        config_path: Absolute or relative path to the YAML config file.

    Returns:
        Dictionary containing the parsed configuration.

    Raises:
        FileNotFoundError: If the config file does not exist.
        yaml.YAMLError: If the file contains invalid YAML.
    """
    config_file = Path(config_path)

    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_file, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if config is None:
        return {}

    return config


# ============================================================
# Drawing Helpers
# ============================================================

# Pre-defined BGR color map for each detection class category
_CLASS_COLOR_MAP: Dict[str, Tuple[int, int, int]] = {
    "person":  (0, 0, 255),     # Red
    "vehicle": (255, 165, 0),   # Orange
    "animal":  (0, 255, 0),     # Green
}

# Default color for unknown classes
_DEFAULT_COLOR: Tuple[int, int, int] = (255, 255, 255)  # White


def get_color_for_class(class_name: str) -> Tuple[int, int, int]:
    """
    Return a BGR colour tuple for a given detection class category.

    Colour mapping:
        - person  -> Red   (0, 0, 255)
        - vehicle -> Orange (255, 165, 0)
        - animal  -> Green (0, 255, 0)

    Args:
        class_name: The class category name (e.g. "person", "vehicle", "animal").

    Returns:
        BGR colour tuple.
    """
    return _CLASS_COLOR_MAP.get(class_name.lower(), _DEFAULT_COLOR)


def draw_bbox(
    frame: np.ndarray,
    bbox: Tuple[int, int, int, int],
    label: str = "",
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
) -> np.ndarray:
    """
    Draw a bounding box with an optional label on the frame.

    The label is rendered above the top-left corner of the bounding box
    with a filled background rectangle for readability.

    Args:
        frame: The image (BGR numpy array) to draw on.
        bbox: Bounding box as (x1, y1, x2, y2).
        label: Text label to display above the box.
        color: BGR colour for the box and label background.
        thickness: Line thickness for the bounding box.

    Returns:
        The frame with the bounding box and label drawn on it.
    """
    x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])

    # Draw the bounding box rectangle
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

    # Draw the label if provided
    if label:
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_thickness = 1

        # Measure text size to draw a background rectangle
        (text_w, text_h), baseline = cv2.getTextSize(
            label, font, font_scale, font_thickness
        )

        # Position the label above the bounding box (clamp to top of image)
        label_y = max(y1 - 6, text_h + 4)

        # Filled rectangle behind the text
        cv2.rectangle(
            frame,
            (x1, label_y - text_h - 4),
            (x1 + text_w + 4, label_y + baseline),
            color,
            cv2.FILLED,
        )

        # White text for readability
        cv2.putText(
            frame,
            label,
            (x1 + 2, label_y - 2),
            font,
            font_scale,
            (255, 255, 255),
            font_thickness,
            cv2.LINE_AA,
        )

    return frame


def draw_detection_info(
    frame: np.ndarray,
    text: str,
    position: Tuple[int, int] = (10, 30),
    color: Tuple[int, int, int] = (0, 255, 0),
) -> np.ndarray:
    """
    Draw an information text panel on the frame.

    Renders text with a semi-transparent dark background panel to ensure
    readability regardless of the underlying image content.

    Args:
        frame: The image (BGR numpy array) to draw on.
        text: The information text to display.
        position: Top-left (x, y) position for the text.
        color: BGR colour for the text.

    Returns:
        The frame with the info panel drawn on it.
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    font_thickness = 1

    # Measure text for the background panel
    (text_w, text_h), baseline = cv2.getTextSize(
        text, font, font_scale, font_thickness
    )

    x, y = position
    padding = 6

    # Draw semi-transparent dark background
    overlay = frame.copy()
    cv2.rectangle(
        overlay,
        (x - padding, y - text_h - padding),
        (x + text_w + padding, y + baseline + padding),
        (0, 0, 0),
        cv2.FILLED,
    )
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    # Draw the text
    cv2.putText(
        frame, text, (x, y), font, font_scale, color, font_thickness, cv2.LINE_AA
    )

    return frame


# ============================================================
# File / Directory Helpers
# ============================================================

def create_output_directory(path: str) -> str:
    """
    Create the specified output directory (including parents) if it does
    not already exist.

    Args:
        path: Path to the directory to create.

    Returns:
        The absolute path string of the created directory.
    """
    dir_path = Path(path)
    dir_path.mkdir(parents=True, exist_ok=True)
    return str(dir_path.resolve())


# ============================================================
# Timestamp Formatting
# ============================================================

def format_timestamp(seconds: float) -> str:
    """
    Convert a duration in seconds to a human-readable HH:MM:SS string.

    Args:
        seconds: Duration in seconds (can be float).

    Returns:
        Formatted string in HH:MM:SS format.
    """
    total_seconds = int(max(seconds, 0))
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


# ============================================================
# Image Resizing
# ============================================================

def resize_with_aspect_ratio(
    image: np.ndarray,
    width: Optional[int] = None,
    height: Optional[int] = None,
    inter: int = cv2.INTER_AREA,
) -> np.ndarray:
    """
    Resize an image while preserving its aspect ratio.

    Exactly one of *width* or *height* must be provided.  The other
    dimension is computed automatically to maintain the original
    aspect ratio.

    Args:
        image: Source image as a numpy array.
        width: Desired width in pixels (or None to auto-calculate).
        height: Desired height in pixels (or None to auto-calculate).
        inter: OpenCV interpolation method (default: INTER_AREA).

    Returns:
        Resized image as a numpy array.

    Raises:
        ValueError: If neither width nor height is specified.
    """
    h, w = image.shape[:2]

    if width is None and height is None:
        raise ValueError("At least one of 'width' or 'height' must be specified.")

    if width is not None and height is not None:
        # Both provided – resize to exact dimensions (caller's intent)
        new_size = (width, height)
    elif width is not None:
        # Calculate height to preserve aspect ratio
        ratio = width / float(w)
        new_size = (width, int(h * ratio))
    else:
        # Calculate width to preserve aspect ratio
        ratio = height / float(h)
        new_size = (int(w * ratio), height)

    return cv2.resize(image, new_size, interpolation=inter)
