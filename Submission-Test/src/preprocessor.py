"""
Frame Preprocessor
==================
Provides the ``FramePreprocessor`` class which applies a configurable
preprocessing pipeline to each video frame before further analysis.

Pipeline steps (applied in order):
    1. Resize to configured dimensions
    2. Gaussian blur for noise reduction
    3. CLAHE contrast enhancement in LAB colour space

Usage::

    from src.preprocessor import FramePreprocessor
    from src.utils import load_config

    config = load_config("config/settings.yaml")
    preprocessor = FramePreprocessor(config["preprocessing"])
    processed = preprocessor.preprocess(frame)
"""

from typing import Dict, Tuple

import cv2
import numpy as np


class FramePreprocessor:
    """
    Applies a three-stage preprocessing pipeline to video frames.

    The pipeline parameters are driven by the ``preprocessing`` section
    of the project configuration file (settings.yaml).

    Attributes:
        resize_width: Target width after resizing.
        resize_height: Target height after resizing.
        blur_kernel: Gaussian blur kernel size (must be odd).
        clahe_clip: CLAHE clip limit for contrast enhancement.
        clahe_grid: CLAHE tile grid size.
    """

    def __init__(self, config: Dict) -> None:
        """
        Initialise the preprocessor with values from the config dictionary.

        Args:
            config: The ``preprocessing`` section of settings.yaml, expected
                     keys: resize_width, resize_height, gaussian_blur_kernel,
                     clahe_clip_limit, clahe_tile_grid_size, enable_contrast_stretching,
                     enable_sharpening, gamma.
        """
        # --- Resize parameters ---
        self.resize_width: int = config.get("resize_width", 640)
        self.resize_height: int = config.get("resize_height", 480)

        # --- Gaussian blur parameters ---
        kernel = config.get("gaussian_blur_kernel", 5)
        # Ensure the kernel size is a positive odd number
        self.blur_kernel: int = kernel if kernel % 2 == 1 else kernel + 1

        # --- CLAHE (Contrast Limited Adaptive Histogram Equalisation) ---
        self.clahe_clip: float = config.get("clahe_clip_limit", 2.0)
        self.clahe_grid: int = config.get("clahe_tile_grid_size", 8)

        # Pre-create the CLAHE object (reusable across frames)
        self._clahe = cv2.createCLAHE(
            clipLimit=self.clahe_clip,
            tileGridSize=(self.clahe_grid, self.clahe_grid),
        )

        # --- Dynamic Image Enhancements ---
        self.enable_contrast_stretching: bool = config.get("enable_contrast_stretching", False)
        self.enable_sharpening: bool = config.get("enable_sharpening", False)
        self.gamma: float = config.get("gamma", 1.0)

        # Pre-build lookup table for Gamma correction
        if self.gamma != 1.0:
            invGamma = 1.0 / self.gamma
            self._gamma_lut = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        else:
            self._gamma_lut = None

        # Sharpening kernel
        self._sharpening_kernel = np.array([
            [0, -1, 0],
            [-1, 5, -1],
            [0, -1, 0]
        ], dtype=np.float32)

    # ============================================================
    # Public Interface
    # ============================================================

    def preprocess(self, frame: np.ndarray) -> np.ndarray:
        """
        Run the full preprocessing pipeline on a single frame.

        Steps applied in order:
            1. Resize to (resize_width × resize_height)
            2. Gaussian blur for noise reduction
            3. Contrast Stretching (if enabled)
            4. CLAHE contrast enhancement on the L channel of LAB
            5. Gamma Correction (if enabled)
            6. Image Sharpening (if enabled)

        Args:
            frame: Input BGR image as a numpy array.

        Returns:
            Preprocessed BGR image as a numpy array.
        """
        result = self._resize(frame)
        result = self._apply_blur(result)
        
        if self.enable_contrast_stretching:
            result = self._apply_contrast_stretching(result)
            
        result = self._apply_clahe(result)
        
        if self.gamma != 1.0:
            result = self._apply_gamma_correction(result)
            
        if self.enable_sharpening:
            result = self._apply_sharpening(result)
            
        return result

    def compare_before_after(
        self, original: np.ndarray
    ) -> np.ndarray:
        """
        Create a side-by-side comparison of the original and preprocessed
        frames for visual debugging.

        Both images are resized to the configured dimensions before
        concatenation so they share the same height.

        Args:
            original: The raw input frame (BGR).

        Returns:
            A horizontally concatenated image showing the original (left)
            and preprocessed (right) frames side by side.
        """
        # Resize the original to match the preprocessed dimensions
        original_resized = cv2.resize(
            original, (self.resize_width, self.resize_height)
        )

        preprocessed = self.preprocess(original)

        # Add labels to each half
        label_font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(
            original_resized, "Original", (10, 30),
            label_font, 0.8, (0, 255, 255), 2, cv2.LINE_AA,
        )
        cv2.putText(
            preprocessed, "Preprocessed", (10, 30),
            label_font, 0.8, (0, 255, 255), 2, cv2.LINE_AA,
        )

        return np.hstack([original_resized, preprocessed])

    # ============================================================
    # Private Pipeline Steps
    # ============================================================

    def _resize(self, frame: np.ndarray) -> np.ndarray:
        """
        Resize the frame to the configured width and height.

        Args:
            frame: Input BGR image.

        Returns:
            Resized BGR image.
        """
        return cv2.resize(
            frame,
            (self.resize_width, self.resize_height),
            interpolation=cv2.INTER_AREA,
        )

    def _apply_blur(self, frame: np.ndarray) -> np.ndarray:
        """
        Apply Gaussian blur for noise reduction.

        Args:
            frame: Input BGR image.

        Returns:
            Blurred BGR image.
        """
        return cv2.GaussianBlur(
            frame,
            (self.blur_kernel, self.blur_kernel),
            0,
        )

    def _apply_clahe(self, frame: np.ndarray) -> np.ndarray:
        """
        Enhance contrast using CLAHE on the L channel of LAB colour space.

        Process:
            1. Convert BGR → LAB
            2. Apply CLAHE to the L (lightness) channel
            3. Merge channels and convert LAB → BGR

        Args:
            frame: Input BGR image.

        Returns:
            Contrast-enhanced BGR image.
        """
        # Convert to LAB colour space
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)

        # Split into L, A, B channels
        l_channel, a_channel, b_channel = cv2.split(lab)

        # Apply CLAHE to the lightness channel only
        l_enhanced = self._clahe.apply(l_channel)

        # Merge the enhanced L channel back with A and B
        lab_enhanced = cv2.merge([l_enhanced, a_channel, b_channel])

        # Convert back to BGR
        return cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)

    def _apply_contrast_stretching(self, frame: np.ndarray) -> np.ndarray:
        """Apply min-max contrast stretching using OpenCV normalization."""
        return cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX)

    def _apply_gamma_correction(self, frame: np.ndarray) -> np.ndarray:
        """Apply Power-Law (Gamma) transformation using a pre-built lookup table."""
        if self._gamma_lut is not None:
            return cv2.LUT(frame, self._gamma_lut)
        return frame

    def _apply_sharpening(self, frame: np.ndarray) -> np.ndarray:
        """Apply a standard 2D sharpening kernel filter to emphasize borders."""
        return cv2.filter2D(frame, -1, self._sharpening_kernel)
