"""
Motion Detector
===============
Provides the ``MotionDetector`` class which detects motion in video frames
using OpenCV's background subtraction algorithms (MOG2 or KNN).

The detector supports an optional Region-of-Interest (ROI) mask so that
motion outside the restricted area is ignored.

Usage::

    from src.motion_detector import MotionDetector
    from src.utils import load_config

    config = load_config("config/settings.yaml")
    detector = MotionDetector(config["motion_detection"])

    motion_found, mask, contours = detector.detect_motion(frame)
"""

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


class MotionDetector:
    """
    Detects motion in video frames using background subtraction.

    Supports both **MOG2** and **KNN** algorithms (selectable via config).
    Morphological operations are applied to the foreground mask to remove
    noise, and contours below a configurable area threshold are discarded.

    Attributes:
        algorithm: Name of the active background subtraction algorithm.
        min_contour_area: Minimum contour area (px²) to count as motion.
        learning_rate: Background model learning rate (-1 = automatic).
    """

    def __init__(self, config: Dict) -> None:
        """
        Initialise the motion detector with parameters from the config
        dictionary.

        Args:
            config: The ``motion_detection`` section of settings.yaml.
                    Expected keys: algorithm, history, var_threshold,
                    detect_shadows, min_contour_area, learning_rate.
        """
        # --- Configuration ---
        self.algorithm: str = config.get("algorithm", "MOG2").upper()
        history: int = config.get("history", 500)
        var_threshold: int = config.get("var_threshold", 40)
        detect_shadows: bool = config.get("detect_shadows", True)
        self.min_contour_area: int = config.get("min_contour_area", 500)
        self.learning_rate: float = config.get("learning_rate", -1)

        # --- Create the background subtractor ---
        self._bg_subtractor = self._create_subtractor(
            self.algorithm, history, var_threshold, detect_shadows
        )

        # --- Morphological structuring element (5×5 ellipse) ---
        self._morph_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (5, 5)
        )

    # ============================================================
    # Public Interface
    # ============================================================

    def detect_motion(
        self,
        frame: np.ndarray,
        roi_mask: Optional[np.ndarray] = None,
    ) -> Tuple[bool, np.ndarray, List[np.ndarray]]:
        """
        Detect motion in a single frame.

        Steps:
            1. Apply background subtractor to obtain the foreground mask.
            2. If *roi_mask* is provided, AND the masks to limit detection
               to the region of interest.
            3. Apply morphological operations (erode → dilate) to reduce
               noise and fill small gaps.
            4. Find contours in the cleaned mask.
            5. Filter contours by ``min_contour_area``.

        Args:
            frame: Input BGR frame as a numpy array.
            roi_mask: Optional single-channel binary mask (255 = inside ROI,
                      0 = outside).  Must match *frame* height and width.

        Returns:
            A tuple of:
                - motion_detected (bool): True if any qualifying contour
                  was found.
                - motion_mask (np.ndarray): The cleaned foreground mask
                  (single-channel, uint8).
                - contours (list[np.ndarray]): List of contours that
                  exceeded the minimum area threshold.
        """
        # Step 1: Background subtraction
        fg_mask = self._bg_subtractor.apply(
            frame, learningRate=self.learning_rate
        )

        # Step 2: Restrict to ROI if mask is provided
        if roi_mask is not None:
            fg_mask = cv2.bitwise_and(fg_mask, roi_mask)

        # Step 3: Morphological cleanup — erode to remove thin noise,
        # then dilate to restore object shapes
        cleaned_mask = self._apply_morphology(fg_mask)

        # Step 4: Find contours
        contours_all, _ = cv2.findContours(
            cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # Step 5: Filter by minimum contour area
        valid_contours = [
            c for c in contours_all
            if cv2.contourArea(c) >= self.min_contour_area
        ]

        motion_detected = len(valid_contours) > 0

        return motion_detected, cleaned_mask, valid_contours

    def reset(self) -> None:
        """
        Reset the background model.

        This is useful when the camera view changes or after a long pause
        in processing, so the background model can be re-learnt from
        scratch using the current algorithm settings.
        """
        # Re-read stored parameters and create a fresh subtractor
        if self.algorithm == "KNN":
            self._bg_subtractor = cv2.createBackgroundSubtractorKNN()
        else:
            self._bg_subtractor = cv2.createBackgroundSubtractorMOG2()

    # ============================================================
    # Private Helpers
    # ============================================================

    @staticmethod
    def _create_subtractor(
        algorithm: str,
        history: int,
        var_threshold: int,
        detect_shadows: bool,
    ) -> cv2.BackgroundSubtractor:
        """
        Factory method to instantiate the appropriate OpenCV background
        subtractor.

        Args:
            algorithm: ``"MOG2"`` or ``"KNN"``.
            history: Number of frames used to build the background model.
            var_threshold: Variance threshold (used by MOG2).
            detect_shadows: Whether shadow detection is enabled.

        Returns:
            An OpenCV ``BackgroundSubtractor`` instance.

        Raises:
            ValueError: If *algorithm* is not ``"MOG2"`` or ``"KNN"``.
        """
        if algorithm == "KNN":
            return cv2.createBackgroundSubtractorKNN(
                history=history,
                detectShadows=detect_shadows,
            )
        elif algorithm == "MOG2":
            return cv2.createBackgroundSubtractorMOG2(
                history=history,
                varThreshold=var_threshold,
                detectShadows=detect_shadows,
            )
        else:
            raise ValueError(
                f"Unsupported background subtraction algorithm: '{algorithm}'. "
                "Choose 'MOG2' or 'KNN'."
            )

    def _apply_morphology(self, mask: np.ndarray) -> np.ndarray:
        """
        Clean the foreground mask using morphological operations.

        The sequence — *erode* then *dilate* (i.e. morphological opening)
        — removes small noise blobs while preserving larger motion regions.
        A second *dilate* pass helps fill small gaps inside detected areas.

        Args:
            mask: Raw foreground mask (uint8, single channel).

        Returns:
            Cleaned binary mask.
        """
        # Threshold the mask to pure binary (shadows appear as 127 in MOG2)
        _, binary = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)

        # Erode to remove thin noise
        eroded = cv2.erode(binary, self._morph_kernel, iterations=1)

        # Dilate to restore object boundaries and fill small holes
        dilated = cv2.dilate(eroded, self._morph_kernel, iterations=2)

        return dilated
