"""
Plate Detector Module
=====================
Detects and reads number plates from vehicle bounding boxes using
classical image processing (edge detection + contour analysis) for
plate localisation and EasyOCR for optical character recognition.

Usage:
    plate_det = PlateDetector(config['plate_detection'])
    vehicle_result = plate_det.detect_plate(frame, vehicle_bbox)
"""

from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np
from src import Detection, VehicleResult


# ============================================================
# Plate Detector Class
# ============================================================

class PlateDetector:
    """
    Detects number plates in vehicle crops using edge detection and
    contour filtering, then reads plate text with EasyOCR.

    Pipeline:
        1. Crop vehicle region from frame.
        2. Grayscale → bilateral filter → Canny edges.
        3. Find contours → filter for rectangular, plate-like shapes.
        4. Enhance the candidate plate region (threshold + resize).
        5. Run EasyOCR and filter by confidence.
        6. Return VehicleResult with plate bbox in original frame coords.

    Attributes:
        min_plate_width: Minimum width (px) for a plate candidate.
        min_plate_height: Minimum height (px) for a plate candidate.
        ocr_languages: Language list for EasyOCR.
        ocr_confidence_threshold: Minimum OCR confidence to accept text.
        _reader: Lazy-loaded EasyOCR Reader instance.
    """

    def __init__(self, config: Dict) -> None:
        """
        Initialise plate detector parameters (EasyOCR is lazy-loaded).

        Args:
            config: Dictionary from settings.yaml under 'plate_detection'.
                    Expected keys: min_plate_width, min_plate_height,
                    ocr_languages, ocr_confidence_threshold.
        """
        self.min_plate_width: int = config.get("min_plate_width", 60)
        self.min_plate_height: int = config.get("min_plate_height", 20)
        self.ocr_languages: List[str] = config.get("ocr_languages", ["en"])
        self.ocr_confidence_threshold: float = config.get(
            "ocr_confidence_threshold", 0.3
        )

        # EasyOCR is heavy – delay import and init until first use
        self._reader = None

        print(
            f"[PlateDetector] Initialised  |  "
            f"min_plate=({self.min_plate_width}×{self.min_plate_height}), "
            f"OCR languages={self.ocr_languages}"
        )

    # ============================================================
    # Public Methods
    # ============================================================

    
    def detect_plate(
        self, frame: np.ndarray, vehicle_bbox: Tuple[int, int, int, int]
    ) -> VehicleResult:
        """Detect a plate using contours, with a lower-vehicle OCR fallback."""
        vx1, vy1, vx2, vy2 = vehicle_bbox

        detection = Detection(
            bbox=vehicle_bbox,
            class_name="vehicle",
            class_label="vehicle",
            confidence=0.0,
            frame_index=0,
        )

        frame_h, frame_w = frame.shape[:2]
        vx1 = max(0, vx1)
        vy1 = max(0, vy1)
        vx2 = min(frame_w, vx2)
        vy2 = min(frame_h, vy2)

        crop_w = vx2 - vx1
        crop_h = vy2 - vy1

        if crop_w < self.min_plate_width or crop_h < self.min_plate_height:
            return VehicleResult(detection=detection, plate_detected=False)

        vehicle_crop = frame[vy1:vy2, vx1:vx2]
        gray = cv2.cvtColor(vehicle_crop, cv2.COLOR_BGR2GRAY)
        filtered = cv2.bilateralFilter(gray, 11, 17, 17)
        edges = cv2.Canny(filtered, 30, 200)

        contours, _ = cv2.findContours(
            edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )
        plate_contour = self._find_plate_contour(contours, crop_w, crop_h)

        if plate_contour is not None:
            cx, cy, cw, ch = cv2.boundingRect(plate_contour)
            plate_crop = gray[cy:cy + ch, cx:cx + cw]
            enhanced_plate = self._enhance_plate(plate_crop)
            plate_text = self._run_ocr(enhanced_plate)

            if plate_text:
                plate_x1, plate_y1 = vx1 + cx, vy1 + cy
                plate_x2, plate_y2 = plate_x1 + cw, plate_y1 + ch
                plate_bbox = (plate_x1, plate_y1, plate_x2, plate_y2)
                plate_image = frame[plate_y1:plate_y2, plate_x1:plate_x2].copy()

                return VehicleResult(
                    detection=detection,
                    plate_detected=True,
                    plate_bbox=plate_bbox,
                    plate_text=plate_text,
                    plate_image=plate_image,
                )

        # Fallback: try the lower central portion of the vehicle crop.
        h, w = vehicle_crop.shape[:2]
        fallback_crop = vehicle_crop[
            int(h * 0.72):int(h * 0.99),
            int(w * 0.15):int(w * 0.85)
        ]

        if fallback_crop.size == 0:
            return VehicleResult(detection=detection, plate_detected=False)

        fallback_gray = cv2.cvtColor(fallback_crop, cv2.COLOR_BGR2GRAY)
        fallback_gray = cv2.resize(
            fallback_gray, None, fx=3, fy=3,
            interpolation=cv2.INTER_CUBIC
        )
        fallback_text = self._run_ocr(fallback_gray)

        if not fallback_text:
            return VehicleResult(detection=detection, plate_detected=False)

        plate_x1 = vx1 + int(crop_w * 0.15)
        plate_y1 = vy1 + int(crop_h * 0.72)
        plate_x2 = vx1 + int(crop_w * 0.85)
        plate_y2 = vy1 + int(crop_h * 0.99)

        return VehicleResult(
            detection=detection,
            plate_detected=True,
            plate_bbox=(plate_x1, plate_y1, plate_x2, plate_y2),
            plate_text=fallback_text,
            plate_image=frame[plate_y1:plate_y2, plate_x1:plate_x2].copy(),
        )


    # ============================================================
    # Private / Helper Methods
    # ============================================================

    def _find_plate_contour(
        self,
        contours: Tuple,
        crop_w: int,
        crop_h: int,
    ) -> Optional[np.ndarray]:
        """
        Search through contours for a rectangle that matches typical
        number-plate proportions.

        Criteria:
            - Approximated polygon has exactly 4 corners.
            - Width and height meet minimum thresholds.
            - Aspect ratio is between 1.5 and 6.0 (typical plate shapes).
            - Area is not too large (< 50 % of the crop).

        Args:
            contours: Contours found by cv2.findContours.
            crop_w: Width of the vehicle crop (for relative size checks).
            crop_h: Height of the vehicle crop.

        Returns:
            The best matching contour, or None if no plate-like rectangle
            was found.
        """
        # Sort contours by area, largest first
        sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)

        # Only inspect the top 30 contours to keep processing fast
        for contour in sorted_contours[:30]:
            # Approximate the contour to a polygon
            perimeter: float = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.018 * perimeter, True)

            # We want exactly 4 corners (rectangle)
            if len(approx) != 4:
                continue

            # Get bounding rectangle
            _, _, w, h = cv2.boundingRect(approx)

            # Check minimum size
            if w < self.min_plate_width or h < self.min_plate_height:
                continue

            # Aspect ratio check (plates are wider than tall)
            if h == 0:
                continue
            aspect_ratio: float = w / h
            if not (1.5 <= aspect_ratio <= 6.0):
                continue

            # Reject if the contour covers more than 50 % of the crop
            contour_area: float = cv2.contourArea(approx)
            crop_area: int = crop_w * crop_h
            if contour_area > 0.5 * crop_area:
                continue

            return approx

        return None

    def _enhance_plate(self, plate_gray: np.ndarray) -> np.ndarray:
        """
        Enhance a grayscale plate crop for better OCR accuracy.

        Steps:
            1. Resize to a standard height of 100 px (maintain aspect ratio).
            2. Apply adaptive thresholding to binarise the image.

        Args:
            plate_gray: Grayscale plate image.

        Returns:
            Enhanced (binary) plate image.
        """
        h, w = plate_gray.shape[:2]

        # Resize to fixed height for consistent OCR input
        target_height: int = 100
        if h > 0:
            scale: float = target_height / h
            new_w: int = max(1, int(w * scale))
            plate_resized: np.ndarray = cv2.resize(
                plate_gray, (new_w, target_height), interpolation=cv2.INTER_CUBIC
            )
        else:
            plate_resized = plate_gray

        # Adaptive threshold for binarisation
        plate_binary: np.ndarray = cv2.adaptiveThreshold(
            plate_resized,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11,  # Block size
            2,   # C constant
        )

        return plate_binary

    def _run_ocr(self, plate_image: np.ndarray) -> Optional[str]:
        """
        Run EasyOCR on an enhanced plate image and return the
        concatenated text if it passes the confidence threshold.

        The EasyOCR Reader is lazy-loaded on first call to avoid
        heavy initialisation when the module is imported.

        Args:
            plate_image: Enhanced (binary) plate image.

        Returns:
            Recognised plate text string, or None if OCR failed or
            confidence was below the threshold.
        """
        # --- Lazy-load EasyOCR reader ---
        if self._reader is None:
            try:
                import easyocr
                self._reader = easyocr.Reader(
                    self.ocr_languages,
                    gpu=False,
                    verbose=False,
                )
                print("[PlateDetector] EasyOCR reader loaded")
            except ImportError:
                print("[PlateDetector] ERROR: easyocr is not installed")
                return None

        # --- Run OCR ---
        try:
            results = self._reader.readtext(plate_image)
        except Exception as e:
            print(f"[PlateDetector] OCR error: {e}")
            return None

        if not results:
            return None

        # --- Filter by confidence and concatenate text ---
        accepted_parts: List[str] = []
        for bbox, text, confidence in results:
            if confidence >= self.ocr_confidence_threshold:
                accepted_parts.append(text.strip())

        if not accepted_parts:
            return None

        
        # Join all recognised text fragments.
        plate_text: str = " ".join(accepted_parts).upper()

        
        # Normalize OCR output for Indian vehicle registration plates.
        import re

        cleaned = re.sub(r"[^A-Z0-9]", "", plate_text.upper())

        def normalize_candidate(candidate):
            """Normalize a candidate matching the AA00AA0000 format."""
            if len(candidate) != 10:
                return None

            state = candidate[:2]
            district = candidate[2:4].translate(str.maketrans({
                "O": "0", "Q": "0", "D": "0",
                "I": "1", "L": "1", "Z": "2",
                "S": "5", "B": "8", "G": "6",
            }))
            series = candidate[4:6].translate(str.maketrans({
                "0": "O", "1": "I", "2": "Z",
                "5": "S", "6": "G", "8": "B",
            }))
            number = candidate[6:10].translate(str.maketrans({
                "O": "0", "Q": "0", "D": "0",
                "I": "1", "L": "1", "Z": "2",
                "S": "5", "B": "8", "G": "6",
            }))

            if (
                state.isalpha()
                and district.isdigit()
                and series.isalpha()
                and number.isdigit()
            ):
                return state + district + series + number

            return None

        # Try the complete OCR text first.
        normalized = normalize_candidate(cleaned)

        # If an extra leading character exists, try removing it.
        # Accept this only when the remaining text matches the format.
        if normalized is None and len(cleaned) == 11:
            normalized = normalize_candidate(cleaned[1:])

        if normalized is not None:
            plate_text = normalized

        return plate_text


