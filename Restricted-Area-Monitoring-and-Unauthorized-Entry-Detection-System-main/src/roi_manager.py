"""
ROI Manager
===========
Provides the ``ROIManager`` class for defining, persisting, and
visualising a polygon-based Region of Interest (ROI) on video frames.

The ROI is used by other modules (e.g. ``MotionDetector``) to restrict
processing to the restricted area of the scene.

Usage::

    from src.roi_manager import ROIManager

    roi = ROIManager()
    roi.set_polygon([(100, 100), (400, 100), (400, 400), (100, 400)])

    mask = roi.generate_mask((480, 640))        # frame_shape = (H, W)
    inside = roi.is_inside_roi((120, 150, 200, 300))  # bbox
    annotated = roi.draw_roi(frame)

    roi.save_roi("config/roi.yaml")
    roi.load_roi("config/roi.yaml")
"""

from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import yaml


class ROIManager:
    """
    Manage a polygon-shaped Region of Interest for restricted-area
    monitoring.

    Attributes:
        polygon: List of (x, y) vertex coordinates defining the ROI, or
                 None if no ROI has been set.
    """

    def __init__(self) -> None:
        """Initialise the ROI manager with no polygon set."""
        self.polygon: Optional[List[Tuple[int, int]]] = None
        # Cached numpy array version of the polygon (N×1×2, int32)
        self._polygon_np: Optional[np.ndarray] = None

    # ============================================================
    # Polygon Setup
    # ============================================================

    def set_polygon(self, points: List[Tuple[int, int]]) -> None:
        """
        Define the ROI as a list of (x, y) polygon vertices.

        The polygon must have at least 3 points to form a valid area.

        Args:
            points: Ordered list of (x, y) tuples defining the polygon
                    vertices.

        Raises:
            ValueError: If fewer than 3 points are provided.
        """
        if len(points) < 3:
            raise ValueError(
                f"A polygon requires at least 3 points, got {len(points)}."
            )

        self.polygon = list(points)

        # Convert to the numpy shape OpenCV expects: (N, 1, 2), int32
        self._polygon_np = np.array(
            self.polygon, dtype=np.int32
        ).reshape((-1, 1, 2))

    # ============================================================
    # Mask Generation
    # ============================================================

    def generate_mask(
        self, frame_shape: Tuple[int, int]
    ) -> np.ndarray:
        """
        Generate a binary mask from the ROI polygon.

        Pixels inside the polygon are set to 255 (white); pixels outside
        are 0 (black).

        Args:
            frame_shape: (height, width) of the target frame.

        Returns:
            Single-channel uint8 mask of the same spatial size as the frame.

        Raises:
            RuntimeError: If no polygon has been set.
        """
        if self._polygon_np is None:
            raise RuntimeError(
                "No ROI polygon has been set. Call set_polygon() first."
            )

        height, width = frame_shape[:2]
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(mask, [self._polygon_np], 255)

        return mask

    # ============================================================
    # Point-in-Polygon Test
    # ============================================================

    def is_inside_roi(self, bbox: Tuple[int, int, int, int]) -> bool:
        """
        Check whether the centre of a bounding box falls inside the ROI
        polygon.

        Uses ``cv2.pointPolygonTest`` which returns a positive value if
        the point is inside, zero if on the edge, and negative if outside.

        Args:
            bbox: Bounding box as (x1, y1, x2, y2).

        Returns:
            True if the bbox centre is inside or on the ROI polygon,
            False otherwise.  Also returns False if no polygon is set.
        """
        if self._polygon_np is None:
            return False

        # Compute the bounding-box centre for general objects.
        x1, y1, x2, y2 = bbox
        centre_x = (x1 + x2) / 2.0
        centre_y = (y1 + y2) / 2.0
        return self.is_point_inside_roi((centre_x, centre_y))

    def is_point_inside_roi(self, point: Tuple[float, float]) -> bool:
        """Return True when a point is inside or on the ROI polygon.

        Vehicle checks should use the bottom-centre (ground-contact) point,
        which is generally more useful than the centre of the full vehicle box.
        """
        if self._polygon_np is None:
            return False
        distance = cv2.pointPolygonTest(
            self._polygon_np,
            (float(point[0]), float(point[1])),
            measureDist=False,
        )
        return distance >= 0

    # ============================================================
    # Visualisation
    # ============================================================

    def draw_roi(
        self,
        frame: np.ndarray,
        color: Tuple[int, int, int] = (0, 255, 0),
        alpha: float = 0.3,
    ) -> np.ndarray:
        """
        Draw the ROI polygon overlay on a frame with transparency.

        A semi-transparent filled polygon is blended onto the frame, and
        the polygon outline is drawn on top for clarity.

        Args:
            frame: Input BGR image (will not be modified in place).
            color: BGR colour for the ROI overlay.
            alpha: Opacity of the filled polygon (0.0 = invisible,
                   1.0 = fully opaque).

        Returns:
            A copy of the frame with the ROI overlay applied.
            Returns an unmodified copy if no polygon is set.
        """
        output = frame.copy()

        if self._polygon_np is None:
            return output

        # Create the semi-transparent filled overlay
        overlay = output.copy()
        cv2.fillPoly(overlay, [self._polygon_np], color)
        cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)

        # Draw the polygon outline
        cv2.polylines(
            output,
            [self._polygon_np],
            isClosed=True,
            color=color,
            thickness=2,
        )

        return output

    # ============================================================
    # Persistence (YAML)
    # ============================================================

    def save_roi(self, filepath: str) -> None:
        """
        Save the current ROI polygon points to a YAML file.

        Args:
            filepath: Destination file path (e.g. ``config/roi.yaml``).

        Raises:
            RuntimeError: If no polygon has been set.
        """
        if self.polygon is None:
            raise RuntimeError(
                "No ROI polygon to save. Call set_polygon() first."
            )

        # Convert tuples to plain lists for cleaner YAML output
        data = {
            "roi_polygon": [list(pt) for pt in self.polygon]
        }

        output_path = Path(filepath)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, default_flow_style=False)

    def load_roi(self, filepath: str) -> None:
        """
        Load ROI polygon points from a YAML file.

        The file must contain a top-level key ``roi_polygon`` whose value
        is a list of [x, y] coordinate pairs.

        Args:
            filepath: Path to the YAML file containing the ROI definition.

        Raises:
            FileNotFoundError: If the file does not exist.
            KeyError: If the file does not contain ``roi_polygon``.
            ValueError: If the polygon data is invalid.
        """
        roi_path = Path(filepath)

        if not roi_path.exists():
            raise FileNotFoundError(f"ROI file not found: {filepath}")

        with open(roi_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        if "roi_polygon" not in data:
            raise KeyError(
                f"YAML file '{filepath}' does not contain 'roi_polygon' key."
            )

        raw_points = data["roi_polygon"]

        if not isinstance(raw_points, list) or len(raw_points) < 3:
            raise ValueError(
                "roi_polygon must be a list of at least 3 [x, y] points."
            )

        # Convert each [x, y] list to a (x, y) tuple
        points = [(int(pt[0]), int(pt[1])) for pt in raw_points]
        self.set_polygon(points)
