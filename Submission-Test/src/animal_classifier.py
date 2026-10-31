"""
Animal Classifier Module
========================
Classifies animals detected by YOLO into domestic or wild categories
and assigns a risk level based on configuration lookups.

Risk-level logic:
    - domestic  → 'low'
    - wild      → 'medium'
    - high-risk → 'high'  (overrides 'medium' for specific wild animals)

Usage:
    classifier = AnimalClassifier(config['animal_classification'])
    animal_result = classifier.classify(detection)
"""

from typing import Dict, List, Set
from src import Detection, AnimalResult


# ============================================================
# Animal Classifier Class
# ============================================================

class AnimalClassifier:
    """
    Classifies an animal Detection into a category (domestic / wild)
    and risk level (low / medium / high) using simple lookup tables
    loaded from the project configuration.

    Attributes:
        _domestic_animals: Set of COCO label strings for domestic animals.
        _wild_animals: Set of COCO label strings for wild animals.
        _high_risk_animals: Set of COCO label strings for high-risk animals.
    """

    def __init__(self, config: Dict) -> None:
        """
        Initialise the classifier from the animal_classification config.

        Args:
            config: Dictionary from settings.yaml under
                    'animal_classification'.  Expected keys:
                    domestic_animals, wild_animals, high_risk_animals.
        """
        # --- Build lookup sets for O(1) membership tests ---
        self._domestic_animals: Set[str] = set(
            config.get("domestic_animals", ["cat", "dog"])
        )
        self._wild_animals: Set[str] = set(
            config.get("wild_animals", [])
        )
        self._high_risk_animals: Set[str] = set(
            config.get("high_risk_animals", ["bear", "elephant"])
        )

        print(
            f"[AnimalClassifier] Initialised  |  "
            f"domestic={sorted(self._domestic_animals)}, "
            f"wild={sorted(self._wild_animals)}, "
            f"high_risk={sorted(self._high_risk_animals)}"
        )

    # ============================================================
    # Public Methods
    # ============================================================

    def classify(self, detection: Detection) -> AnimalResult:
        """
        Classify a detected animal by its COCO class label.

        Decision logic:
            1. Read the specific COCO label (e.g. 'dog', 'bear') from
               detection.class_label.
            2. Check if it appears in the domestic or wild sets.
            3. Assign risk level:
                - domestic animals       → 'low'
                - wild animals           → 'medium'
                - high-risk wild animals → 'high' (overrides 'medium')
                - unknown animals        → 'medium' (default caution)

        Args:
            detection: A Detection object with class_name == 'animal'.

        Returns:
            AnimalResult with animal_type, category, and risk_level.
        """
        animal_type: str = detection.class_label  # e.g. 'dog', 'bear'

        # --- Determine category ---
        if animal_type in self._domestic_animals:
            category: str = "domestic"
            risk_level: str = "low"

        elif animal_type in self._wild_animals:
            category = "wild"
            # High-risk animals override the default 'medium'
            if animal_type in self._high_risk_animals:
                risk_level = "high"
            else:
                risk_level = "medium"

        else:
            # Unknown animal – treat with caution
            category = "unknown"
            risk_level = "medium"

        return AnimalResult(
            detection=detection,
            animal_type=animal_type,
            category=category,
            risk_level=risk_level,
        )
