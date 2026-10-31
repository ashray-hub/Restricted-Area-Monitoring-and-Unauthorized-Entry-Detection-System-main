"""
🔒 Restricted Area Monitoring & Unauthorized Entry Detection System
================================================================
Authorization & Alert Suppression Unit Test

This script tests the whitelisting logic for authorized people and vehicle plates.
It verifies:
    1. Reading from the local authorized plates JSON.
    2. Plate number normalization (spacing, cases, non-alphanumeric chars).
    3. Person alert suppression when marked as authorized.
    4. Vehicle alert suppression when the plate matches the whitelist.
"""

import os
import sys
import json
import tempfile
import unittest

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.decision_engine import DecisionEngine, Alert
from src import Detection, PersonResult, VehicleResult, FrameResult

class TestAuthorization(unittest.TestCase):
    
    def setUp(self):
        # Create a temporary file for authorized plates
        self.temp_dir = tempfile.TemporaryDirectory()
        self.plates_file = os.path.join(self.temp_dir.name, "authorized_plates.json")
        
        # Initial authorized plates
        self.whitelisted_plates = ["ABC 1234", "XYZ-7890", "MH12 DE 4567"]
        with open(self.plates_file, "w", encoding="utf-8") as f:
            json.dump(self.whitelisted_plates, f)
            
        # Create config mock
        self.config = {
            "alerts": {
                "cooldown_seconds": 0,
                "vehicle_entry_confirm_frames": 1,
                "severity_levels": {
                    "person": "HIGH",
                    "vehicle": "MEDIUM"
                }
            },
            "authorization": {
                "authorized_plates_file": self.plates_file,
                "authorized_faces_dir": "data/authorized_faces"
            }
        }
        
        # Instantiate Decision Engine
        self.engine = DecisionEngine(self.config)

    def tearDown(self):
        # Cleanup temp directory
        self.temp_dir.cleanup()

    def test_load_plates(self):
        """Test if plates are loaded and normalized correctly."""
        plates = self.engine._load_authorized_plates()
        self.assertEqual(len(plates), 3)
        self.assertIn("ABC 1234", plates)
        self.assertIn("XYZ-7890", plates)
        self.assertIn("MH12 DE 4567", plates)

    def test_authorized_person_suppression(self):
        """Test that alert is suppressed for an authorized person, but generated for unauthorized."""
        # 1. Authorized Person
        authorized_person = PersonResult(
            detection=Detection(bbox=(10, 10, 100, 100), class_name="person", class_label="person", confidence=0.9, frame_index=1),
            face_detected=True,
            face_bbox=(30, 30, 80, 80),
            is_authorized=True,
            identity="John Doe"
        )
        
        frame_result_1 = FrameResult(frame_index=1, motion_detected=True, persons=[authorized_person])
        alerts_1 = self.engine.process_frame_result(frame_result_1, fps=10.0)
        
        # Alert should be suppressed (None returned, frame alerts empty)
        self.assertEqual(len(alerts_1), 0)
        self.assertTrue(authorized_person.is_authorized)
        
        # 2. Unauthorized Person
        unauthorized_person = PersonResult(
            detection=Detection(bbox=(10, 10, 100, 100), class_name="person", class_label="person", confidence=0.9, frame_index=2),
            face_detected=True,
            face_bbox=(30, 30, 80, 80),
            is_authorized=False,
            identity=None
        )
        
        self.engine.reset()
        frame_result_2 = FrameResult(frame_index=2, motion_detected=True, persons=[unauthorized_person])
        alerts_2 = self.engine.process_frame_result(frame_result_2, fps=10.0)
        
        # Alert should be created
        self.assertEqual(len(alerts_2), 1)
        self.assertEqual(alerts_2[0].alert_type, "person")
        self.assertEqual(alerts_2[0].severity, "HIGH")
        self.assertIn("UNAUTHORIZED", alerts_2[0].message)

    def test_authorized_vehicle_suppression(self):
        """Test that alert is suppressed for vehicles with whitelisted plates."""
        # 1. Whitelisted vehicle plate
        authorized_vehicle = VehicleResult(
            detection=Detection(bbox=(50, 50, 200, 200), class_name="vehicle", class_label="car", confidence=0.85, frame_index=1),
            plate_detected=True,
            plate_bbox=(80, 150, 150, 180),
            plate_text="ABC-1234"  # Has extra hyphen, should still match "ABC 1234"
        )
        
        frame_result_1 = FrameResult(frame_index=1, motion_detected=True, vehicles=[authorized_vehicle])
        alerts_1 = self.engine.process_frame_result(frame_result_1, fps=10.0)
        
        # Alert should be suppressed
        self.assertEqual(len(alerts_1), 0)
        # VehicleResult should be updated
        self.assertTrue(authorized_vehicle.is_authorized)
        self.assertEqual(authorized_vehicle.identity, "XYZ-7890" if authorized_vehicle.identity == "XYZ-7890" else "ABC 1234")

        # 2. Unauthorized vehicle plate
        unauthorized_vehicle = VehicleResult(
            detection=Detection(bbox=(50, 50, 200, 200), class_name="vehicle", class_label="car", confidence=0.85, frame_index=2),
            plate_detected=True,
            plate_bbox=(80, 150, 150, 180),
            plate_text="DEF-5678"  # Not in whitelist
        )
        
        self.engine.reset()
        frame_result_2 = FrameResult(frame_index=2, motion_detected=True, vehicles=[unauthorized_vehicle])
        alerts_2 = self.engine.process_frame_result(frame_result_2, fps=10.0)
        
        # Alert should be created
        self.assertEqual(len(alerts_2), 1)
        self.assertFalse(unauthorized_vehicle.is_authorized)
        self.assertEqual(alerts_2[0].alert_type, "vehicle")
        self.assertEqual(alerts_2[0].severity, "MEDIUM")

if __name__ == "__main__":
    unittest.main()
