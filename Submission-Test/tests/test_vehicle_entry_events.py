
import json
import os
import tempfile
import unittest

from src import Detection, VehicleResult, FrameResult
from src.decision_engine import DecisionEngine


class TestVehicleEntryEvents(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.plates_file = os.path.join(
            self.temp_dir.name, "authorized_plates.json"
        )

        with open(self.plates_file, "w", encoding="utf-8") as f:
            json.dump(["ABC 1234"], f)

        self.config = {
            "alerts": {
                "cooldown_seconds": 0,
                "vehicle_entry_confirm_frames": 2,
                "vehicle_exit_confirm_frames": 2,
                "severity_levels": {
                    "person": "HIGH",
                    "vehicle": "MEDIUM",
                },
            },
            "authorization": {
                "authorized_plates_file": self.plates_file,
                "authorized_faces_dir": self.temp_dir.name,
            },
        }

        self.engine = DecisionEngine(self.config)

    def tearDown(self):
        self.temp_dir.cleanup()

    
    def make_vehicle(self, bbox, inside, frame_index):
        detection = Detection(
            bbox=bbox,
            class_name="vehicle",
            class_label="Car",
            confidence=0.95,
            frame_index=frame_index,
        )
        return VehicleResult(
            detection=detection,
            in_restricted_zone=inside,
        )


    
    def process(self, frame_index, bbox, inside):
        result = FrameResult(
            frame_index=frame_index,
            motion_detected=False,
        )
        result.vehicle_observations.append(
            self.make_vehicle(bbox, inside, frame_index)
        )
        return self.engine.process_frame_result(result, fps=10)


    def test_entry_generates_one_alert(self):
        self.assertEqual(
            self.process(0, (10, 10, 50, 50), False), []
        )
        self.assertEqual(
            self.process(1, (12, 10, 52, 50), True), []
        )

        alerts = self.process(2, (14, 10, 54, 50), True)
        self.assertEqual(len(alerts), 1)
        self.assertEqual(alerts[0].alert_type, "vehicle")
        self.assertTrue(
            alerts[0].details["entry_observed_from_outside"]
        )

    def test_vehicle_remaining_inside_does_not_repeat_alert(self):
        self.process(0, (10, 10, 50, 50), False)
        self.process(1, (12, 10, 52, 50), True)
        first_alerts = self.process(2, (14, 10, 54, 50), True)
        self.assertEqual(len(first_alerts), 1)

        later_alerts = self.process(3, (16, 10, 56, 50), True)
        self.assertEqual(later_alerts, [])

    def test_exit_and_reentry_generates_new_alert(self):
        self.process(0, (10, 10, 50, 50), False)
        self.process(1, (12, 10, 52, 50), True)
        first_alerts = self.process(2, (14, 10, 54, 50), True)
        self.assertEqual(len(first_alerts), 1)

        self.process(3, (16, 10, 56, 50), False)
        self.process(4, (18, 10, 58, 50), False)
        self.process(5, (20, 10, 60, 50), True)
        second_alerts = self.process(6, (22, 10, 62, 50), True)

        self.assertEqual(len(second_alerts), 1)
        
    
    def test_unreadable_plate_alert_updates_when_plate_becomes_readable(self):
        # First, the vehicle enters with an unreadable plate.
        self.process(0, (10, 10, 50, 50), False)
        self.process(1, (12, 10, 52, 50), True)

        first_alerts = self.process(2, (14, 10, 54, 50), True)

        self.assertEqual(len(first_alerts), 1)
        self.assertFalse(first_alerts[0].details["plate_detected"])
        original_timestamp = first_alerts[0].timestamp
        original_frame = first_alerts[0].frame_index

        # Later, the same vehicle's plate becomes readable.
        readable_vehicle = self.make_vehicle(
            (16, 10, 56, 50), True, 3
        )
        readable_vehicle.plate_detected = True
        readable_vehicle.plate_text = "KA25AB1234"

        result = FrameResult(
            frame_index=3,
            motion_detected=False,
        )
        result.vehicle_observations.append(readable_vehicle)

        later_alerts = self.engine.process_frame_result(result, fps=10)

        # No duplicate alert should be created.
        self.assertEqual(later_alerts, [])
        self.assertEqual(len(self.engine.alerts), 1)

        # The original alert should now contain the readable plate.
        updated_alert = self.engine.alerts[0]
        self.assertTrue(updated_alert.details["plate_detected"])
        self.assertEqual(
            updated_alert.details["plate_text"], "KA25AB1234"
        )
        self.assertIn("KA25AB1234", updated_alert.message)

        # Preserve the original entry timestamp and frame.
        self.assertEqual(updated_alert.timestamp, original_timestamp)
        self.assertEqual(updated_alert.frame_index, original_frame)



if __name__ == "__main__":
    unittest.main()
