"""
F1-Score, Precision, Recall, and Accuracy Calculator
======================================================
This script demonstrates how to evaluate the system's performance on your CCTV video.
It defines a ground truth list (when the intruder actually entered) and compares it
against the system's logs to compute standard classification metrics:
    - True Positives (TP)
    - False Positives (FP)
    - False Negatives (FN)
    - Precision
    - Recall
    - F1-Score
    - Accuracy
"""

import os
import pandas as pd
import numpy as np

def calculate_metrics():
    print("=" * 60)
    print("  Restricted Area Monitoring - Performance Metrics Calculator")
    print("=" * 60)

    # 1. Define Ground Truth Annotations for the Sample Video
    # Format: {frame_index: "expected_class"}
    # Let's say in your 647-frame sample video:
    #   - An unauthorized person is visible in the ROI from frame 20 to frame 35
    #   - A vehicle is visible in the ROI from frame 120 to frame 150
    #   - All other frames are clear (no threat)
    ground_truth = {}
    for f in range(20, 36):
        ground_truth[f] = "person"
    for f in range(120, 151):
        ground_truth[f] = "vehicle"

    # 2. Load the System's Detections / Alerts
    # We will try to read from the actual detection log CSV, or fall back to mock detections if empty
    log_path = os.path.join("output", "detection_log.csv")
    predictions = {}

    if os.path.exists(log_path):
        try:
            df = pd.read_csv(log_path)
            # Group by frame index and get the class name
            for _, row in df.iterrows():
                f_idx = int(row.get("FrameIndex", 0))
                cls_name = str(row.get("Class", "")).lower()
                # If the alert was not cooldown suppressed, log it as a prediction
                if not row.get("CooldownSuppressed", False):
                    predictions[f_idx] = cls_name
            print(f"[INFO] Loaded {len(df)} entries from log: {log_path}")
        except Exception as e:
            print(f"[WARNING] Error reading {log_path}: {e}. Initializing default predictions.")
    
    # If no predictions loaded (or file doesn't exist), load mock predictions for demonstration
    if not predictions:
        print("[INFO] Log file empty or missing. Loading representative predictions for demonstration:")
        # Mocking system behavior:
        #   - Person detected at frame 22 (causes alert, starts cooldown of 5s = 60 frames @ 12fps)
        #   - Vehicle detected at frame 120 (causes alert, starts cooldown of 5s = 60 frames)
        #   - One false alarm (noise) at frame 80 (detected as animal)
        predictions = {
            22: "person",
            80: "animal",     # False positive (e.g., leaves detected as animal)
            120: "vehicle"
        }

    # 3. Calculate Confusion Matrix Elements
    # We analyze frame by frame. Since the alert engine uses a COOLDOWN (e.g. 5 seconds / 60 frames),
    # we evaluate on a "window-based" intrusion detection metric (standard in security systems)
    # or frame-based metrics.
    # Let's do an Intrusion Event-based evaluation which matches CCTV security criteria.
    print("\n[EVALUATION] Checking Intrusion Events:")
    
    # Defined Event 1: Human Intrusion (frames 20 - 35)
    human_event_gt = True
    human_event_pred = any(predictions.get(f) == "person" for f in range(20, 36))
    
    # Defined Event 2: Vehicle Intrusion (frames 120 - 150)
    vehicle_event_gt = True
    vehicle_event_pred = any(predictions.get(f) == "vehicle" for f in range(120, 151))
    
    # Normal Non-intrusion windows (we check other intervals, say 5 clear intervals of 30 frames each)
    clear_windows = [
        (0, 19),
        (36, 119),
        (151, 200)
    ]
    
    # Calculate TP, FP, FN, TN
    tp = 0
    fp = 0
    fn = 0
    tn = 0
    
    # Check Human Event
    if human_event_gt:
        if human_event_pred:
            tp += 1
            print("  [OK] [TP] Human Intrusion (Frames 20-35) successfully detected!")
        else:
            fn += 1
            print("  [FAIL] [FN] Human Intrusion (Frames 20-35) missed by the system!")
            
    # Check Vehicle Event
    if vehicle_event_gt:
        if vehicle_event_pred:
            tp += 1
            print("  [OK] [TP] Vehicle Intrusion (Frames 120-150) successfully detected!")
        else:
            fn += 1
            print("  [FAIL] [FN] Vehicle Intrusion (Frames 120-150) missed by the system!")

    # Check False Positives in clear windows
    for start, end in clear_windows:
        window_has_fp = any(f in predictions for f in range(start, end + 1))
        if window_has_fp:
            fp += 1
            trigger_frames = [f for f in range(start, end + 1) if f in predictions]
            print(f"  [FAIL] [FP] False alarm triggered in clear window (Frames {start}-{end}) at Frame(s) {trigger_frames}!")
        else:
            tn += 1
            print(f"  [OK] [TN] Clear window (Frames {start}-{end}) remained clear.")

    # 4. Calculate Mathematical Ratios
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0

    # 5. Output Report
    print("\n" + "=" * 60)
    print("  CLASSIFICATION METRICS REPORT")
    print("=" * 60)
    print(f"  True Positives (TP)  : {tp}  (Correctly identified intrusions)")
    print(f"  False Positives (FP) : {fp}  (False alerts during clean periods)")
    print(f"  False Negatives (FN) : {fn}  (Missed actual intrusions)")
    print(f"  True Negatives (TN)  : {tn}  (Correctly ignored clean periods)")
    print("-" * 60)
    print(f"  Precision            : {precision:.2f}  (TP / [TP + FP])")
    print(f"  Recall               : {recall:.2f}  (TP / [TP + FN])")
    print(f"  F1-Score             : {f1_score:.2f}  (Harmonic mean of Precision & Recall)")
    print(f"  Accuracy             : {accuracy:.2%}")
    print("=" * 60)
    
    # 6. Explanation of how to improve scores:
    print("\n[TIPS] Examiner Tips for Improving Metrics:")
    print("  1. If Precision is low (High FP):")
    print("     - Increase 'min_contour_area' in config/settings.yaml to ignore small noise.")
    print("     - Increase YOLOv8 'confidence_threshold' to prevent false object identifications.")
    print("  2. If Recall is low (High FN):")
    print("     - Decrease YOLOv8 'confidence_threshold' to capture faint/far objects.")
    print("     - Lower 'var_threshold' in background subtraction to detect subtle motion.")
    print("  3. Pre-trained vs Custom:")
    print("     - Pre-trained models (YOLOv8 COCO, DeepFace Facenet) are benchmarked on millions of general images.")
    print("     - Their F1-scores on general benchmarks are: YOLOv8n ~0.82 F1, Facenet LFW ~0.99 F1.")
    print("=" * 60)

if __name__ == "__main__":
    calculate_metrics()
