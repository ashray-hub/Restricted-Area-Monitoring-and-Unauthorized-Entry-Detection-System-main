import csv
import os
from collections import defaultdict

GROUND_TRUTH_PATH = os.path.join("scratch", "ground_truth_events.csv")
ALERTS_PATH = os.path.join("output", "alerts.csv")


def load_ground_truth():
    events = []

    with open(GROUND_TRUTH_PATH, "r", newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        required = {"start_frame", "end_frame", "type"}

        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(
                "Ground-truth CSV must contain: start_frame,end_frame,type"
            )

        for line_number, row in enumerate(reader, start=2):
            if not row.get("start_frame", "").strip():
                continue

            start = int(row["start_frame"])
            end = int(row["end_frame"])
            event_type = row["type"].strip().lower()

            if start < 0 or end < start or not event_type:
                raise ValueError(
                    f"Invalid ground-truth event on CSV line {line_number}"
                )

            events.append({
                "start": start,
                "end": end,
                "type": event_type
            })

    return events


def load_alerts():
    if not os.path.exists(ALERTS_PATH):
        raise FileNotFoundError(
            f"{ALERTS_PATH} not found. Process a video and export its alerts first."
        )

    alerts = []

    with open(ALERTS_PATH, "r", newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        required = {"Frame", "Type"}

        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError("Alerts CSV must contain Frame and Type columns.")

        for line_number, row in enumerate(reader, start=2):
            if not row.get("Frame", "").strip():
                continue

            frame = int(row["Frame"])
            event_type = row["Type"].strip().lower()

            if frame < 0 or not event_type:
                raise ValueError(f"Invalid alert on CSV line {line_number}")

            alerts.append({"frame": frame, "type": event_type})

    return alerts


def calculate_metrics():
    ground_truth = load_ground_truth()
    alerts = load_alerts()

    if not ground_truth:
        print("No ground-truth events are annotated yet.")
        print(f"Add real event ranges to: {GROUND_TRUTH_PATH}")
        print("Example format: 20,35,person")
        print("Use only frame ranges verified by watching your actual video.")
        return

    matched_alert_indexes = set()
    true_positives = 0
    false_negatives = 0

    for event in ground_truth:
        match_index = next(
            (
                index for index, alert in enumerate(alerts)
                if index not in matched_alert_indexes
                and alert["type"] == event["type"]
                and event["start"] <= alert["frame"] <= event["end"]
            ),
            None
        )

        if match_index is not None:
            matched_alert_indexes.add(match_index)
            true_positives += 1
        else:
            false_negatives += 1

    false_positives = len(alerts) - len(matched_alert_indexes)

    precision = (
        true_positives / (true_positives + false_positives)
        if true_positives + false_positives else 0.0
    )
    recall = (
        true_positives / (true_positives + false_negatives)
        if true_positives + false_negatives else 0.0
    )
    f1_score = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0.0
    )

    print("\n===== Event-Based Evaluation =====")
    print(f"Ground-truth events : {len(ground_truth)}")
    print(f"System alerts       : {len(alerts)}")
    print(f"True positives (TP) : {true_positives}")
    print(f"False positives (FP): {false_positives}")
    print(f"False negatives (FN): {false_negatives}")
    print(f"Precision           : {precision:.4f} ({precision * 100:.2f}%)")
    print(f"Recall              : {recall:.4f} ({recall * 100:.2f}%)")
    print(f"F1-score            : {f1_score:.4f} ({f1_score * 100:.2f}%)")
    print("\nNote: These are event-level metrics, not frame-level metrics.")
    print("Accuracy and true negatives are not calculated because this")
    print("event-only evaluation does not define verified negative frames.")


if __name__ == "__main__":
    try:
        calculate_metrics()
    except (OSError, ValueError, KeyError) as error:
        print(f"[ERROR] {error}")
