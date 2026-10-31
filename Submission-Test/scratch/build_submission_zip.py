from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

project = Path.cwd()
archive = project.parent / "Restricted-Area-Monitoring-Submission.zip"

excluded_dirs = {
    ".venv",
    ".pytest_cache",
    "__pycache__",
    ".git",
}

excluded_files = {
    "README.backup.md",
    "EVALUATION_GUIDE.backup.md",
    "calculate_f1_metrics.backup.py",
    "fix_evaluation_guide.py",
    "check_frame30.py",
    "inspect_vehicle_crop.py",
    "test_ocr_direct.py",
    "test_ocr_variants.py",
    "test_plate_fallback.py",
    "test_plate_frame.py",
    "test_plate_normalization.py",
    "check_urls.py",
    "debug_verification.py",
    "download_and_extract.py",
    "extract_face.py",
    "trim_video.py",
}

with ZipFile(archive, "w", ZIP_DEFLATED) as zf:
    for path in project.rglob("*"):
        if not path.is_file():
            continue

        relative = path.relative_to(project)

        if any(part in excluded_dirs for part in relative.parts):
            continue
        if path.name in excluded_files:
            continue
        if path.name.endswith((".bak", ".ocr-backup")):
            continue

        zf.write(path, relative.as_posix())

print(f"Created: {archive}")
print(f"Size: {archive.stat().st_size:,} bytes")

with ZipFile(archive, "r") as zf:
    names = set(zf.namelist())
    required = [
        "app.py",
        "README.md",
        "requirements.txt",
        "src/decision_engine.py",
        "src/plate_detector.py",
        "src/roi_manager.py",
        "config/settings.yaml",
        "config/authorized_plates.json",
        "tests/test_authorization.py",
        "tests/test_vehicle_entry_events.py",
        "doc/EVALUATION_GUIDE.md",
        "output/alerts.csv",
        "output/annotated_output.mp4",
    ]

    print("\nRequired-file verification:")
    for item in required:
        print(f"[{'FOUND' if item in names else 'MISSING'}] {item}")

    unwanted = [
        name for name in names
        if any(part in excluded_dirs for part in name.split("/"))
        or name.endswith((".bak", ".ocr-backup"))
    ]
    print(f"\nExcluded directory/backup violations: {len(unwanted)}")
