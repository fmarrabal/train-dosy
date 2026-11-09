"""Prepare a fresh neural validation run from the distributed source and traces."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import shutil

BASE = Path(__file__).resolve().parent
HELPERS = ("neural_study.py", "analyse_study.py", "check_reproducibility.py",
           "extra_neural_diagnostics.py", "audit_complete_inputs.py", "render_neural_report.py")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_snapshot():
    snapshot = BASE / "source_snapshot"
    manifest = json.loads((snapshot / "manifest.json").read_text(encoding="utf-8"))
    for item in manifest["files"]:
        source = snapshot / item["file"].replace("\\", "/")
        if digest(source) != item["sha256"]:
            raise ValueError(f"Source checksum differs for {source}")
    if len(list((snapshot / "training").glob("trace_*.json.gz"))) != 80:
        raise ValueError("Expected 80 frozen training traces")
    return snapshot


def restore_snapshot(original):
    """Restore absent snapshot members from an explicitly supplied source tree."""
    snapshot = BASE / "source_snapshot"
    manifest = json.loads((snapshot / "manifest.json").read_text(encoding="utf-8"))
    imports = []
    for item in manifest["files"]:
        relative = Path(item["file"].replace(chr(92), "/"))
        source, target = original / relative, snapshot / relative
        if digest(source) != item["sha256"]:
            raise ValueError(f"Original differs from recorded data: {source}")
        if target.is_file() and digest(target) != item["sha256"]:
            raise ValueError(f"Existing snapshot differs from recorded data: {target}")
        if not target.exists():
            imports.append((source, target))
    for source, target in imports:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    check_snapshot()


def prepare(destination):
    destination = Path(destination).resolve()
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError("The replay directory must be empty.")
    snapshot = check_snapshot()
    requirements = BASE.parent / "requirements_neural.txt"
    if not requirements.is_file():
        raise FileNotFoundError(requirements)
    destination.mkdir(parents=True, exist_ok=True)
    shutil.copytree(snapshot, destination / "source_snapshot")
    for name in HELPERS:
        shutil.copy2(BASE / name, destination / name)
    shutil.copy2(requirements, destination / "requirements_neural.txt")
    commands = ["python neural_study.py train", "python check_reproducibility.py",
                "python analyse_study.py --freeze", "python neural_study.py run --workers 4",
                "python analyse_study.py", "python extra_neural_diagnostics.py",
                "python audit_complete_inputs.py", "python render_neural_report.py"]
    record = {"runner_sha256": digest(BASE / "neural_study.py"),
              "copied_helpers": {name: digest(destination / name) for name in HELPERS},
              "numerical_settings_changed": False, "commands": commands}
    (destination / "replay_adaptation.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(destination)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path)
    parser.add_argument("--verify-snapshot", action="store_true")
    parser.add_argument("--snapshot-original", action="store_true",
                        help="Restore missing members from the recorded original source tree.")
    parser.add_argument("--original-directory", type=Path,
                        help="Original source directory (or set TRAIN_DOSY_NEURAL_SOURCE).")
    args = parser.parse_args()
    if args.snapshot_original:
        original = args.original_directory or os.environ.get("TRAIN_DOSY_NEURAL_SOURCE")
        if not original:
            parser.error("--snapshot-original requires --original-directory or TRAIN_DOSY_NEURAL_SOURCE")
        restore_snapshot(Path(original).resolve())
    if not (args.destination or args.verify_snapshot or args.snapshot_original):
        parser.error("Specify a fresh --destination or --verify-snapshot.")
    if args.verify_snapshot:
        check_snapshot()
        print("Verified source and 80 training traces")
    if args.destination:
        prepare(args.destination)
