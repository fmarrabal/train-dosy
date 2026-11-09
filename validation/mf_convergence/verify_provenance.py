"""Verify the recorded MATLAB source against the files shipped in this layout."""
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parent
OLD_GRID = "mf_original/dosy_v4/+dosyv4/"
NEW_GRID = "+dosygrid/"


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def recorded_source_bytes(data):
    """Reverse only the two MATLAB package aliases changed during file curation."""
    return (data.replace(b"dosygrid.", b"dosyv4.")
                .replace(b"fullfile(here,'..')", b"fullfile(here,'..','mf_original','dosy_v4')"))


def check_manifest(filename, subtree):
    rows = json.loads((ROOT / filename).read_text(encoding="utf-8"))
    checks = []
    for row in rows:
        recorded_path = row["file"].replace("\\", "/")
        if not recorded_path.startswith(subtree + "/"):
            continue
        path = recorded_path.replace(OLD_GRID, NEW_GRID)
        data = (ROOT / path).read_bytes()
        actual = sha256(data)
        recovered = sha256(recorded_source_bytes(data))
        if recovered != row["sha256"]:
            raise ValueError(f"Recorded source digest differs for {path}")
        checks.append({"file": path, "recorded_path": recorded_path,
                       "packaged_sha256": actual, "recorded_sha256": recovered,
                       "namespace_changed": actual != recovered})
    if not checks:
        raise ValueError(f"No {subtree} entries in {filename}")
    return checks


if __name__ == "__main__":
    frozen = check_manifest("source_hashes.json", "source_frozen")
    monitored = check_manifest("instrumented_start_hashes.json", "instrumented")
    if (len(frozen), len(monitored)) != (8, 11):
        raise ValueError("Incomplete recorded reference or execution-source manifest")
    print(json.dumps({"passed": True, "frozen_reference": frozen,
                      "monitored_study": monitored}, indent=2))
