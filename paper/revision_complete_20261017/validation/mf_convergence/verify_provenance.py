"""Verify immutable source snapshots and the version actually executed."""
from pathlib import Path
import hashlib
import json
import subprocess

root = Path(__file__).resolve().parent
repo = root.parents[3]
digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
records = []
for item in json.loads((root / 'source_hashes.json').read_text()):
    relative = Path(item['file'])
    if relative.parts[0] != 'source_frozen':
        continue  # The initial instrumentation draft is not the execution manifest.
    snapshot = root / relative
    original = repo / 'matlab' / 'reference' / Path(*relative.parts[1:])
    records.append(dict(file=str(relative),initial_hash_matches=digest(snapshot)==item['sha256'],
                        live_original_matches=snapshot.read_bytes()==original.read_bytes()))
running = []
for item in json.loads((root / 'instrumented_start_hashes.json').read_text()):
    running.append(dict(file=item['file'],unchanged=digest(root/item['file'])==item['sha256']))
git = subprocess.run(['git','diff','--quiet','v0.1.0','--','matlab/reference/mf_robust',
                      'matlab/reference/mf_original/dosy_v4'],cwd=repo,capture_output=True,text=True)
report=dict(source_records=records,executed_source_records=running,
            baseline_equal_to_frozen_tag_v010=git.returncode==0,
            git_check_returncode=git.returncode,
            passed=all(r['initial_hash_matches'] and r['live_original_matches'] for r in records)
                   and all(r['unchanged'] for r in running) and git.returncode==0)
(root/'provenance_audit.json').write_text(json.dumps(report,indent=2))
print(json.dumps({k:report[k] for k in ['passed','baseline_equal_to_frozen_tag_v010']},indent=2))
assert report['passed'], 'Source provenance changed: inspect provenance_audit.json.'
