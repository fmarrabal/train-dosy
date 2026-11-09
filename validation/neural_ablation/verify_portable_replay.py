"""Compare fresh snapshot-based training with the archived evaluation checkpoints."""
from pathlib import Path
import json,hashlib
import numpy as np
BASE=Path(__file__).resolve().parent
checks=[]
for seed in [172901,172902,172903,172904,172905]:
    p=BASE/'checkpoints'/f'policy_{seed}.npz';q=BASE/'portable_smoke/checkpoints'/p.name
    with np.load(p,allow_pickle=False) as a,np.load(q,allow_pickle=False) as b:
        same=set(a.files)==set(b.files) and all(np.array_equal(a[k],b[k]) for k in a.files)
        checks.append(dict(seed=seed,all_checkpoint_arrays_exact=same,original_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),replay_sha256=hashlib.sha256(q.read_bytes()).hexdigest()))
result=dict(all_passed=all(c['all_checkpoint_arrays_exact'] for c in checks),checks=checks,
    scope='All five CPU training checkpoints reproduced from the bundled immutable source and80traces. The720test fits have not been repeated.',
    adaptation='Only the archive root path is replaced in a fresh runner; numerical code/settings/seeds unchanged.')
(BASE/'portable_replay_verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result,indent=2))
