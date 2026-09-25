"""Offline evidence audit after local examples, MATLAB and client runs."""
from pathlib import Path
import csv,json,hashlib,sys,platform,importlib.metadata as md
import numpy as np
from scipy.io import loadmat
ROOT=Path(__file__).resolve().parents[1]
def read(name):return json.loads((ROOT/name).read_text(encoding='utf-8'))
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
source=read('provenance/source_hashes.json')
assert all(digest(ROOT/r['path'])==r['sha256'] for r in source)
summary=read('examples/summary.json')
assert len(summary)==6 and all(x['success'] and x['selected_rank']==x['true_rank'] for x in summary)
csharp=read('examples/three_components/csharp_result.json');python=read('examples/three_components/RAI-S.json')
np.testing.assert_allclose(csharp['prediction'],python['prediction'],rtol=1e-12,atol=1e-12)
matlab=read('examples/two_components/matlab_api_result.json');python=read('examples/two_components/RAI-S.json')
np.testing.assert_allclose(matlab['prediction'],python['prediction'],rtol=1e-12,atol=1e-12)
mf=read('examples/one_component/MF_API.json')
direct=loadmat(ROOT/'examples/one_component/MF_AUTO.mat',simplify_cells=True)
np.testing.assert_allclose(mf['X'],direct['X'],rtol=1e-10,atol=1e-12)
assert mf['success'] and mf['selected_rank']==1 and mf['kkt']<=1e-6
assert np.asarray(mf['X']).shape[0]==256
expected='0ce63ab29e7cfbffc867ec569a6540827b9e63da9a30d5a125ce26558c639ed3'
assert digest(ROOT/'paper/reproduction_inputs/ilt_unified_benchmark_20260924/results/r3_snr200_rep1_mf_auto.mat')==expected
# Exported renderer has no dependency on estimator imports or truth in the API.
frozen=list(csv.DictReader((ROOT/'benchmarks/frozen/metrics.csv').open(encoding='utf-8')))
smoke=list(csv.DictReader((ROOT/'tmp/atomic_smoke/metrics.csv').open(encoding='utf-8')))
max_delta=0.
for row in smoke:
    original=next(r for r in frozen if r['id']==row['id'] and r['method']==row['method'] and r['partition']=='test')
    assert int(row['rank'])==int(original['rank'])
    max_delta=max(max_delta,abs(float(row['test_clean_rmse_sigma'])-float(original['test_clean_rmse_sigma'])))
assert max_delta<1e-5
deps=['numpy','scipy','pydantic','fastapi','uvicorn','starlette','pytest','httpx','matplotlib','setuptools']
versions={d:md.version(d) for d in deps}
(ROOT/'requirements-lock.txt').write_text('\n'.join(f'{d}=={v}' for d,v in versions.items())+'\n')
report=dict(python=sys.version.split()[0],platform=platform.platform(),dependencies=versions,
    immutable_sources_verified=len(source),examples=summary,csharp_http_prediction_parity=True,
    matlab_http_prediction_parity=True,native_mf_api_prediction_parity=True,
    mf_kkt=mf['kkt'],mf_selected_rank=mf['selected_rank'],mf_bins=256,
    benchmark_smoke_fits=len(smoke),benchmark_smoke_max_clean_rmse_delta=max_delta,
    original_mf_figure_sha256=expected,
    limits=['C# is a client, not an independent solver','No full new 48-case rerun; immutable prior metrics retained',
            'No new real-data validation','No neural retraining in public package'])
(ROOT/'verification').mkdir(exist_ok=True)
(ROOT/'verification/release_audit.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in report.items() if k not in ['dependencies','examples','limits']},indent=2))

