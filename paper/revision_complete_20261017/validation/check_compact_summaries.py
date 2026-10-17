"""Read-only checks of compact outcomes; no fits, downloads or output changes."""
from pathlib import Path
import json
B=Path(__file__).resolve().parent
def load(p):return json.loads(p.read_text(encoding='utf-8'))
weak=[load(p) for p in (B/'weak_power/results').glob('*.json')]
assert len(weak)==96
recovered={m:sum(bool(o['metrics'].get('minority_rate_recovered',False)) for r in weak for o in r['outcomes'] if o['method']==m) for m in ('RAI','DOME')}
assert recovered=={'RAI':45,'DOME':44}
mf=[load(p) for p in (B/'mf_convergence/results').glob('*.json')]
assert len(mf)==36 and sum(r['numerical_success'] for r in mf)==14
ext=[load(p) for p in (B/'external/results').glob('*.json')]
analysed=[r for r in ext if not(r['kind']=='iid' and r['index']==0)]
assert len(ext)==504 and len(analysed)==497
nulls=list((B.parents[1]/'revision_20261017/validation/run_02/null_results').glob('*.json'))
assert len(nulls)==300
traces=list((B/'neural_ablation/source_snapshot/training').glob('trace_*.json.gz'))
checkpoints=list((B/'neural_ablation/checkpoints').glob('*.npz'))
inputs=list((B/'mf_convergence/inputs').glob('*.mat'))
assert (len(traces),len(checkpoints),len(inputs))==(80,5,9)
print(json.dumps(dict(passed=True,null_matrices=300,weak_matrices=96,weak_rate_recovery=recovered,
    selected_mf_fits=36,mf_kkt_passes=14,external_recorded=504,external_analysed=497,
    training_traces=80,checkpoints=5,mf_synthetic_inputs=9,fit_execution=False),indent=2))
