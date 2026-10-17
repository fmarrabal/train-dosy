"""Read-only numerical aggregation; no fit, selection, or parameter changes."""
from pathlib import Path
import csv,hashlib,json
import numpy as np
from scipy.stats import beta
ROOT=Path(__file__).resolve().parents[3];BASE=Path(__file__).resolve().parent

def dump(path,obj):path.write_text(json.dumps(obj,indent=2,allow_nan=False),encoding='utf-8')
def ci(k,n):return dict(events=k,n=n,rate=k/n,exact_two_sided_95=[0. if k==0 else float(beta.ppf(.025,k,n-k+1)),1. if k==n else float(beta.ppf(.975,k+1,n-k))],exact_one_sided_upper_95=1. if k==n else float(beta.ppf(.95,k+1,n-k)))
nulls=[json.loads(p.read_text()) for p in sorted((BASE/'run_02/null_results').glob('*.json'))]
strata=[]
for sigma in [.02,.01,.005]:
    for mode in ['fixed_window','independent_discovery']:
        for method in ['RAI','DOME']:
            rows=[o for n in nulls if n['sigma']==sigma for o in n['outcomes'] if o['mode']==mode and o['method']==method]
            valid=[o for o in rows if o['error'] is None]
            strata.append(dict(sigma=sigma,mode=mode,method=method,**ci(sum(o['rank']>0 for o in valid),len(valid)),
                              fits=sum(o['fit_performed'] for o in valid),errors=len(rows)-len(valid)))
dump(BASE/'run_02/null_strata.json',dict(strata=strata,
    pooled_scope='Three prespecified noise-scale strata,100 independent problems each; stratified exact intervals avoid requiring identical event probabilities across scales. Pooled binomial intervals in summary are descriptive for the equally weighted generator and not a general FWER guarantee.',
    paired_methods='The same300 inputs are fitted by both methods: this is not600 independent replicates.'))
rows=list(csv.DictReader((BASE/'run_02/ablation_metrics.csv').open()));rng=np.random.default_rng(9925);indices=rng.integers(0,24,(10000,24));paired=[]
for method in ['RAI','DOME']:
    for old,new in [('native_order_common_refit','prediction_only'),('prediction_only','screen_and_prediction')]:
        a={r['id']:r for r in rows if r['method']==method and r['arm']==old};b={r['id']:r for r in rows if r['method']==method and r['arm']==new}
        for field in ['spectral_leakage_fraction','unmatched_mass_fraction','test_clean_rmse_sigma','w1_logD']:
            delta=np.array([float(b[k][field])-float(a[k][field]) for k in sorted(a)])
            paired.append(dict(method=method,from_arm=old,to_arm=new,metric=field,mean_change=float(delta.mean()),
                               descriptive_bootstrap95=np.quantile(delta[indices].mean(1),[.025,.975]).tolist()))
dump(BASE/'run_02/ablation_paired.json',dict(paired=paired,bootstrap_replicates=10000,seed=9925,
    scope='Post-hoc24-problem paired descriptive intervals; no simultaneous coverage or independent confirmation claim.'))
original=list(csv.DictReader((ROOT/'paper/evidence/metrics.csv').open()));ids={f'test_{i:03d}_' for i in list(range(6))+list(range(8,14))}
baseline=[]
for method in ['RAI','RAI_S','DOME','DOME_S']:
    q=[r for r in original if r['partition']=='test' and r['method']==method and any(r['id'].startswith(prefix) for prefix in ids)]
    baseline.append(dict(method=method,n=len(q),correct_order=sum(r['correct_rank']=='True' for r in q),
                         order_and_rates=sum(r['correct_rank']=='True' and r['matched_components']==r['r_true'] for r in q),
                         **{f:float(np.mean([float(r[f]) for r in q])) for f in ['spectral_leakage_fraction','unmatched_mass_fraction','test_clean_rmse_sigma','w1_logD']}))
dump(BASE/'distortion_stress/baseline_same12.json',dict(baseline=baseline,
    source='paper/evidence/metrics.csv',source_sha256=hashlib.sha256((ROOT/'paper/evidence/metrics.csv').read_bytes()).hexdigest(),
    note='Same12 underlying truths and original masks. AR1 replaces the noise realization; other distortions retain original Gaussian background.'))
status=[]
for p in sorted((BASE/'distortion_stress/results').glob('*.json')):
    case=json.loads(p.read_text())
    for r in case['results']:
        if r.get('error'):ok=False
        elif r['method']=='RAI':ok=r['diagnostics'].get('optimizer',{}).get('success')
        else:ok=r['numerical_status_reported']
        status.append(dict(id=case['id'],kind=case['kind'],method=r['method'],success=ok))
dump(BASE/'distortion_stress/status_audit.json',dict(results=status,
    known=sum(r['success'] is not None for r in status),failed=sum(r['success'] is False for r in status),
    unknown=sum(r['success'] is None for r in status),
    note='Native RAI status is diagnostics.optimizer.success; top-level numerical_status_reported was null. Extraction only, no refit or changed metrics.'))
for q in strata:
    if q['method']=='RAI' and q['mode']=='fixed_window':print('sigma',q['sigma'],q['events'],q['n'],q['exact_two_sided_95'])
print('baseline',baseline)
