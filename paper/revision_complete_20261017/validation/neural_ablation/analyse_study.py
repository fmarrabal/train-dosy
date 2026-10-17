"""Final reporting; pooled bootstrap resamples whole problems within each domain."""
from pathlib import Path
import json,hashlib,argparse
from datetime import datetime,timezone
import numpy as np
BASE=Path(__file__).resolve().parent
SEEDS=[172901,172902,172903,172904,172905]
DOMAINS=['ID','OOD_noise','OOD_acquisition']
METHODS=['classical_shortlist']+[f'net_seed_{s}' for s in SEEDS]

def dump(p,v):p.write_text(json.dumps(v,indent=2,allow_nan=False),encoding='utf-8')
def freeze():
    p=BASE/'analysis_protocol.json';digest=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    if p.exists():assert json.loads(p.read_text())['script_sha256']==digest;return
    dump(p,dict(frozen_utc=datetime.now(timezone.utc).isoformat(),script_sha256=digest,bootstrap_seed=2026101899,replicates=10000,
        correction='Review clarified pooled uncertainty: preserve domain sizes by resampling problems within each domain. Original neural_study summary pooled bootstrap is unstratified; final_summary.json from this script is authoritative for intervals. No solver/selection/checkpoint or endpoint was changed.',
        sampling_unit='Whole synthetic problem; average five neural predictions only at the metric level before calculating paired differences. Train runs share data; no600-case pseudo-replication.',
        scope='Conditional on the80-problem trace bank and five declared independent seed runs. Includes failed numerical fits in metrics; execution errors and absent pairs reported separately.',
        endpoints={'W1':'105 signal-present problems, nulls undefined and excluded','clean_RMSE_sigma':'all120 problems including15nulls'},
        subgroup_scope='Domain and scenario summaries are descriptive; no multiplicity-adjusted or clinical/chemical validation claim.',
        budget_scope='Six rounds and shortlist1 with fallback match algorithmic caps, not realized time/fit counts. Solver timings exclude checkpoint load/serialization; reused trace cost reported separately.'))

def final():
    rows=[json.loads(p.read_text())['metrics'] for p in sorted((BASE/'results').glob('*.json'))]
    assert len(rows)==720,'Wait for all120x6 outcomes; do not select a convenient subset'
    lookup={(r['index'],r['method']):r for r in rows};assert len(lookup)==720
    checks=[]
    for i in range(120):
        rr=[lookup[i,m] for m in METHODS]
        checks.append(dict(index=i,all_methods_present=True,observation_hash_shared=len({r.get('observation_sha256') for r in rr})==1))
    summaries=[];rng=np.random.default_rng(2026101899);contrasts=[]
    for domain in ['all']+DOMAINS:
        rr=[r for r in rows if domain=='all' or r['domain']==domain]
        for m in METHODS:
            x=[r for r in rr if r['method']==m];valid=[r for r in x if not r.get('error')]
            summary=dict(domain=domain,method=m,n=len(x),errors=len(x)-len(valid),numerical_failures=sum(not r['success'] for r in valid))
            for metric in ['w1_logD','test_clean_rmse_sigma','seconds','fixed_tree_fits','policy_fallbacks']:
                v=[r[metric] for r in valid if r.get(metric) is not None]
                summary[metric]=float(np.mean(v)) if v else None;summary[metric+'_n']=len(v)
            nulls=[r for r in valid if r['r_true']==0]
            summary.update(null_cases=len(nulls),null_clean_rmse_sigma=float(np.mean([r['test_clean_rmse_sigma'] for r in nulls])) if nulls else None,
                null_mean_fitted_mass=float(np.mean([r['mass'] for r in nulls])) if nulls else None)
            summaries.append(summary)
        for metric in ['w1_logD','test_clean_rmse_sigma','seconds','fixed_tree_fits']:
            delta_by_domain={d:[] for d in (DOMAINS if domain=='all' else [domain])};per_seed={s:[] for s in SEEDS}
            missing=[]
            for i in sorted({r['index'] for r in rr}):
                b=lookup[i,'classical_shortlist'];nn=[lookup[i,f'net_seed_{s}'] for s in SEEDS]
                if metric=='w1_logD' and b['r_true']==0:continue
                if b.get('error') or b.get(metric) is None or any(n.get('error') or n.get(metric) is None for n in nn):missing.append(i);continue
                dif=np.array([n[metric]-b[metric] for n in nn]);delta_by_domain[b['domain']].append(float(dif.mean()))
                for s,d in zip(SEEDS,dif):per_seed[s].append(float(d))
            arrays={d:np.asarray(v) for d,v in delta_by_domain.items() if v};n=sum(len(v) for v in arrays.values())
            if not n:continue
            boot=np.zeros(10000)
            for d,v in arrays.items():boot+=v[rng.integers(0,len(v),size=(10000,len(v)))].sum(axis=1)
            boot/=n
            allvalues=np.concatenate(list(arrays.values()))
            contrasts.append(dict(domain=domain,metric=metric,n_problems=n,domain_counts={d:len(v) for d,v in arrays.items()},
                mean=float(allvalues.mean()),paired_stratified_problem_bootstrap95=np.quantile(boot,[.025,.975]).tolist(),
                seed_mean_differences={str(s):float(np.mean(v)) for s,v in per_seed.items()},missing_pairs=missing,
                sign='neural seed-average minus classical; negative favors neural for error/time/fit-count endpoints'))
    training=[json.loads(p.read_text()) for p in sorted((BASE/'training').glob('seed_*.json'))]
    replay=json.loads((BASE/'reproducibility_checks.json').read_text())
    result=dict(methods=METHODS,n_problems=120,n_fits=720,summaries=summaries,paired_differences=contrasts,
        errors=[r for r in rows if r.get('error')],numerical_failures=[{k:r[k] for k in ['index','domain','scenario','method','relative_gap']} for r in rows if not r['success'] and not r.get('error')],
        input_pair_checks=checks,all_pair_checks_passed=all(c['observation_hash_shared'] for c in checks),
        training=[{k:r[k] for k in ['seed','seconds','best_epoch','checkpoint_sha256']} for r in training],
        training_seconds=sum(r['seconds'] for r in training),historical_trace_generation_seconds=replay['original_trace_generation_seconds'],
        reproducibility_all_passed=replay['all_passed'],total_solver_seconds=sum(r.get('seconds',0) for r in rows),
        timing=json.loads((BASE/'run_timing.json').read_text()),analysis_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    dump(BASE/'final_summary.json',result)
    print(json.dumps({'pairs_verified':result['all_pair_checks_passed'],'errors':len(result['errors']),'failures':len(result['numerical_failures']),'main_contrasts':[c for c in contrasts if c['domain']=='all']},indent=2))

def main():
    p=argparse.ArgumentParser();p.add_argument('--freeze',action='store_true');a=p.parse_args();freeze()
    if not a.freeze:final()
if __name__=='__main__':main()
