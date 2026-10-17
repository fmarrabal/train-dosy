"""Descriptive outcome concordance and family summaries; no model selection."""
from pathlib import Path
import json
import numpy as np
BASE=Path(__file__).resolve().parent
SEEDS=[172901,172902,172903,172904,172905]

def main():
    payloads=[json.loads(p.read_text()) for p in (BASE/'results').glob('*.json')]
    rows=[p['metrics'] for p in payloads]
    assert len(rows)==720,'Summarize the complete panel only'
    concordance=[]
    for seed in SEEDS:
        pred_diffs=[];mass_diffs=[];exact_pred=0;exact_mass=0;missing=[]
        for i in range(120):
            pa=BASE/'arrays'/f'case_{i:03d}__classical_shortlist.npz';pb=BASE/'arrays'/f'case_{i:03d}__net_seed_{seed}.npz'
            if not pa.exists() or not pb.exists():missing.append(i);continue
            with np.load(pa,allow_pickle=False) as a,np.load(pb,allow_pickle=False) as b:
                dp=float(np.linalg.norm(a['prediction']-b['prediction'])/max(np.linalg.norm(a['prediction']),1e-300))
                dx=float(np.linalg.norm(a['X']-b['X'])/max(np.linalg.norm(a['X']),1e-300))
                pred_diffs.append(dp);mass_diffs.append(dx);exact_pred+=np.array_equal(a['prediction'],b['prediction']);exact_mass+=np.array_equal(a['X'],b['X'])
        concordance.append(dict(seed=seed,n=len(pred_diffs),missing=missing,exact_prediction_arrays=int(exact_pred),exact_mass_arrays=int(exact_mass),
            prediction_relative_difference_below_1e_10=sum(d<=1e-10 for d in pred_diffs),mass_relative_difference_below_1e_8=sum(d<=1e-8 for d in mass_diffs),
            max_prediction_relative_difference=max(pred_diffs),max_mass_relative_difference=max(mass_diffs)))
    groups=[]
    for label,predicate in [('atomic',lambda r:r['scenario'] not in ('null','broad')),('broad',lambda r:r['scenario']=='broad'),('null',lambda r:r['scenario']=='null')]:
        for method in ['classical_shortlist']+[f'net_seed_{s}' for s in SEEDS]:
            rr=[r for r in rows if r['method']==method and predicate(r)];ok=[r for r in rr if not r.get('error')]
            values=[r['w1_logD'] for r in ok if r['w1_logD'] is not None]
            groups.append(dict(group=label,method=method,n=len(rr),errors=len(rr)-len(ok),numerical_failures=sum(not r['success'] for r in ok),
                w1_logD=float(np.mean(values)) if values else None,test_clean_rmse_sigma=float(np.mean([r['test_clean_rmse_sigma'] for r in ok])) if ok else None,
                seconds=float(np.mean([r['seconds'] for r in ok])) if ok else None))
    statuses=[]
    for method in ['classical_shortlist']+[f'net_seed_{s}' for s in SEEDS]:
        pp=[p for p in payloads if p['metrics']['method']==method and not p['metrics'].get('error')]
        per_problem=[]
        for p in pp:
            recorded=p['history'];failed=sum(not h['optimizer']['success'] for h in recorded)
            per_problem.append(dict(index=p['metrics']['index'],failed_search_fits=failed,recorded_search_fits=len(recorded),
                final_failed=not p['diagnostics']['optimizer']['success'],
                selected_training_path_failed=not p['diagnostics']['selection_optimizer']['success'],
                fixed_tree_count_matches_history=len(recorded)+1==p['metrics']['fixed_tree_fits']))
        statuses.append(dict(method=method,n=len(pp),failed_search_fits=sum(p['failed_search_fits'] for p in per_problem),
            recorded_search_fits=sum(p['recorded_search_fits'] for p in per_problem),
            problems_with_failed_search_fit=sum(p['failed_search_fits']>0 for p in per_problem),
            final_failed=sum(p['final_failed'] for p in per_problem),
            selected_training_path_failed=sum(p['selected_training_path_failed'] for p in per_problem),
            fixed_tree_counts_consistent=all(p['fixed_tree_count_matches_history'] for p in per_problem),per_problem=per_problem))
    report=dict(concordance=concordance,family_groups=groups,search_statuses=statuses,
        scope='Descriptive audit after complete evaluation. No thresholds select a model or case; array-difference tolerances characterize numerical agreement only.')
    (BASE/'extra_diagnostics.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
