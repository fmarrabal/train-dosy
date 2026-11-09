from pathlib import Path
import os,json
os.environ['OPENBLAS_NUM_THREADS']='1'
import numpy as np
BASE=Path(__file__).parent
METHODS=['RAI_S','DOME_S','SCORE','SCORE_NNLS','LOCODOSY','Tikhonov','PALMA']
LABELS={'RAI_S':'RAI-S','DOME_S':'DOME-S','SCORE':'SCORE-CV (signed)','SCORE_NNLS':'SCORE-NNLS-CV','LOCODOSY':'LOCODOSY (signed)','Tikhonov':'Positive Tikhonov','PALMA':'PALMA objective + CV'}
def interval(x,seed=11791):
    x=np.asarray(x);rng=np.random.default_rng(seed)
    samples=np.array([rng.choice(x,len(x),replace=True).mean() for _ in range(10000)])
    return [float(v) for v in np.quantile(samples,[.025,.975])]
def main():
    data=[json.loads(p.read_text()) for p in (BASE/'results').glob('*.json')]
    # The first matrix was used to profile adapters before the full comparison.
    # Retain it in raw outputs and exclude it symmetrically in reported iid means.
    rows=[r for r in data if not(r['kind']=='iid' and r['index']==0)]
    summary=[]
    for kind in ['iid','baseline','correlated']:
        for method in METHODS:
            group=[r for r in rows if r['kind']==kind and r['method']==method]
            positive=[r for r in group if r['scenario']!='null']
            if not group:continue
            vals=[r['metrics']['test_clean_rmse_sigma'] for r in positive if 'test_clean_rmse_sigma' in r['metrics']]
            shapes=[r['metrics']['w1_logD'] for r in positive if 'w1_logD' in r['metrics']]
            q=dict(kind=kind,method=method,n=len(group),signal_n=len(positive),returned=sum(r['error'] is None for r in group),success=sum(r['success'] for r in group),positive_representation=sum(r['physical'] for r in group),E=float(np.mean(vals)) if vals else None,E_n=len(vals),W1=float(np.mean(shapes)) if shapes else None,W1_n=len(shapes),seconds=sum(r['seconds'] for r in group))
            if method=='PALMA':
                flags=[f for r in group for f in r.get('diagnostics',{}).get('column_converged',[])]
                q['column_converged']=sum(flags);q['column_n']=len(flags)
            summary.append(q)
    paired=[]
    for kind in ['iid','baseline','correlated']:
        reference={r['index']:r for r in rows if r['kind']==kind and r['method']=='RAI_S' and r['scenario']!='null'}
        for method in METHODS[2:]:
            matches=[(reference[r['index']],r) for r in rows if r['kind']==kind and r['method']==method and r['index'] in reference]
            delta=[b['metrics']['test_clean_rmse_sigma']-a['metrics']['test_clean_rmse_sigma'] for a,b in matches if 'test_clean_rmse_sigma' in a['metrics'] and 'test_clean_rmse_sigma' in b['metrics']]
            if delta:paired.append(dict(kind=kind,comparison=method+' minus RAI_S',n=len(delta),mean=float(np.mean(delta)),interval95=interval(delta),inference='Descriptive paired problem bootstrap; no multiplicity adjustment, all returned estimates including nonconverged fits.'))
    (BASE/'final_summary.json').write_text(json.dumps(dict(raw_results=len(data),analysed_results=len(rows),pilot_index_excluded_from_iid=0,summaries=summary,paired=paired),indent=2),encoding='utf-8')
    table=r'''\subsection{Shared external-reference comparison}\label{app:externalcomparison}
The additional comparison uses the same frozen observations, masks and outer
test acquisitions. The physical interval is common to grid methods and
positive-measure eligibility. Native continuous fits are retained with their
original rate constraints rather than silently projected into that interval. Matrix
zero served as an adapter pilot and is retained in the raw outputs but excluded
symmetrically from the iid summary, leaving 47 matrices (41 signal-present and
six null). Twelve underlying signal-present problems also receive each of the
baseline and correlated-noise stresses. These are retrospective comparisons,
not a new untouched synthetic population. Outer test rows are not used in
the internal parameter or order decisions.

SCORE uses the attributed GNAT translation, signed conditional least squares
and its original initialization, with a declared reserved-acquisition wrapper
selecting orders zero through four. Candidate selection enforces interval
eligibility, but the final continuous refit is unconstrained in log rate;
out-of-interval results are retained and flagged. The positive-amplitude
sensitivity uses the same wrapper and proposal budget. LOCODOSY retains its native local SVD
order and reduction rules on the common contiguous signal windows; its sum of
local components is not a global molecular count. Its native rate domain has
zero as the lower limit and the common upper limit. LOCODOSY uses the training
observations for its order rule; only the validation-selected adapters use
the internal split. Five of the 47 iid LOCODOSY outputs contain rates outside
the declared positive-measure interval; none is projected or discarded.
No outer test row participates in these decisions. Positive Tikhonov selects
its curvature weight from $\{0.01,1,100,10^4\}$ on internal validation.
The PALMA constrained objective uses a training-only positive low-$b$ scale,
entropy weights $\{0.01,0.1,0.5\}$ selected on internal validation, a declared
95\% Gaussian noise-ball radius and 5000 final PPXA+ iterations. A batched
Woodbury graph projection was checked against the reviewed scalar QR engine;
this is an implementation of the stated objective, not a claim of identical
performance to all upstream application presets.

\begin{table}[htbp]\centering\small
\caption{Common external-reference comparison. $E_U$ is clean external error
in noise units over signal-present cases. Numerical success uses each method's
own criterion; all returned outputs enter $E_U$, including nonconverged ones.
The denominator after $W_{1,h}$ counts only outputs admitting a nonnegative
measure within the declared interval and is therefore a different population
for signed methods.}\label{tab:externalexecuted}
\begin{tabularx}{\linewidth}{@{}Xrrr@{}}\toprule
Method & Numerical success & $E_U$ & $W_{1,h}$ [n]\\\midrule
'''
    for q in summary:
        if q['kind']!='iid':continue
        shape='-- [0]' if q['W1'] is None else f"{q['W1']:.4f} [{q['W1_n']}]"
        e='--' if q['E'] is None else f"{q['E']:.5f}"
        table+=f"{LABELS[q['method']]} & {q['success']}/{q['n']} & {e} & {shape}\\\\\n"
    table+=r'''\bottomrule\end{tabularx}\end{table}

This table does not rank chemical identification. Signed cancellations may
lower prediction error without producing a positive component representation,
and unresolved PPXA+ optimality prevents interpreting a budget-limited PALMA
result as its exact optimum. All convergence flags and curve-level constraint
diagnostics are retained. CONTIN, UPEN, DRECT and DREAM have not been run in
this common comparison; the interface and validation requirements in Table~
\ref{tab:externalmethods} remain applicable. Positive Tikhonov is not relabeled
as the complete CONTIN procedure. These data replace comparison-by-citation
with a bounded executed comparison to the closest joint-component references
and two positive distributional controls.

On the twelve baseline-stress cases, positive SCORE has a slightly smaller
mean clean error than RAI-S ($1.00045$ versus $1.01092$ noise units); the
paired descriptive interval for the difference is $[-0.02583,0.00196]$ and
includes zero. There is no demonstrated robust advantage on this offset
panel. Correlated-noise results and all paired contrasts are included in
the machine-readable supplement; they are not independent replications of
the underlying twelve signal configurations.
'''
    (BASE/'external_results.tex').write_text(table,encoding='utf-8')
    print('External analysis',len(data),'fits; iid pilot excluded; summaries',len(summary))
if __name__=='__main__':main()
