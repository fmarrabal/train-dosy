"""Summarize every recorded fit, retaining numerical failures in all endpoints."""
from pathlib import Path
import json, hashlib, csv, platform
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root=Path(__file__).resolve().parent
rows=[json.loads(p.read_text()) for p in sorted((root/'results').glob('*.json'))]
config=json.loads((root/'config.json').read_text())
for row in rows:
    if row['completed'] and not row['error']:
        chosen=[s for s in row['starts'] if s['post_objective']==row['objective'] and s['post_kkt']==row['final_kkt']]
        assert chosen, f"Cannot identify selected start: {row['id']} warm{row['warm_sweeps']}"
        row['selected_start']=chosen[0]['start']
        row['selected_pre_kkt']=chosen[0]['pre_kkt']
        row['selected_pre_objective']=chosen[0]['pre_objective']
groups=[]
expected={(f'rank{r}_snr{s}',w) for r in config['ranks'] for s in config['snrs'] for w in config['warm_sweeps']}
for warm in config['warm_sweeps']:
    rr=[r for r in rows if r['warm_sweeps']==warm]
    done=[r for r in rr if r['completed'] and not r['error']]
    starts=[s for r in done for s in r['starts']]
    def med(k):return float(np.median([r[k] for r in done])) if done else None
    groups.append(dict(warm_cap=warm,recorded=len(rr),completed=len(done),failures=sum(not r['numerical_success'] for r in done),
        selected_converged=sum(r['numerical_success'] for r in done),warm_only_stationary=sum(s['pre_kkt']<=config['kkt_tol'] for s in starts),
        starts=len(starts),final_starts_stationary=sum(s['numerical_success'] for s in starts),
        median_cpu_seconds=med('cpu_seconds'),median_wall_seconds=med('wall_seconds'),
        median_test_clean_rmse_sigma=med('test_clean_rmse_sigma'),maximum_selected_kkt=max([r['final_kkt'] for r in done],default=None),
        runtime_checks=sum(s['runtime_checks'] for s in starts),unresolved_runtime_checks=sum(s['unresolved_runtime_checks'] for s in starts),
        forced_interior_fallbacks=sum(s['curvature_fallbacks'] for s in starts),
        final_condition_supported=sum(s['final_condition_supported'] for s in starts),
        sqp_limit_starts=sum(s['sqp_exitflag']==0 for s in starts),
        S_nonpositive_flags=sum(s['S_solver_nonpositive_flags'] for s in starts),
        early_warm_stops=sum(s['warm_iterations']<warm for s in starts),
        median_pre_kkt=float(np.median([s['pre_kkt'] for s in starts])) if starts else None,
        median_post_kkt=float(np.median([s['post_kkt'] for s in starts])) if starts else None,
        maximum_pre_kkt=max([s['pre_kkt'] for s in starts],default=None),
        minimum_runtime_amplitude_ratio=min([s['minimum_amplitude_ratio'] for s in starts],default=None),
        median_relative_SQP_objective_decrease=float(np.median([(s['pre_objective']-s['post_objective'])/max(abs(s['pre_objective']),1e-15) for s in starts])) if starts else None))
lookup={(r['id'],r['warm_sweeps']):r for r in rows if r['completed'] and not r['error']}
paired=[]
for id in sorted(set(r['id'] for r in rows)):
    rr=[lookup[id,w] for w in config['warm_sweeps'] if (id,w) in lookup]
    if not rr:continue
    best=min(r['objective'] for r in rr)
    for r in rr:
        paired.append(dict(id=id,warm_cap=r['warm_sweeps'],objective=r['objective'],gap_to_best_attained=r['objective']-best,
            relative_gap_to_best_attained=(r['objective']-best)/max(abs(best),1e-15),test_clean_rmse_sigma=r['test_clean_rmse_sigma'],numerical_success=r['numerical_success']))
summary=dict(expected=len(expected),recorded=len(rows),all_expected_recorded=set((r['id'],r['warm_sweeps']) for r in rows)==expected,
    completed=sum(r['completed'] and not r['error'] for r in rows),selected_converged=sum(r['numerical_success'] and not r['error'] for r in rows),
    errors=[r for r in rows if r['error']],numerical_failures=[dict(id=r['id'],warm_cap=r['warm_sweeps'],kkt=r.get('final_kkt')) for r in rows if r['completed'] and not r['numerical_success']],
    total_cpu_seconds=sum(r['cpu_seconds'] for r in rows),total_serial_wall_seconds=sum(r['wall_seconds'] for r in rows),
    groups=groups,paired_objectives=paired,fits=rows,
    analysis_environment=dict(python=platform.python_version(),platform=platform.platform(),numpy=np.__version__,matplotlib=matplotlib.__version__),
    scope='Nine paired synthetic inputs, one replicate per rank/SNR combination; fixed true factor count isolates optimization. This is not automatic order validation, experimental validation, or a proof of a universally sufficient warm budget.')
(root/'summary.json').write_text(json.dumps(summary,indent=2))
(root/'metrics.json').write_text(json.dumps(rows,indent=2))
flat=[]
for r in rows:
    if not r['completed'] or r['error']:continue
    for s in r['starts']:
        flat.append(dict(id=r['id'],rank=r['rank'],snr=r['snr'],warm_cap=r['warm_sweeps'],start=s['start'],
            selected=s['start']==r['selected_start'],pre_kkt=s['pre_kkt'],post_kkt=s['post_kkt'],
            pre_objective=s['pre_objective'],post_objective=s['post_objective'],numerical_success=s['numerical_success'],
            actual_warm_iterations=s['warm_iterations'],sqp_iterations=s['sqp_iterations'],sqp_exitflag=s['sqp_exitflag'],
            runtime_checks=s['runtime_checks'],unresolved_runtime_checks=s['unresolved_runtime_checks'],
            curvature_fallbacks=s['curvature_fallbacks'],final_condition_supported=s['final_condition_supported'],
            total_fit_cpu_seconds=r['cpu_seconds'],total_fit_wall_seconds=r['wall_seconds']))
if flat:
    with (root/'all_starts.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(flat[0]));writer.writeheader();writer.writerows(flat)
print(json.dumps({k:summary[k] for k in ['recorded','completed','selected_converged','total_cpu_seconds']},indent=2))
if not summary['all_expected_recorded']:
    raise SystemExit(0)

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.titlesize':9,'axes.labelsize':8,'xtick.labelsize':8,'ytick.labelsize':8,'legend.fontsize':8,'pdf.fonttype':42,'savefig.dpi':180})
fig,axs=plt.subplots(3,3,figsize=(7,6.4),layout='constrained',sharex=True,sharey=True)
for i,rank in enumerate([1,2,3]):
 for j,snr in enumerate([50,100,200]):
    ax=axs[i,j];id=f'rank{rank}_snr{snr}';rr=[lookup.get((id,w)) for w in config['warm_sweeps']]
    for start,color in [(0,'#db8b38'),(1,'#869baa')]:
        yy=[r['starts'][start]['post_kkt'] if r is not None else np.nan for r in rr]
        ax.semilogy(config['warm_sweeps'],np.maximum(yy,1e-16),'o--',color=color,lw=.8,ms=3,label=f'Start {start+1}')
    yy=[r['final_kkt'] if r is not None else np.nan for r in rr]
    ax.semilogy(config['warm_sweeps'],np.maximum(yy,1e-16),'o-',color='#176b96',lw=1.5,ms=3,label='Selected')
    ax.axhline(config['kkt_tol'],color='#a23a3a',ls=':',lw=1)
    ax.set_title(f'Rank {rank}, SNR {snr}');ax.set_xticks(config['warm_sweeps']);ax.spines[['top','right']].set_visible(False)
    if j==0:ax.set_ylabel('Final scaled KKT residual')
    if i==2:ax.set_xlabel('Alternating warm-start cap')
axs[0,0].legend(frameon=False,fontsize=8)
fig.savefig(root/'convergence_kkt.pdf');fig.savefig(root/'convergence_kkt.png');plt.close(fig)

table=[]
for g in groups:
    table.append(f"{g['warm_cap']} & {g['warm_only_stationary']}/{g['starts']} & {g['selected_converged']}/{g['completed']} & {g['median_cpu_seconds']:.1f} & {g['maximum_selected_kkt']:.2e} & {g['median_test_clean_rmse_sigma']:.4f}" + r" \\")
latex=r'''\subsection{Controlled TRAIn-MF convergence and runtime curvature checks}\label{app:mfconvergence}
The supplementary study uses an isolated, instrumented copy of the regularized
MATLAB reference version 5.1, without substituting the later signed-data solver.
Nine synthetic matrices cross one, two and three finite-width shared profiles
with SNR 50, 100 and 200. Each contains 32 acquisitions and 24 signal-bearing
spectral channels with correlated amplitudes and overlapping peaks. The fitted
factor count is fixed to the generating count to isolate optimization; this
experiment does not validate automatic order selection. All 256 diffusion
nodes span $[0.1,15]\times10^{-9}\,\mathrm{m^2\,s^{-1}}$. The log-profile width is
$0.14$, with generating centers $(0.7,2.3,7.1)\times10^{-9}\,\mathrm{m^2\,s^{-1}}$
used in order as the factor count increases. The acquisition design is
$b_i=2.2\times10^9(i/31)^{1.7}$ for $i=0,\ldots,31$; every fourth row is reserved
for external prediction, leaving 24 fitting rows. Independent Gaussian noise
has $\sigma=1/\mathrm{SNR}$ after normalizing the maximum unattenuated signal to one.
The observations, two initialization seeds, penalties and tolerances are
held fixed across alternating warm-start caps of 1, 4, 8 and 16. Every arm has
the same subsequent 250-iteration SQP cap, with $\lambda_S=\lambda_A=10^{-8}$,
scaled KKT tolerance $10^{-6}$, objective tolerance $10^{-9}$, and inner-solver
tolerance $10^{-10}$ with a 2000-iteration cap. An early warm stop is permitted only
by the original objective-and-KKT rule. The configuration, seeds and inputs were
stored before inversion. A start already meeting the KKT rule may omit SQP,
as in the frozen reference. Numerical failures remain in the reported endpoints.

\begin{table}[htbp]\centering\small
\caption{Controlled warm-start comparison. Warm-only stationarity counts both
starts; final stationarity concerns the selected fit. CPU times are medians per
two-start fit, including instrumentation. The maximum final KKT and median clean
external prediction error include unconverged outputs.}
\begin{tabular}{rrrrrr}\toprule
Warm cap & Pre-SQP pass & Final pass & CPU (s) & Max KKT & RMSE/$\sigma$\\\midrule
'''+ '\n'.join(table)+r'''
\bottomrule\end{tabular}\end{table}

For the interior second-density-difference operator and $\lambda_S>0$ used
here, before every profile-block update and at each reported SQP iterate the revision checks
the conditional curvature criterion through $Kv\ne0$ and full row rank of $A$,
using the declared relative tolerance $10^{-10}$. If the criterion is numerically
unresolved before a profile update, that block uses an interior-point constrained
least-squares fallback with the same objective and constraints; the original
feasibility, descent and KKT checks still apply. No additional ridge, component
merging or chemical uniqueness claim is introduced. Numerical support at finitely
many checked iterates is not a uniform coercivity estimate or a convergence
certificate for the alternating/SQP procedure. Five targeted tests include
independent amplitude rows, duplicated rows, a constant kernel, zero amplitudes,
and an actual duplicated-row fallback. A separate warm-eight regression against
the uninstrumented frozen reference gives exactly equal predictions on the
declared rank-one check. These checks concern the new instrumented experiment,
not runtime checks retroactively attributed to the archived figure.

The panel contains one realization per rank/SNR combination, and diagnoses
finite-budget optimization rather than proving factor convergence or physical
identification. In particular, a larger alternating warm budget cannot be
declared universally sufficient merely because it is larger.
'''
audit_path=root/'completed_output_audit.json'
if audit_path.exists():
    audit=json.loads(audit_path.read_text())
    summary['replay_audit_passed']=audit['passed']
    summary['warm8_vs16']=audit['warm8_vs16']
    assert audit['passed'], 'Completed-output replay failed; do not generate a final interpretation.'
    pairs=audit['warm8_vs16']
    bywarm={g['warm_cap']:g for g in groups}
    runtime_total=sum(g['runtime_checks'] for g in groups)
    runtime_unresolved=sum(g['unresolved_runtime_checks'] for g in groups)
    runtime_fallback=sum(g['forced_interior_fallbacks'] for g in groups)
    final_support=sum(g['final_condition_supported'] for g in groups)
    limit_starts=sum(g['sqp_limit_starts'] for g in groups)
    S_solver_flags=sum(g['S_nonpositive_flags'] for g in groups)
    pre_pass=sum(g['warm_only_stationary'] for g in groups)
    total_starts=sum(g['starts'] for g in groups)
    no_sqp_starts=sum(s['sqp_exitflag'] is None for r in rows for s in r['starts'])
    unresolved_events=[dict(id=r['id'],warm_cap=r['warm_sweeps'],start=s['start'],checks=s['unresolved_runtime_checks'],fallbacks=s['curvature_fallbacks'])
        for r in rows for s in r['starts'] if s['unresolved_runtime_checks']]
    max_pred_delta=max(p['prediction_difference_external_rmse_sigma'] for p in pairs)
    median_pred_delta=float(np.median([p['prediction_difference_external_rmse_sigma'] for p in pairs]))
    max_obj_delta=max(abs(p['relative_objective_difference_16_minus_8']) for p in pairs)
    paragraph=(f"Only {pre_pass}/{total_starts} warm-only starts pass the declared KKT tolerance. "
        f"After the common SQP refinement, the selected fits pass in "
        + ', '.join(f"{g['selected_converged']}/9" for g in groups)
        + f" cases for caps 1, 4, 8 and 16, respectively; {limit_starts}/{total_starts} starts terminate at the SQP iteration cap. "
        f"SQP runs in {total_starts-no_sqp_starts}/{total_starts} starts; any omitted refinement follows the original KKT rule. "
        f"Across the nine paired inputs, warm-eight and warm-sixteen selected predictions differ by "
        f"a median {median_pred_delta:.4f} and maximum {max_pred_delta:.4f} noise units on external rows, "
        f"while their largest absolute relative objective difference is {100*max_obj_delta:.4f}\\%. "
        "Small predictive or objective differences do not replace the failed KKT checks. "
        "Eight warm sweeps are an initialization budget, and neither eight nor sixteen "
        "guarantees stationary termination under this finite SQP budget.")
    counts=(f"The experiment records {runtime_total} before-block/reported-SQP-iterate curvature checks, "
        f"with {runtime_unresolved} numerically unresolved conditions and {runtime_fallback} "
        f"condition-triggered interior-point fallbacks. The final condition is supported for "
        f"{final_support}/{total_starts} starts. The targeted duplicate-row test exercises the "
        "fallback separately. Reevaluation of all 36 saved fits reproduces the stored "
        "pre-SQP/final objectives, KKT residuals and training-row predictions, checks "
        "common initializations and feasibility, and verifies monotonicity of saved "
        "objective histories at the original tolerance. This is a fresh-process audit "
        "of saved states, not a second execution of the optimizer or a reevaluation of "
        "every historical iterate. Runtime support and finite-budget "
        "stationarity are thus empirically different diagnostics.")
    if len(unresolved_events)==4 and runtime_unresolved==4 and {e['warm_cap'] for e in unresolved_events}=={1,4,8,16} and {(e['id'],e['start']) for e in unresolved_events}=={('rank3_snr200',2)}:
        counts += " The four flagged checks arise in the same second initialization of the rank-three/SNR-200 input, once per warm cap; they are not four independent data realizations."
    latex+='\n\n'+paragraph+'\n\n'+counts+'\n'
    summary['interpretation']={'warm_only_KKT_pass':pre_pass,'starts':total_starts,
        'runtime_checks':runtime_total,'unresolved_runtime_checks':runtime_unresolved,
        'forced_interior_fallbacks':runtime_fallback,'final_condition_supported':final_support,'unresolved_events':unresolved_events,
        'SQP_cap_starts':limit_starts,'SQP_runs':total_starts-no_sqp_starts,'starts_omitting_SQP':no_sqp_starts,'S_block_nonpositive_optimizer_flags':S_solver_flags,
        'median_external_prediction_difference8vs16_sigma':median_pred_delta,
        'max_external_prediction_difference8vs16_sigma':max_pred_delta,'max_abs_relative_objective_difference8vs16':max_obj_delta}
    (root/'summary.json').write_text(json.dumps(summary,indent=2))
    methods=(root/'EVIDENCE_EN.txt').read_text()
    methods=methods.split('\nFINAL EXECUTED OUTCOMES\n')[0]
    methods=methods.replace('Status: RUNNING. This methods record contains no final outcome claim.',
        f"Status: COMPLETE. All 36 fits and the completed-output replay audit are recorded; {summary['selected_converged']}/36 selected fits pass the declared KKT tolerance.")
    outcomes=['FINAL EXECUTED OUTCOMES',paragraph.replace(r'\%','%'),counts,'',
        f"Total summed CPU time: {summary['total_cpu_seconds']:.1f} s. Total serial sum of per-fit wall times: {summary['total_serial_wall_seconds']:.1f} s. These are not parallel-study elapsed time.",
        f"The S-block least-squares solves returned {S_solver_flags} nonpositive optimizer flags. All returned starts remain feasible in the independent replay audit.",
        'Warm cap | warm-only KKT pass / starts | final selected KKT pass / fits | median CPU s | maximum selected KKT | median clean external RMSE/sigma']
    outcomes += [f"{g['warm_cap']} | {g['warm_only_stationary']}/{g['starts']} | {g['selected_converged']}/{g['completed']} | {g['median_cpu_seconds']:.1f} | {g['maximum_selected_kkt']:.6g} | {g['median_test_clean_rmse_sigma']:.6g}" for g in groups]
    outcomes += ['', 'KKT pass is independently recomputed from the full constrained objective. It is not the same field as the SQP exit flag. Nonstationary outputs remain in objective and prediction summaries; no additional fitting or tolerance relaxation was used to remove failures.',
        'summary.json is the authoritative complete summary. all_starts.csv contains both starts and selection flags for all 36 fits. mf_results.tex is the manuscript fragment; convergence_kkt.pdf is the compact scientific figure. completed_output_audit.json records the replay, and provenance_audit.json verifies unchanged source trees.']
    (root/'EVIDENCE_EN.txt').write_text(methods.rstrip()+'\n\n'+'\n\n'.join(outcomes)+'\n')
(root/'mf_results.tex').write_text(latex)
manifest=[]
for p in sorted(root.rglob('*')):
 if p.is_file() and p.suffix in ['.m','.py','.json','.mat','.tex','.txt','.pdf','.png','.log','.csv'] and 'work_rebuild_check' not in p.parts and p.name!='final_manifest.json':
    manifest.append(dict(file=str(p.relative_to(root)),sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
(root/'final_manifest.json').write_text(json.dumps(manifest,indent=2))
