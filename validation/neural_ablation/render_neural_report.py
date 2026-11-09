"""Generate a compact English report and editable LaTeX from completed evidence."""
from pathlib import Path
import json
BASE=Path(__file__).resolve().parent
SEEDS=[172901,172902,172903,172904,172905]

def f(x,n=5):return f'{x:.{n}f}' if x is not None else 'NA'
def ci(c):return '['+', '.join(f(v) for v in c['paired_stratified_problem_bootstrap95'])+']'

def main():
    z=json.loads((BASE/'final_summary.json').read_text());assert z['n_fits']==720
    extra=json.loads((BASE/'extra_diagnostics.json').read_text())
    audit=json.loads((BASE/'complete_input_audit.json').read_text())
    assert all(audit[k] for k in ['all_case_checks_passed','all_output_checks_passed','snapshot_hashes_unchanged','test_ids_unique','seeds_exact']), 'Do not claim verified evidence while a provenance/output check fails'
    assert z['reproducibility_all_passed'] and z['all_pair_checks_passed']
    search_counts='; '.join(f"{r['method']}: {r['failed_search_fits']}/{r['recorded_search_fits']} failed search fits in {r['problems_with_failed_search_fit']}/120 problems; {r['selected_training_path_failed']} failed selected training paths" for r in extra['search_statuses'])
    control_search=next(r for r in extra['search_statuses'] if r['method']=='classical_shortlist')
    neural_search=[r for r in extra['search_statuses'] if r['method']!='classical_shortlist']
    failed_range=[r['failed_search_fits'] for r in neural_search]
    problem_fail=[r['problems_with_failed_search_fit'] for r in extra['search_statuses']]
    problem_fail_text=str(problem_fail[0]) if len(set(problem_fail))==1 else f'{min(problem_fail)}--{max(problem_fail)}'
    assert all(r['selected_training_path_failed']==0 and r['final_failed']==0 for r in extra['search_statuses']), 'Revise status prose if a selected or final fit fails'
    search_tex=f"Search-stage checks failed for {control_search['failed_search_fits']}/{control_search['recorded_search_fits']} control subproblems and {sum(failed_range)}/{sum(r['recorded_search_fits'] for r in neural_search)} neural subproblems ({min(failed_range)}--{max(failed_range)} per policy). Each method encountered a failed search fit in {problem_fail_text}/120 problems, but all selected training paths and final refits passed. These are solver-event counts, not independent statistical trials."
    families={(r['group'],r['method']):r for r in extra['family_groups']}
    atomic_worse=sum(families['atomic',f'net_seed_{s}']['w1_logD']>families['atomic','classical_shortlist']['w1_logD'] for s in SEEDS)
    broad_better=sum(families['broad',f'net_seed_{s}']['w1_logD']<families['broad','classical_shortlist']['w1_logD'] for s in SEEDS)
    null_equal=all(families['null',f'net_seed_{s}']['test_clean_rmse_sigma']==families['null','classical_shortlist']['test_clean_rmse_sigma'] for s in SEEDS)
    summary={(r['domain'],r['method']):r for r in z['summaries']}
    dif={(r['domain'],r['metric']):r for r in z['paired_differences']}
    rows=[];latex=[]
    for method in z['methods']:
        r=summary['all',method];label='Classical shortlist' if method=='classical_shortlist' else 'RAI-Net '+method.split('_')[-1]
        passed=r['n']-r['errors']-r['numerical_failures']
        rows.append(f"| {label} | {f(r['w1_logD'])} | {f(r['test_clean_rmse_sigma'])} | {passed}/{r['n']} | {r['errors']} | {f(r['seconds'],2)} | {f(r['fixed_tree_fits'],2)} |")
        latex.append(f"{label} & {f(r['w1_logD'])} & {f(r['test_clean_rmse_sigma'])} & {passed}/{r['n']} & {f(r['seconds'],2)} & {f(r['fixed_tree_fits'],2)} \\\\")
    delta=dif['all','w1_logD'];error=dif['all','test_clean_rmse_sigma'];duration=dif['all','seconds'];fits=dif['all','fixed_tree_fits']
    assert all(summary['all',m]['w1_logD_n']==105 and summary['all',m]['test_clean_rmse_sigma_n']==120 for m in z['methods']), 'A missing endpoint requires an explicit incomplete-panel report'
    assert delta['n_problems']==105 and error['n_problems']==120
    domain_lines=[];domain_latex=[]
    for domain in ['ID','OOD_noise','OOD_acquisition']:
        d=dif[domain,'w1_logD'];e=dif[domain,'test_clean_rmse_sigma']
        domain_lines.append(f"- {domain}: W1 difference {f(d['mean'])}, 95% interval {ci(d)}, n={d['n_problems']} signal-present problems; clean-error difference {f(e['mean'])}, interval {ci(e)}, n={e['n_problems']} problems.")
        name=domain.replace('_',r'\_')
        dd=list(d['seed_mean_differences'].values())
        domain_latex.append(f"{name}: $\\Delta W_1={f(d['mean'])}$, interval ${ci(d)}$ ($n={d['n_problems']}$); individual seed differences range from ${f(min(dd))}$ to ${f(max(dd))}$.")
    maxmean=max(delta['seed_mean_differences'].values());minmean=min(delta['seed_mean_differences'].values())
    report='\n'.join([
        '# Independent multi-seed RAI-Net study','',
        'The prespecified study completed 120 new whole problems with one deterministic shortlist control and five separately seeded RAI-Net fits (720 returned method outcomes). The five networks reuse the same 80 historical development problems, split into 64 training and 16 validation problems separately for each seed. Mean and scale use only training records; exact checks and deterministic replay passed for all five fits. Checkpoint selection used validation loss only. The test contains 72 problems within the physical training domain, 24 with unseen noise levels, and 24 with unseen acquisition designs. No test cases or numerical failures were removed.','',
        '| Method | W1, signal present | Clean RMSE / sigma | Numerical success | Execution errors | Solver seconds | Fixed-tree fits |',
        '|---|---:|---:|---:|---:|---:|---:|',*rows,'',
        f"W1 uses 105 signal-present problems; zero-mass truth makes null W1 undefined. Prediction error uses all 120 problems, including 15 null controls. The seed-average neural-minus-classical W1 difference is {f(delta['mean'])}, with paired stratified problem-bootstrap 95% interval {ci(delta)}. The five individual seed differences range from {f(minmean)} to {f(maxmean)}. The clean-error difference is {f(error['mean'])}, interval {ci(error)}. These are averages of per-seed metric values, not an ensemble predictor.",'',
        *domain_lines,'',
        f"The post hoc family breakdown is descriptive: {atomic_worse}/5 policies have higher mean W1 than the control on the 90 atomic problems, whereas {broad_better}/5 have lower mean W1 on the 15 broad problems. Identical prediction error for all methods on the 15 null problems: {null_equal}. Family breakdowns were not used to choose a network or checkpoint.",'',
        'Search-stage numerical diagnostics (separate from final-fit status): '+search_counts+'.','',
        f"The neural-minus-classical solver-time difference is {f(duration['mean'],3)} seconds per problem (interval {ci(duration)}), and the fixed-tree-fit difference is {f(fits['mean'],3)} (interval {ci(fits)}). All algorithms have the same six-round cap, one-candidate shortlist with fallback, penalty paths, numerical verifier and inner iteration budget. Realized costs are not assumed equal. The deterministic shortlist comparator is not the exhaustive classical controller or the separate atomic RAI estimator.",'',
        f"The full evaluation took {z['timing']['seconds']:.1f} seconds of wall time with {z['timing']['workers']} workers. Summed solver time was {z['total_solver_seconds']:.1f} seconds. The five network fits and checkpoint writes took {z['training_seconds']:.3f} seconds after data loading and the Torch import. The reused traces record {z['historical_trace_generation_seconds']:.1f} seconds of summed historical generation time; this is not wall time for the present revision. Solver timing excludes checkpoint loading, metrics and serialization. Timing overlapped other CPU work and supports no hardware-independent or end-to-end speed claim.",'',
        'Four best validation checkpoints occurred at the fixed 400-epoch limit, and one at epoch 64. The protocol evaluates a specified bounded training procedure; it does not establish training convergence or rule out other architectures/training budgets. The bootstrap resamples entire problems within domains and averages the five policies before forming a paired contrast. It is conditional on the reused training bank and the declared seeds; 720 outputs are not 720 independent test problems. Neither a general neural-superiority claim nor equivalence follows from this study.','',
        'No experimental mixtures, chemical identities or instrument-calibrated OOD distributions were validated. The new frequency mask uses fitting rows only; metric evaluation remains conditional on selected channels. The 80 reused traces retain their original preprocessing. All raw inputs, outputs, numerical statuses, checkpoints, source hashes, and replay scripts are archived. The portable source snapshot reproduced all five checkpoint files byte-for-byte; the 720 test fits were run once.','',
        'Authoritative results: final_summary.json. Additional family summaries and prediction-array agreement: extra_diagnostics.json. See REPRODUCTION_NOTES_EN.txt for scope and rerun commands.'
    ])
    (BASE/'NEURAL_REPORT_EN.md').write_text(report+'\n',encoding='utf-8')
    tex='\n'.join([
        '% Draft fragment generated from final_summary.json; manuscript not edited.',
        r'\paragraph{Independent multi-seed neural evaluation.}',
        r'\label{app:neuralexpanded}',
        'Five independently seeded CPU network fits reused the 80-problem trace bank, with disjoint 64/16 whole-problem training/validation partitions within each run. Feature normalization uses training records only; deterministic checkpoint replay was exact. A new 120-problem panel contains 72 problems in the original physical domain, 24 at unseen SNRs (25, 35 and 500), and 24 with unseen acquisition exponents and diffusion weightings. All five policies and a deterministic proposal-order control use the same six-round cap, shortlist of one with fallback, and numerical verifier. This control is the classical shortlist variant, not the exhaustive controller or the atomic RAI estimator.',
        r'\begin{table}[H]',
        r'\caption{Independent neural evaluation. $W_{1,h}$ averages 105 signal-present problems; $E_U$ includes all 120 problems, including 15 null controls. Numerical success is a solver criterion. Times are mean solver seconds and exclude checkpoint loading and serialization.}',
        r'\label{tab:neuralexpanded}',r'\centering',r'\begin{tabular}{lrrrrr}',r'\toprule',
        r'Method & $W_{1,h}$ & $E_U$ & Numerical success & Seconds & Tree fits \\',r'\midrule',*latex,r'\bottomrule',r'\end{tabular}',r'\end{table}',
        f"The seed-average neural-minus-classical $W_{{1,h}}$ difference is ${f(delta['mean'])}$ (paired problem-bootstrap 95\\% interval ${ci(delta)}$); the corresponding $E_U$ difference is ${f(error['mean'])}$ (${ci(error)}$). The pooled bootstrap resamples whole problems within each domain, preserving domain counts; the five seed outputs on a problem are paired, not independent tests. Individual seed mean $W_{{1,h}}$ differences range from ${f(minmean)}$ to ${f(maxmean)}$.",
        ' '.join(domain_latex),
        f"The experiment recorded {len(z['errors'])} execution errors and {len(z['numerical_failures'])} failed final numerical checks across the 720 method outcomes. Failed numerical outputs remain in metric calculations; they are not removed to improve the averages.",
        search_tex,
        f"Mean solver time differs by ${f(duration['mean'],3)}$ seconds per problem and fixed-tree fits by ${f(fits['mean'],3)}$ (neural minus classical). Actual cost is reported because equal caps do not imply equal realized effort. The five network fits and checkpoint writes took {z['training_seconds']:.3f} seconds after data loading and the Torch import. The reused traces record {z['historical_trace_generation_seconds']:.1f} seconds of summed historical generation time, not wall time for this revision. These timings do not establish an end-to-end speedup. Four best epochs equal the 400-epoch budget; training convergence is not asserted. These conditional, synthetic results do not establish universal neural superiority, equivalence, or experimental chemical identification."
    ])
    (BASE/'neural_results.tex').write_text(tex+'\n',encoding='utf-8')
    print('Wrote NEURAL_REPORT_EN.md and neural_results.tex')

if __name__=='__main__':main()
