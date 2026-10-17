from pathlib import Path
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import sys,json
import numpy as np
from threadpoolctl import threadpool_limits
import run_varian as runner
exp=runner.exp;HERE=runner.HERE;METHODS=runner.METHODS
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.ticker import FixedLocator, FixedFormatter, NullFormatter

def main():
    inp=np.load(HERE/'input.npz');meta=json.loads((HERE/'preprocessing.json').read_text());records=[]
    test=inp['outer_test'];train=~test;b=inp['b'];y=inp['Y'];ppm=inp['ppm']
    for method in METHODS:
        r=json.loads((HERE/'results'/f'{method}.json').read_text());assert r['status']=='completed'
        d=np.load(HERE/'results'/f'{method}.npz');pred=np.exp(-b[:,None]*d['D'])@d['A']
        assert np.allclose(pred,d['prediction'],atol=1e-10,rtol=1e-10)
        nrmse=np.sqrt(np.mean((pred[test]-y[test])**2))/np.sqrt(np.mean(y[train]**2))
        assert np.isclose(nrmse,r['test_nrmse'],rtol=1e-10,atol=1e-12)
        assert r['source_input_sha256']==exp.sha(HERE/'input.npz')
        records.append({k:v for k,v in r.items() if k not in ['diagnostics','local_solver_calls']})
    discovery=np.zeros(len(b),bool);discovery[np.array(meta['discovery_positions_one_based'])-1]=True
    assert not discovery[test].any()
    assert not discovery[inp['inner_validation_rows']].any()
    mask,sigma,details=exp.detect(inp['full_Y'],b,inp['full_ppm'],discovery)
    assert np.array_equal(np.flatnonzero(mask),inp['indices'])
    assert np.isclose(sigma,float(inp['sigma']),rtol=1e-12)
    # Preserve archived metadata keys; normalize their Windows separators only
    # when resolving the audit path on the current operating system.
    for p,sha in meta['source_hashes'].items():
        assert exp.sha(HERE.parent/Path(p.replace('\\','/')))==sha
    rai=np.load(HERE/'results/RAI-S.npz');dome=np.load(HERE/'results/DOME-S.npz')
    agreement=float(np.linalg.norm(rai['prediction']-dome['prediction'])/np.linalg.norm(rai['prediction']))
    summary=dict(status='completed',dataset='PropPentSuc_DOSY.fid',independent_experiments=1,
        records=records,RAI_DOME_relative_prediction_difference=agreement,
        preprocessing=meta,all_five_prediction_replays_passed=True,discovery_mask_replay_passed=True,
        source_hash_replay_passed=True,known_global_order=False,independent_D_truth=False,
        conclusion='Independent public experimental predictive illustration. Signed SCORE predicts best here; RAI-S and DOME-S agree, reach the four-atom cap and flag mismatch. No superiority or chemically correct order claim.')
    exp.save_json(HERE/'summary.json',summary)
    # Every estimated spectral channel is drawn. Original full-spectral traces
    # above maps are decimated for display only after all fits are frozen.
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,'axes.titlesize':12.5})
    fig=plt.figure(figsize=(8.3,10.4));outer=fig.add_gridspec(3,2,left=.10,right=.98,top=.925,bottom=.055,wspace=.30,hspace=.30)
    for j,(method,r) in enumerate(zip(METHODS,records)):
        gs=outer[j//2,j%2].subgridspec(2,1,height_ratios=[1,3.8],hspace=.08)
        top=fig.add_subplot(gs[0]);ax=fig.add_subplot(gs[1],sharex=top)
        for row in range(len(b)):top.plot(inp['full_ppm'][::4],inp['full_Y'][row,::4],lw=.4,alpha=.8)
        top.set_title(method);top.set_yticks([]);top.tick_params(labelbottom=False)
        fullmask=np.zeros(len(inp['full_ppm']),bool);fullmask[inp['indices']]=True
        edges=np.diff(np.r_[False,~fullmask,False].astype(int))
        for a,z in zip(np.flatnonzero(edges==1),np.flatnonzero(edges==-1)):
            ax.axvspan(inp['full_ppm'][a],inp['full_ppm'][z-1],color='#e8e8e8',alpha=.6,lw=0,zorder=-5)
        d=np.load(HERE/'results'/f'{method}.npz');A=d['A'];D=d['D']*1e9;vmax=abs(A).max()
        for i,rate in enumerate(D):
            live=abs(A[i])>vmax*.002
            ax.scatter(ppm[live],np.full(live.sum(),rate),c=np.log10(abs(A[i,live])/vmax),cmap='viridis',norm=Normalize(-2.7,0),s=3,marker='_',rasterized=True)
            negative=live&(A[i]<0)
            if negative.any():ax.scatter(ppm[negative],np.full(negative.sum(),rate),c='#bd3d61',s=2,marker='_',rasterized=True)
        ax.set_yscale('log');ax.set_ylim(.1,3);ax.set_xlim(7.2,-.5);ax.grid(axis='y',alpha=.12)
        ax.yaxis.set_major_locator(FixedLocator([.1,.3,1,3]));ax.yaxis.set_major_formatter(FixedFormatter(['0.1','0.3','1','3']));ax.yaxis.set_minor_formatter(NullFormatter())
        ax.set_xlabel('Chemical shift (ppm)');ax.set_ylabel(r'$D$ ($10^{-9}$ m$^2$ s$^{-1}$)')
        flag='\nCap + mismatch' if method in ['RAI-S','DOME-S'] else ('\nLocal atoms; not species' if method=='LOCODOSY-SCORE' else '')
        ax.text(.03,.96,f"Held-out NRMSE {100*r['test_nrmse']:.3f}%"+flag,transform=ax.transAxes,va='top',fontsize=12,bbox=dict(facecolor='white',edgecolor='none',alpha=.85,pad=2))
    legend=fig.add_subplot(outer[2,1]);legend.axis('off')
    legend.text(0,.97,'Same observations and split',weight='bold',fontsize=12,va='top',transform=legend.transAxes)
    legend.text(0,.82,'30 acquisitions; 7 held out.\n4,074 signal channels fitted.\n\nPink: negative amplitudes.\nGrey: unestimated channels.\n\nNominal diffusion scale.\nNo chemical labels assigned.',fontsize=12,va='top',linespacing=1.08,transform=legend.transAxes)
    cbax=legend.inset_axes([0,.02,.90,.055]);cb=fig.colorbar(ScalarMappable(norm=Normalize(-2.7,0),cmap='viridis'),cax=cbax,orientation='horizontal',ticks=[-2,-1,0]);cbax.set_title(r'$\log_{10}(|A|/\max|A|)$ per method',fontsize=12,pad=4)
    fig.suptitle('Public DOSY Toolbox experiment\nPropPentSuc_DOSY.fid (30 acquisitions)',fontsize=12.5,y=.99)
    (HERE/'figures').mkdir(exist_ok=True)
    for ext in ['png','pdf']:fig.savefig(HERE/'figures'/f'public_mixture_comparison.{ext}',dpi=220,bbox_inches='tight')
    plt.close(fig)
    rows='\n'.join(f"{r['method']} & {100*r['test_nrmse']:.3f} & {r['selected_rank']}{'$^{*}$' if r['method']=='LOCODOSY-SCORE' else ''} & {'local' if r['numerical_success'] is None else ('yes' if r['numerical_success'] else 'no')} \\\\" for r in records)
    tex=r'''\subsection{A public multicomponent DOSY example}\label{app:publicmixture}
We additionally processed the public \texttt{PropPentSuc\_DOSY.fid} example
distributed by the official DOSY Toolbox downloads page\cite{dosytoolboxdata}.
Its internal sample text documents propanol and sucrose in D$_2$O; its directory
name is consistent with the 1-propanol/1-pentanol/sucrose example illustrated
in the GNAT manual\cite{gnatmanual}. The latter does not explicitly identify
this archived file, so pentanol is not treated as a verified additional label.
Neither a certified complete composition nor independently measured diffusion
coefficients accompany the example used here. It therefore supplies a public
multicomponent experimental check, not known-order or absolute-$D$ ground truth.

The archive contains 30 acquired gradient levels. Acquisitions
4, 8, 12, 16, 20, 24 and 28 were reserved for external prediction;
five of the remaining acquisitions formed the internal validation split.
A single common linear ACME phase\cite{acme} was estimated from the other
18 discovery acquisitions after Fourier transformation and then frozen for
all methods and acquisitions. The archived phase alone had left visible
dispersion in discovery-only quality control. We applied no baseline
subtraction, absolute-value rectification or spectral alignment.
All 4,074 original spectral channels within the discovery-detected positive
absorption support were retained from the 32,768-point spectrum. Channels
outside that support remain unestimated. The common order cap was four.

Following the GNAT Varian import convention, the nominal encoding was
\begin{equation}
b_i=\gamma^2 T_{\mathrm{enc}}
 \bigl(g_{i,\mathrm{DAC}}c_{\mathrm{DAC}}\,10^{-2}\bigr)^2,
\end{equation}
where $T_{\mathrm{enc}}$ is the archived \texttt{dosytimecubed} parameter and
$c_{\mathrm{DAC}}$ is the archived conversion to G\,cm$^{-1}$.
The same homogeneous-gradient kernel was used for every method.
A separate generic gradient-calibration file in the download was retained
as provenance but not imposed without an exact acquisition-calibration link.
Nonuniform gradients and other unmodelled acquisition effects therefore
remain possible explanations for residual curvature; the nominal $D$ scale
is not an independently calibrated accuracy claim.

\begin{table}[ht]
\caption{One public experimental dataset, shared support and acquisition
holdout. Numerical convergence and chemical identification are distinct.}
\label{tab:publicmixturerevision}
\centering\small
\begin{tabular}{lrrl}\toprule
Method & Test NRMSE (\%) & Atoms & Converged \\ \midrule
__ROWS__
\bottomrule\end{tabular}
\par\smallskip\footnotesize $^{*}$Sum of local contributions, not a global
molecular count. All local solver calls converged.
\end{table}

RAI-S and DOME-S returned effectively identical predictions (relative
difference __AGREEMENT__), both reaching the cap and activating the
atomic-model mismatch diagnostic. The positive-amplitude SCORE variant
agreed closely with them. Signed SCORE had the lowest prediction error;
its negative-amplitude fraction was 9.54\% of total absolute amplitude.
Unlike the unstable signed fits in some resin blocks, this fit converged
and was not flagged by the declared kernel-conditioning or amplitude-expansion
checks. Thus its predictive advantage cannot be dismissed as numerical failure,
although a signed decomposition does not establish nonnegative constituent
spectra. LOCODOSY also predicted better than the positive global fits.
These results do not demonstrate superiority of RAI-S or DOME-S over the
references, or an independent benefit of their proposal mechanisms.
The single archived experiment, correlated processed noise
(adjacent-background correlation 0.983), unknown full composition and nominal
gradient calibration preclude a population-level detection claim or certified
chemical assignments. Solvent/reference resonances were not silently removed,
and an atomic row is not equated with a molecule.

\begin{figure}[ht]
\centering\includegraphics[width=\textwidth]{figures/public_mixture_comparison.pdf}
\caption{Public DOSY Toolbox example. Measured spectral families appear above
the training-only fitted atomic maps. Every selected spectral channel is fitted;
only the display of the upper traces is decimated. Pink denotes signed negative
amplitudes, grey denotes unestimated channels, and the 0.2\% relative display
cutoff does not prune the fit. Atomic rows have no inferred diffusion width or
assigned chemical identity.}
\label{fig:publicmixturerevision}
\end{figure}
'''.replace('__ROWS__',rows).replace('__AGREEMENT__',r'$7.6\times10^{-11}$')
    (HERE.parent/'public_mixture_results.tex').write_text(tex,encoding='utf-8')
    (HERE/'README.txt').write_text('''PUBLIC EXPERIMENTAL PREDICTION CHECK - not certified chemical ground truth

Source: official DOSY Toolbox datasets_v09.zip, URL/hash in ../toolbox_legacy_download.json.
Actual raw dataset: ../legacy_data/TestData/DOSY/PropPentSuc_DOSY.fid.
Standalone prepared input: input.npz (all 30 acquisitions on all 4,074 selected channels).
Original spectrum: 32,768 channels, full processed matrix included for support replay.

If raw files are absent from the package, run ../download_varian.py first.
The download checks the official archive hash and extracts only this dataset.
It is also required before ../summarize_varian.py audits the raw-source hashes.
Reproduce from raw with ../prepare_varian.py, then ../run_varian.py,
then ../summarize_varian.py, using the repository venv and nmrglue 0.11 from
../vendor_runtime. The module-only loader avoids an unrelated Tecmag/NumPy2
incompatibility; the actual Varian reader and ACME objective are unmodified.
RAI NNLS batching is the same equivalence-checked acceleration used in the
resin validation, with no objective, starting-point or solver-budget changes.

The sample text names propanol and sucrose. The directory suggests pentanol,
and GNAT manual PDF pages 74/88 illustrates propanol/pentanol/sucrose, but the
manual does not explicitly name this archived file. We do not assign the
third solute as certified composition and never supply a true global order.
Solvent and reference resonances are retained. There is no independently
known D, no mass conversion, and no NUG correction inferred from fitted peaks.
The generic nugcal400 file is retained but not treated as a matched calibration.

Preparation freezes external and internal splits before phase/support fitting.
Common ACME phase uses only 18 discovery acquisitions. Outer test is 4,8,12,
16,20,24,28; five other rows are internal validation. No test row selects phase,
mask, normalization, model order, amplitudes or diffusion rates. No channel
subsampling is used during fitting. Full top traces are decimated only for display.

All five prediction replays and source/support hashes passed. Read summary.json
for every diagnostic. Signed SCORE has the best predictive error here and is
numerically stable by the declared checks. RAI-S/DOME-S reach cap and flag
mismatch. This is one experiment, not a detection-rate validation cohort.

Unusable candidates retained separately for audit: GNAT Diffusion.zip has
8 acquisitions and was NOT fitted; the fructose/propanol dataset has 10.
The published minimum of 12 training acquisitions was never bypassed.
''',encoding='utf-8')
    files=[p for p in HERE.rglob('*') if p.is_file() and p.name!='artifact_manifest.json']
    exp.save_json(HERE/'artifact_manifest.json',{str(p.relative_to(HERE)):exp.sha(p) for p in files})
    print('PUBLIC VALIDATION COMPLETE: five prediction replays, all raw-source and support checks passed',flush=True)
    print('RAI-DOME relative prediction difference',agreement,flush=True)

if __name__=='__main__':
    with threadpool_limits(limits=1):main()
