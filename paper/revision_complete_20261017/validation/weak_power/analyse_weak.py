from pathlib import Path
import os,json
os.environ['OPENBLAS_NUM_THREADS']='1'
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).parents[2]
BASE=Path(__file__).parent
summary=json.loads((BASE/'summary.json').read_text())['rows']
levels=[.02,.04,.08,.16];snrs=[50,100,200]
fig,axes=plt.subplots(1,2,figsize=(8.5,3.5),sharex=True,sharey=True,layout='constrained')
for ax,method in zip(axes,['RAI','DOME']):
    hit=np.zeros((4,3),int)
    for q in summary:
        if q['method']==method:hit[levels.index(q['minority_amplitude_factor']),snrs.index(q['snr'])]=q['recovered']
    im=ax.imshow(hit/8,origin='lower',vmin=0,vmax=1,cmap='Blues',aspect='auto')
    for i in range(4):
        for j in range(3):ax.text(j,i,f'{hit[i,j]}/8',ha='center',va='center',color='white' if hit[i,j]>=5 else '#161616',fontsize=11)
    ax.set_xticks(range(3),snrs);ax.set_yticks(range(4),[f'{v:.2f}' for v in levels])
    ax.set_xlabel('SNR');ax.set_title(method+'-S',fontsize=12)
axes[0].set_ylabel('Minority amplitude factor')
cbar=fig.colorbar(im,ax=axes,label='Minority rate matched (fraction)',shrink=.9)
path=ROOT/'manuscript/figures/weak_power_revision.pdf'
fig.savefig(path);fig.savefig(BASE/'weak_power_revision.png',dpi=150);plt.close(fig)
text=r'''\subsection{Adaptive weak-component recovery}\label{app:weakpower}
Ninety-six new problems span four generating minority-amplitude factors
$(0.02,0.04,0.08,0.16)$, three SNRs $(50,100,200)$ and eight independent
truth/noise realizations per cell. Each factor multiplies the third spectral
template before the total unattenuated signal is normalized; it is not a
molar fraction. The mask is formed from fitting acquisitions only, excluding
both internal validation and external test rows. The frozen adaptive selector
and rate-matching tolerance remain unchanged.

\begin{figure}[htbp]\centering
\includegraphics[width=\linewidth]{figures/weak_power_revision.pdf}
\caption{New weak-component panel. Each cell counts matched minority rates in
eight independent problems, using the original log-rate tolerance 0.25.
The methods use paired inputs. These counts estimate recovery under the
declared generator, not a calibrated probability of chemical presence.}
\label{fig:weakpowerrevision}\end{figure}

All 192 method fits pass their numerical criteria. RAI-S matches the minority
rate in 45 of 96 problems and DOME-S in 44 of 96; these are paired method
results, not 192 independent chemical mixtures. At factor 0.02 the counts
across increasing SNR are $(0,0,1)/8$ for both methods. At factor 0.08 they
are $(1,8,8)/8$. The intermediate factor 0.04 gives $(0,1,4)/8$ for RAI-S
and $(0,0,4)/8$ for DOME-S. These visible failures prevent a general claim
of reliable weak-component discovery. Masking also contributes: mean retained
minority spectral mass varies from about 0.59 at the weakest low-SNR cell to
0.99 at the strongest high-SNR cell. A channel outside the mask is unestimated.

Exact two-sided 95\% Clopper--Pearson intervals are retained per cell in the
machine-readable supplement; for example, zero of eight gives an upper
endpoint 0.369 and eight of eight a lower endpoint 0.631. They are descriptive
cell-wise intervals without multiplicity adjustment. Aggregate counts across
the prespecified SNR/factor mixture are not a universal power guarantee.
'''
(BASE/'weak_results.tex').write_text(text,encoding='utf-8')
print('Weak figure and manuscript fragment ready.')
