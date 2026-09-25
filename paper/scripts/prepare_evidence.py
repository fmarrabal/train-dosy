"""Reaggregate immutable results and draw manuscript figures; never refit models."""
from pathlib import Path
import argparse, csv, hashlib, json, shutil, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.io import loadmat
from scipy.special import ndtr

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT/'reproduction_inputs'
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--source',type=Path,default=DEFAULT);args=ap.parse_args()
    source=args.source/'dosy_support_v2_20260925';bench=args.source/'ilt_unified_benchmark_20260924'
    sys.path.insert(0,str(source))
    from synthetic import generate
    from io_utils import archived
    import io_utils
    io_utils.BENCH=bench
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'pdf.fonttype':42,'savefig.dpi':220})
    evidence=ROOT/'evidence';figs=ROOT/'figures'
    provenance=[]
    for name in ['metrics.csv','summary.json','protocol_frozen.json','FINAL_AUDIT.json','support.py','synthetic.py','evaluate.py','environment.json','vendor_provenance.json','README.md','display_reference.json']:
        p=source/name;shutil.copy2(p,evidence/name);provenance.append({'path':str(p.relative_to(args.source)),'sha256':sha(p),'role':'frozen support study'})
    with (source/'metrics.csv').open(encoding='utf-8',newline='') as f: rows=list(csv.DictReader(f))
    methods=['RAI','RAI_S','DOME','DOME_S']
    def aggregate(rr):
        return dict(n=len(rr),correct=sum(x['correct_rank']=='True' for x in rr),recovered=sum(x['correct_rank']=='True' and int(x['matched_components'])==int(x['r_true']) for x in rr),
          **{key:float(np.mean([float(x[key]) for x in rr])) for key in ['spectral_leakage_fraction','unmatched_mass_fraction','test_clean_rmse_sigma','w1_logD']})
    nonnull=[r for r in rows if r['partition']=='test' and r['scenario'] not in ('null','broad') and r['method'] in methods]
    stats={m:aggregate([r for r in nonnull if r['method']==m]) for m in methods}
    assert all(q['n']==36 for q in stats.values())
    for panel, selected in [('positive',nonnull),('legacy42',[r for r in rows if r['partition']=='test' and r['scenario']!='broad' and r['method'] in methods])]:
        lines=[]
        for m in methods:
            q=aggregate([r for r in selected if r['method']==m])
            lines.append(f"{m.replace('_','-')} & {q['correct']}/{q['n']} & {q['recovered']}/{q['n']} & {100*q['spectral_leakage_fraction']:.3f} & {100*q['unmatched_mass_fraction']:.3f} & {q['test_clean_rmse_sigma']:.4f} & {q['w1_logD']:.4f} \\\\")
        pre=r'\begin{tabular}{lrrrrrr}\toprule'+'\n'+r'Method & Correct order & Order and rates & Leakage & Unmatched mass & $E_U$ & $W_{1,h}$\\\midrule'+'\n'
        (ROOT/'sections'/f'table_{panel}.tex').write_text(pre+'\n'.join(lines)+'\n'+r'\bottomrule\end{tabular}'+'\n',encoding='utf-8')
    changes={}
    for m in ['RAI','DOME']:
        old=sorted([r for r in nonnull if r['method']==m],key=lambda r:r['id'])
        new=sorted([r for r in nonnull if r['method']==m+'_S'],key=lambda r:r['id'])
        assert [r['id'] for r in old]==[r['id'] for r in new]
        changes[m]={}
        for key in ['spectral_leakage_fraction','test_clean_rmse_sigma','w1_logD']:
            d=np.array([float(b[key])-float(a[key]) for a,b in zip(old,new)])
            rng=np.random.default_rng(9925);boot=d[rng.integers(len(d),size=(10000,len(d)))].mean(1)
            changes[m][key]={'delta':float(d.mean()),'bootstrap95':np.quantile(boot,[.025,.975]).tolist()}
    (evidence/'manuscript_aggregation.json').write_text(json.dumps({'positive_test':stats,'paired_changes_positive':changes,'convention':'36 signal-present atomic cases; six nulls separate; no refitting'},indent=2))
    # Scenario recovery: rank and all rates, not just the selected count.
    scenarios=['single','double','triple','close','weak','proportional']
    fig,ax=plt.subplots(figsize=(7.2,2.65),layout='constrained')
    colors=['#96afca','#176b96','#d9ae8e','#ad5a20']
    for j,m in enumerate(methods):
        vals=[aggregate([r for r in nonnull if r['method']==m and r['scenario']==s])['recovered'] for s in scenarios]
        ax.bar(np.arange(6)+(j-1.5)*.18,vals,width=.17,label=m.replace('_','-'),color=colors[j])
    ax.set(xticks=np.arange(6),xticklabels=[s.title() for s in scenarios],ylim=(0,7.6),yticks=[0,2,4,6],ylabel='Correct order and all rates (out of 6)')
    ax.legend(ncol=4,loc='upper center',frameon=False,fontsize=8);ax.spines[['top','right']].set_visible(False)
    fig.savefig(figs/'recovery.pdf');fig.savefig(figs/'recovery.png');plt.close(fig)
    # Both native output measures and the original MF cells are retained.
    def ground(z):
        ed=np.linspace(np.log(.1e-9),np.log(15e-9),257);x=np.zeros((256,int(z['mask'].sum())))
        for d,a,w in zip(z['truth_D'],z['truth_A'][:,z['mask']],z['truth_widths']):
            if w:
                s=np.diff(ndtr((ed-np.log(d))/w));s/=s.sum();x+=s[:,None]*a
            else:x[np.clip(np.searchsorted(ed,np.log(d),side='right')-1,0,255)]+=a
        return x,ed
    def saved(part,case,m):
        p=source/'results'/part/f'{case}__{m}.npz';q=np.load(p)
        return q['X'],q['edges']
    ref=json.loads((source/'display_reference.json').read_text())
    def dosy(z,items,name,vmax=None,floor=.001):
        if vmax is None:vmax=max(float((x/np.diff(e)[:,None]).max()) for label,x,e in items)
        fig=plt.figure(figsize=(8,6.3));gs=fig.add_gridspec(2,2,left=.09,right=.9,bottom=.09,top=.94,wspace=.3,hspace=.28)
        for q,(label,x,e) in enumerate(items):
            sub=gs[q//2,q%2].subgridspec(2,2,height_ratios=[1,4],width_ratios=[8,1],hspace=.02,wspace=.03)
            tr=fig.add_subplot(sub[0,0]);ax=fig.add_subplot(sub[1,0]);side=fig.add_subplot(sub[1,1],sharey=ax)
            for yy in z['Y'][z['train']][::5]:tr.plot(z['ppm'],yy,lw=.45,alpha=.85)
            tr.set_xlim(10,0);tr.axis('off');tr.set_title(label,fontsize=10,pad=4)
            den=x/np.diff(e)[:,None];full=np.full((len(x),len(z['ppm'])),np.nan);full[:,z['mask']]=den
            step=z['ppm'][1]-z['ppm'][0];pe=np.r_[z['ppm'][0]-step/2,z['ppm']+step/2]
            im=ax.pcolormesh(pe,np.exp(e)/1e-9,np.ma.masked_invalid(np.where(full/vmax>=floor,full/vmax,np.nan)),cmap='Blues',norm=LogNorm(floor,1),shading='flat',rasterized=True)
            ax.set(yscale='log',ylim=(.1,15),xlim=(10,0),xlabel='Chemical shift (ppm)',ylabel=r'$D$ ($10^{-9}$ m$^2$ s$^{-1}$)')
            ax.tick_params(labelsize=8);ax.spines[['top','right']].set_visible(False)
            side.plot(den.sum(1),np.exp((e[:-1]+e[1:])/2)/1e-9,color='#176b96',lw=.8);side.axis('off')
        ca=fig.add_axes([.93,.2,.016,.57]);cb=fig.colorbar(im,cax=ca,ticks=[.001,.01,.1,1]);cb.set_label('Cell density / common reference maximum',fontsize=8)
        cb.ax.tick_params(labelsize=8)
        fig.savefig(figs/(name+'.pdf'));fig.savefig(figs/(name+'.png'));plt.close(fig)
    z=archived('r3_snr200_rep1');x,e=ground(z);items=[('Ground truth: finite-width profiles',x,e)]
    f=bench/'results/r3_snr200_rep1_mf_auto.mat';assert sha(f)==ref['MF_source_sha256']
    mf=loadmat(f,simplify_cells=True);d=loadmat(bench/'inputs/r3_snr200_rep1.mat',simplify_cells=True)['D']*1e-9;u=np.log(d);ed=np.r_[u[0],(u[1:]+u[:-1])/2,u[-1]]
    items.append(('MF: automatic factor count',mf['X'],ed));provenance.append({'path':str(f.relative_to(args.source)),'sha256':sha(f),'role':'unaltered MF figure'})
    for m in ['RAI_S','DOME_S']:xx,ee=saved('archive',z['id'],m);items.append((m.replace('_','-')+': automatic order',xx,ee))
    dosy(z,items,'dosy_continuous',ref['common_color_maximum'])
    z=generate(18,'test');x,e=ground(z);items=[('Ground truth: three atoms',x,e)]
    for m in ['DOME','RAI_S','DOME_S']:xx,ee=saved('test',z['id'],m);items.append((m.replace('_','-'),xx,ee))
    dosy(z,items,'dosy_discrete')
    # Geometry illustration with fixed rates: residualized evidence grows with separation.
    b=np.linspace(0,2.2,32);dist=np.geomspace(.003,1,140);vnorm=[];knorm=[]
    for gap in dist:
        k=np.exp(-b[:,None]*np.array([1.,1.+gap]));v=k[:,1]-k[:,0]*(k[:,0]@k[:,1])/(k[:,0]@k[:,0]);vnorm.append(np.linalg.norm(v));knorm.append(np.linalg.norm(k[:,1]))
    fig,axs=plt.subplots(1,2,figsize=(7.3,2.7),layout='constrained')
    axs[0].loglog(dist,np.array(vnorm)/knorm,color='#176b96');axs[0].set(xlabel='Relative separation of two fixed rates',ylabel=r'$\| (I-P_{-j})k_j\|_2 / \|k_j\|_2$')
    for p in [1,9,36]:axs[1].loglog(dist,np.array(vnorm)*np.sqrt(p),label=f'{p} frequencies')
    axs[1].set(xlabel='Relative separation',ylabel=r'Mean score / $(a/\sigma)$');axs[1].legend(frameon=False,fontsize=8)
    for ax in axs:ax.spines[['top','right']].set_visible(False)
    fig.savefig(figs/'geometry.pdf');fig.savefig(figs/'geometry.png');plt.close(fig)
    (evidence/'source_hashes.json').write_text(json.dumps(provenance,indent=2))
    print(json.dumps({'positive':stats,'changes':changes,'figures':4},indent=2))
if __name__=='__main__':main()
