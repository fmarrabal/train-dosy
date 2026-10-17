"""Frozen-method comparison on all detected original spectral channels.

Read-only source data; no chemical truth is assigned. Acquisition holdout is
fixed before fitting. Runtime RAI NNLS batching changes no objective and is
checked against the original implementation before use.
"""
from pathlib import Path
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import sys,json,time,hashlib,argparse,importlib.util,traceback
import numpy as np
from scipy.io import loadmat
from scipy.ndimage import binary_dilation
from threadpoolctl import threadpool_limits

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[3]
BASE=Path(r'E:\ARTICULOS-CIENTIFICOS\20240905_OLIGOQUITOSANO\CODIGO_MEJORADO')
OLD=BASE/'full_spectrum_tp194_20260924'
SIGNAL=BASE/'signal_dosy_tp194_20260924'
LOCOPATH=HERE/'vendor'/'locodosy_native.py'
sys.path.insert(0,str(REPO/'src'))
from train_dosy._frozen import support
from rai_ilt import components
from dome.solver import amplitudes_exact
spec=importlib.util.spec_from_file_location('locodosy_revision',LOCOPATH)
loco=importlib.util.module_from_spec(spec);sys.modules[spec.name]=loco;spec.loader.exec_module(loco)

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def clean(x):
    if isinstance(x,np.ndarray):return clean(x.tolist())
    if isinstance(x,np.generic):return clean(x.item())
    if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [clean(v) for v in x]
    if isinstance(x,float) and not np.isfinite(x):return None
    return x
def save_json(p,x):
    Path(p).parent.mkdir(parents=True,exist_ok=True)
    Path(p).write_text(json.dumps(clean(x),indent=2)+'\n',encoding='utf-8')

def fast_profile(z,b,y,sigma):
    if not np.all(sigma==sigma[0:1,:]):
        raise ValueError('Batch-equivalent RAI requires per-column constant row weights')
    rates=np.exp(z);k=components.exponential_kernel(b,rates)
    a=amplitudes_exact(k,y)
    residual=(k@a-y)/sigma
    derivative=-np.minimum(b[:,None]*rates,745)*k
    gradient=np.sum((derivative.T@(residual/sigma))*a,axis=1)
    return .5*float(np.sum(residual**2)),gradient,a

def validate_batch():
    original=components._profile;rng=np.random.default_rng(17102026)
    errors=[]
    for rank in range(1,5):
        for noise in [0.,.02]:
            b=np.linspace(.01,5,17);z=np.log(np.geomspace(.08,3,rank))
            y=np.exp(-b[:,None]*np.exp(z))@rng.uniform(0,1,(rank,37))+noise*rng.normal(size=(17,37))
            y[:,::7]=noise*rng.normal(size=(17,len(y[:,::7][0])))
            sigma=np.broadcast_to(rng.uniform(.03,.1,(1,37)),y.shape)
            a=original(z,b,y,sigma);c=fast_profile(z,b,y,sigma)
            e=[float(np.linalg.norm(x-v)/max(1,np.linalg.norm(x))) for x,v in zip(a,c)]
            errors.append(dict(rank=rank,noise=noise,relative_errors=e))
            assert max(e)<1e-8,e
    components._profile=fast_profile
    save_json(HERE/'batch_equivalence.json',dict(fixtures=errors,passed=True,
        interpretation='Identical independent NNLS minimizers for constant row weights; no model, objective or starting-point change.',
        original_source=str(Path(components.__file__)),original_sha256=sha(components.__file__),
        batch_source='dome.solver.amplitudes_exact',script_sha256=sha(__file__)))

def regions(mask):
    delta=np.diff(np.r_[False,mask,False].astype(int))
    return np.column_stack((np.flatnonzero(delta==1),np.flatnonzero(delta==-1)))

def detect(y,b,ppm,discovery):
    rows=np.flatnonzero(discovery);rows=rows[np.argsort(b[rows])]
    anchors=rows[:4];tail=rows[-4:]
    lo=y[anchors].mean(0);hi=y[tail].mean(0)
    baseline=np.median(lo);noise=np.median(abs(lo-baseline))/.674489750196082
    noise=max(noise,np.finfo(float).eps*max(1,abs(y[rows]).max()))
    seed=lo-baseline>=8*noise;allowed=lo-baseline>=3*noise;mask=np.zeros(len(ppm),bool)
    for a,z in regions(allowed):
        if seed[a:z].any():mask[a:z]=True
    step=float(np.median(abs(np.diff(ppm))));gap=int(np.floor(.005/step))
    r=regions(mask)
    for (_,z),(a,_) in zip(r[:-1],r[1:]):
        if a-z<=gap:mask[z:a]=True
    mask=binary_dilation(mask,iterations=int(np.ceil(.004/step)))
    # Estimate single-acquisition noise using untouched background in training.
    background=~binary_dilation(mask,iterations=40)
    row_center=np.median(y[rows][:,background],axis=1)
    sigmas=np.median(abs(y[rows][:,background]-row_center[:,None]),axis=1)/.674489750196082
    sigma=float(np.median(sigmas))
    bgpairs=background[:-1]&background[1:]
    l=y[rows,:-1][:,bgpairs].ravel();rr=y[rows,1:][:,bgpairs].ravel()
    corr=float(np.corrcoef(l,rr)[0,1])
    negative=(lo-baseline<=-8*noise)
    return mask,sigma,dict(anchor_rows_zero_based=anchors,discovery_rows_zero_based=rows,
        selected_channels=int(mask.sum()),total_channels=len(mask),regions_original=regions(mask),
        low_mean_noise=noise,background_single_row_sigma=sigmas,background_channels=int(background.sum()),
        background_neighbour_correlation=corr,negative_coherent_points=int(negative.sum()),
        no_baseline_subtraction=True,no_rectification=True,no_channel_downsampling=True)

def prepare():
    (HERE/'inputs').mkdir(parents=True,exist_ok=True)
    cal=json.loads((SIGNAL/'calibration.json').read_text())
    calsrc=Path(cal['source']);assert sha(calsrc)==cal['source_sha256']
    ref=json.loads(calsrc.read_text());bad=[z['path'] for z in ref['sources'] if sha(z['path'])!=z['sha256']]
    assert not bad,bad
    sources=json.loads((OLD/'sources.json').read_text())['experimental_files']
    bad=[p for p,v in sources.items() if sha(p)!=v['sha256']]
    assert not bad,bad[:5]
    prior=json.loads((SIGNAL/'provenance.json').read_text())['unchanged_frozen_inputs_and_solver']
    provenance=dict(source_files={p:v for p,v in sources.items()},calibration=cal,
        calibration_sources=ref['sources'],verified_reference_hashes=len(ref['sources']),verified_experimental_hashes=len(sources),
        input_files={},source_code={str(LOCOPATH):sha(LOCOPATH),str(Path(support.__file__)):sha(support.__file__)},
        known_composition='Not established by source metadata; sample labels and spectral factors are not chemical ground truth.')
    cohort=json.loads((OLD/'cohort.json').read_text())
    records=[]
    for row in cohort:
        identity=row['id'];path=OLD/'inputs'/f'{identity}.mat';finger=sha(path)
        assert prior[str(path)]==finger
        d=loadmat(path,squeeze_me=True);y=d['Y'];ppm=d['ppm'];x=d['x']
        coeff=cal['factors'][identity]['D_1e9_per_kappa'];b=x/(coeff*1e-9)
        test=d['outer_test'].astype(bool);train=~test
        training_rows=np.flatnonzero(train);inner_val_local=np.zeros(train.sum(),bool);inner_val_local[2:-1:4]=True
        discovery=train.copy();discovery[training_rows[inner_val_local]]=False
        mask,sigma,meta=detect(y,b,ppm,discovery)
        scale=float(np.sqrt(np.mean(y[discovery][:,mask]**2)))
        indices=np.flatnonzero(mask);selected=y[:,mask]/scale
        pos=np.flatnonzero(np.r_[True,np.diff(indices)>1]);windows=np.column_stack((pos,np.r_[pos[1:],len(indices)]))
        np.savez_compressed(HERE/'inputs'/f'{identity}.npz',Y=selected,b=b,ppm=ppm[mask],indices=indices,
            sigma=sigma/scale,scale=scale,outer_test=test,inner_validation_rows=training_rows[inner_val_local],
            windows=windows,full_ppm=ppm,full_mean_spectrum=y[training_rows[:4]].mean(0)/scale,
            top_traces=y[:,::4]/scale,top_ppm=ppm[::4])
        rec=dict(id=identity,sample=row['sample'],block=row['block_index'],Delta_ms=1000*row['D20_s'],
            temperature_K=np.mean(row['temperatures_K']),sigma_normalized=sigma/scale,signal_scale=scale,
            input_sha256=finger,selected_input_sha256=sha(HERE/'inputs'/f'{identity}.npz'),mask=meta,
            observed_raw_negative_fraction=float(np.mean(y<0)),original_acquisition_experiments=row['experiments'])
        records.append(rec);provenance['input_files'][str(path)]=finger
        print('PREPARED',identity,mask.sum(),'sigma',sigma/scale,'rho',meta['background_neighbour_correlation'],flush=True)
    save_json(HERE/'cohort.json',records);save_json(HERE/'provenance.json',provenance)
    save_json(HERE/'protocol.json',dict(methods=['RAI-S','DOME-S','SCORE-CV','LOCODOSY-SCORE'],
        outer_test_positions_one_based=[4,8,12,16,20],same_mask_for_all_methods=True,
        mask='8-sigma seeds, 3-sigma connected growth, close gaps <=0.005 ppm, pad 0.004 ppm; four lowest discovery acquisitions only.',
        mask_excludes_outer_and_inner_validation=True,normalization='One discovery RMS scalar for the full selected spectral matrix.',
        sigma='Median background MAD single-acquisition sigma over discovery rows; correlated processed noise is diagnosed, not claimed iid.',
        bounds_D_SI=[.01e-9,15e-9],max_components=4,diffusion_display_bins=256,
        model_order_truth=None,chemical_identity_truth=None,calibration='Previously verified HDO/TBO coefficient; no TTMS or mass conversion.',
        independent_units='Four specimen series, three diffusion-time blocks each. Twelve blocks are not twelve independent mixtures.',
        preprocessing='Existing processed absorption spectra; no new baseline, phase, frequency alignment or sign correction.',
        display='Training fits with held-out acquisitions not reused. All signal channels fitted; top trace decimation is display only.',
        external_baselines='Published-method attributed GNAT translation; SCORE signed amplitudes, native initialization, common CV order; LOCODOSY native SVD local order with same signal windows.'))

def score_cv(b,y,sigma):
    val=np.zeros(len(b),bool);val[2:-1:4]=True;train=~val
    cfg=loco.Config(method='score',max_components=4,auto_reduce=False,max_evaluations=10000)
    candidates=[]
    for q in range(1,5):
        r=loco.score(b[train],y[train],q,cfg)
        pred=np.exp(-b[val,None]*r['D'])@r['amplitudes']
        candidates.append(dict(rank=q,D=r['D'],prediction=pred,sse=float(np.sum(((pred-y[val])/sigma)**2)),
            success=bool(r['success'] and np.all(r['D']>0)),evaluations=r['evaluations']))
    good=[i for i,c in enumerate(candidates) if c['success']]
    if not good:raise RuntimeError('No converged positive-rate SCORE candidate')
    best=min(good,key=lambda i:candidates[i]['sse'])
    eligible=[i for i in good if candidates[i]['sse']<=candidates[best]['sse']+2*np.linalg.norm(candidates[i]['prediction']-candidates[best]['prediction'])/sigma+1e-10]
    pick=min(eligible,key=lambda i:(candidates[i]['rank'],candidates[i]['sse']))
    out=loco.score(b,y,candidates[pick]['rank'],cfg)
    out.update(A=out.pop('amplitudes'),candidate_diagnostics=[{k:v for k,v in c.items() if k!='prediction'} for c in candidates],selected_rank=candidates[pick]['rank'])
    return out

def run(identity,method):
    target=HERE/'results'/f'{identity}__{method}.json'
    if target.exists():print('SKIP',identity,method,flush=True);return
    d=np.load(HERE/'inputs'/f'{identity}.npz');train=~d['outer_test'];test=~train
    b=d['b'];y=d['Y'];sigma=float(d['sigma']);ppm=d['ppm'];start=time.perf_counter()
    try:
        if method in ['RAI-S','DOME-S']:
            r=support.fit_support(b[train],y[train],sigma,ppm,(.01e-9,15e-9),method=method.split('-')[0],config=support.Config(max_components=4))
        elif method=='SCORE-CV':r=score_cv(b[train],y[train],sigma)
        elif method=='LOCODOSY-SCORE':
            r=loco.fit(b[train],y[train],ppm,windows=d['windows'],config=loco.Config(method='score',max_components=4,diffusion_bins=256,max_diffusion=15e-9))
            r['A']=r.pop('amplitudes')
        else:raise ValueError(method)
        D=r['D'];A=r['A'];pred=np.exp(-b[:,None]*D)@A
        error=pred[test]-y[test];train_rms=np.sqrt(np.mean(y[train]**2))
        diag={k:v for k,v in r.items() if k not in ['A','D','prediction','groups','support','train_only_native','mass_grid','legacy_DOSY','windows','window_points']}
        out=dict(id=identity,method=method,seconds=time.perf_counter()-start,status='completed',
            numerical_success=r.get('success',None),selected_rank=len(D),rank_interpretation='local contributions, not species' if method=='LOCODOSY-SCORE' else 'global atoms, not chemical truth',
            D_SI=D,A_negative_mass_ratio=float(np.maximum(-A,0).sum()/max(abs(A).sum(),1e-300)),
            test_rmse=float(np.sqrt(np.mean(error**2))),test_nrmse=float(np.sqrt(np.mean(error**2))/train_rms),
            test_rmse_over_sigma=float(np.sqrt(np.mean(error**2))/sigma),test_bias_over_sigma=float(error.mean()/sigma),
            training_nrmse=float(np.sqrt(np.mean((pred[train]-y[train])**2))/train_rms),
            points=int(y.shape[1]),diagnostics=diag)
        (HERE/'results').mkdir(exist_ok=True)
        np.savez_compressed(target.with_suffix('.npz'),D=D,A=A,prediction=pred,ppm=ppm,b=b,outer_test=test,
            support=r.get('support',np.isfinite(A)),residual_test=error)
        save_json(target,out)
        print('DONE',identity,method,'rank',len(D),'sec',round(out['seconds'],2),'nrmse',out['test_nrmse'],'success',out['numerical_success'],flush=True)
    except Exception as exc:
        save_json(target,dict(id=identity,method=method,status='failed',seconds=time.perf_counter()-start,error=repr(exc),traceback=traceback.format_exc()))
        print('FAILED',identity,method,repr(exc),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--prepare',action='store_true');ap.add_argument('--ids',nargs='*');ap.add_argument('--methods',nargs='*',default=['RAI-S','DOME-S','SCORE-CV','LOCODOSY-SCORE']);args=ap.parse_args()
    with threadpool_limits(limits=1):
        validate_batch()
        if args.prepare:prepare()
        ids=args.ids or [p.stem for p in sorted((HERE/'inputs').glob('*.npz'))]
        for identity in ids:
            for method in args.methods:run(identity,method)
