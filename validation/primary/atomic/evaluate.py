"""Only this stage reads synthetic truth; inversion functions cannot access it."""
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.special import ndtr

def grid(d,a,bins=256):
    edges=np.linspace(np.log(.1e-9),np.log(15e-9),bins+1)
    x=np.zeros((bins,a.shape[1]))
    if len(d):np.add.at(x,np.clip(np.searchsorted(edges,np.log(d),side='right')-1,0,bins-1),a)
    return edges,x

def metrics(z,out):
    mask=z['mask'];d=np.asarray(out['D']);a=np.asarray(out['A']);dt=z['truth_D'];at=z['truth_A'][:,mask]
    b=z['b']; pred=np.exp(-b[:,None]*d)@a
    obs=z['Y'][:,mask];clean=z['Y_clean'][:,mask];te=z['test'];sig=z['sigma']
    edges,x=grid(d,a);xt=np.zeros_like(x)
    for dd,aa,w in zip(dt,at,z['truth_widths']):
        if w>0:
            mass=np.diff(ndtr((edges-np.log(dd))/w));mass/=mass.sum();xt+=mass[:,None]*aa
        else:
            xt+=grid(np.array([dd]),aa[None,:])[1]
    weights=xt.sum(0);total=weights.sum()
    xp=x/np.maximum(x.sum(0),1e-300);tp=xt/np.maximum(xt.sum(0),1e-300)
    wd=np.sum(abs(np.cumsum(xp-tp,axis=0)[:-1])*np.diff(edges)[:-1,None],axis=0)
    wd[x.sum(0)<=0]=edges[-1]-edges[0]
    pairs=[]
    if len(dt) and len(d):
        ii,jj=linear_sum_assignment(abs(np.log(d[:,None]/dt)))
        pairs=[(i,j) for i,j in zip(ii,jj) if abs(np.log(d[i]/dt[j]))<=.25]
    false=float(a.sum());weak=[];leakage=0.;matched_mass=0.
    for i,j in pairs:
        meaningful=at[j]>.001*at[j].max()
        false-=float(a[i,meaningful].sum())
        leakage+=float(a[i,~meaningful].sum());matched_mass+=float(a[i].sum())
        # Recall of physical spectral mass in groups preserved by the method.
        keep=out.get('support',np.ones_like(a,bool))[i]
        weak.append((j,float(at[j,keep].sum()/max(at[j].sum(),1e-300))))
    recalls=dict(weak)
    minority=int(np.argmin(at.sum(1))) if len(dt) else None
    return dict(id=z['id'],scenario=z['scenario'],snr=z['snr'],r_true=len(dt),rank=len(d),
        correct_rank=len(d)==len(dt),matched_components=len(pairs),
        ghost_mass_fraction=max(false,0)/max(float(a.sum()),1e-300),
        spectral_leakage_fraction=leakage/max(matched_mass,1e-300),
        unmatched_mass_fraction=max(float(a.sum())-matched_mass,0)/max(float(a.sum()),1e-300),
        minority_rate_recovered=minority in recalls if minority is not None else None,
        minority_support_recall=recalls.get(minority,0.) if minority is not None else None,
        test_clean_rmse_sigma=float(np.sqrt(np.mean((pred[te]-clean[te])**2))/sig),
        test_observed_rmse_sigma=float(np.sqrt(np.mean((pred[te]-obs[te])**2))/sig),
        w1_logD=float(weights@wd/max(total,1e-300)),mass=float(a.sum()),
        success=bool(out.get('success',True)),kkt=out.get('kkt'),stationarity=out.get('stationarity'))

def map_metrics(z,x,pred):
    # Neural distribution evaluation; do not invent a chemical component count.
    mask=z['mask'];bins=x.shape[0];edges=np.linspace(np.log(.1e-9),np.log(15e-9),bins+1)
    xt=np.zeros_like(x)
    for dd,aa,w in zip(z['truth_D'],z['truth_A'][:,mask],z['truth_widths']):
        if w>0:
            mass=np.diff(ndtr((edges-np.log(dd))/w));mass/=mass.sum();xt+=mass[:,None]*aa
        else:xt+=grid(np.array([dd]),aa[None,:],bins)[1]
    weights=xt.sum(0);p=x/np.maximum(x.sum(0),1e-300);t=xt/np.maximum(weights,1e-300)
    wd=np.sum(abs(np.cumsum(p-t,axis=0)[:-1])*np.diff(edges)[:-1,None],axis=0)
    wd[x.sum(0)<=0]=edges[-1]-edges[0]
    te=z['test'];clean=z['Y_clean'][:,mask]
    return dict(id=z['id'],scenario=z['scenario'],snr=z['snr'],r_true=len(z['truth_D']),
        w1_logD=float(weights@wd/max(weights.sum(),1e-300)),
        test_clean_rmse_sigma=float(np.sqrt(np.mean((pred[te]-clean[te])**2))/z['sigma']))
