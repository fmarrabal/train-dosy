"""PALMA PPXA+ in batches; equivalent Woodbury graph projection for raw K."""
import numpy as np
from scipy.linalg import cho_factor,cho_solve
from scipy.special import wrightomega

def fit(k,y,radius,weight=.1,max_iterations=5000,tolerance=1e-5):
    n,q=k.shape;p=y.shape[1]
    radii=np.broadcast_to(np.asarray(radius,float),(p,))
    factor=cho_factor(np.eye(n)+k@k.T,lower=True)
    def graph(a,b):return a+k.T@cho_solve(factor,b-k@a)
    def ball(v):
        r=v-y;return y+r*np.minimum(1,radius/np.maximum(np.linalg.norm(r,axis=0),np.finfo(float).tiny))
    def prox(v):return weight*wrightomega((v-(1-weight))/weight-1-np.log(weight))
    tmp1=np.full((q,p),1/q);tmp2=k@tmp1;x=graph(tmp1,tmp2)
    done=np.zeros(p,bool);iterations=np.full(p,max_iterations,int)
    for it in range(1,max_iterations+1):
        ids=np.flatnonzero(~done)
        if not len(ids):break
        a=tmp1[:,ids];b=tmp2[:,ids];xx=x[:,ids]
        p1=prox(a);delta=b-y[:,ids]
        rr=radii[ids]
        p2=y[:,ids]+delta*np.minimum(1,rr/np.maximum(np.linalg.norm(delta,axis=0),np.finfo(float).tiny))
        c=graph(p1,p2);e=2*c-xx
        tmp1[:,ids]+=1.99*(e-p1);tmp2[:,ids]+=1.99*(k@e-p2);x[:,ids]+=1.99*(c-xx)
        if it%10!=0 and it!=max_iterations:continue
        p1=prox(tmp1[:,ids]);delta=tmp2[:,ids]-y[:,ids]
        p2=y[:,ids]+delta*np.minimum(1,rr/np.maximum(np.linalg.norm(delta,axis=0),np.finfo(float).tiny))
        c=graph(p1,p2);e=2*c-x[:,ids]
        split=np.sqrt(np.sum((e-p1)**2,axis=0)+np.sum((k@e-p2)**2,axis=0))
        residual=np.linalg.norm(k@p1-y[:,ids],axis=0)
        logs=np.log(np.maximum(p1,np.finfo(float).tiny))
        primal=(1-weight)*p1.sum(0)+weight*np.sum(p1*logs,axis=0)
        z=tmp2[:,ids]-p2
        with np.errstate(over='ignore',invalid='ignore'):
            dual=-np.sum(y[:,ids]*z,axis=0)-rr*np.linalg.norm(z,axis=0)-weight*np.exp((-k.T@z-1)/weight).sum(0)
        gap=primal-dual;rounding=100*np.finfo(float).eps*np.maximum(1,np.maximum(abs(primal),abs(dual)))
        good=(np.maximum(0,residual-rr)<=tolerance*np.maximum(1,rr)) & (split<=tolerance*np.maximum(1,np.maximum(np.linalg.norm(x[:,ids],axis=0),np.linalg.norm(y[:,ids],axis=0)))) & np.isfinite(gap) & (gap>=-rounding) & (gap<=tolerance*np.maximum(1,abs(primal)))
        done[ids[good]]=True;iterations[ids[good]]=it
    mass=prox(tmp1);res=np.linalg.norm(k@mass-y,axis=0)
    return mass,dict(converged=bool(done.all()),column_converged=done.tolist(),column_iterations=iterations.tolist(),residual_norm=res.tolist(),radius=radii.tolist(),iterations=int(iterations.sum()),projection='Woodbury raw-kernel graph, mathematically equivalent; finite arithmetic checked separately',weight=weight,tolerance=tolerance)
