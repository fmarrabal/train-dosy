"""Positive local Hausdorff moments, bounded Laplace approximation and quadrature.

Classical moment constraints and quadrature are used explicitly; no claim of
novelty or statistical identification follows from numerical feasibility.
"""
from dataclasses import dataclass
import numpy as np
from scipy import sparse
from scipy.special import gammaln
from .model import real_array


@dataclass(frozen=True)
class MomentRegion:
    lower: float
    upper: float
    degree: int

    def __post_init__(self):
        if not np.isfinite([self.lower, self.upper]).all() or not 0 < self.lower < self.upper:
            raise ValueError("Moment region needs increasing finite positive bounds")
        if isinstance(self.degree, bool) or not isinstance(self.degree, (int, np.integer)) or self.degree < 2 or self.degree % 2:
            raise ValueError("Moment degree must be an even integer >= 2")

    @property
    def center(self):
        return self.lower + (self.upper-self.lower)/2

    @property
    def halfwidth(self):
        return (self.upper-self.lower)/2


def hankel(m, size, shift=0):
    index = np.arange(size)
    return np.asarray(m)[index[:, None]+index[None, :]+shift]


def hankel_adjoint(matrix, degree):
    out = np.zeros(degree+1)
    for i in range(len(matrix)):
        for j in range(len(matrix)):
            out[i+j] += matrix[i, j]
    return out


def moment_sequence(u, amplitudes, degree):
    u = real_array(u, "local nodes").reshape(-1)
    a = real_array(amplitudes, "amplitudes")
    if a.ndim == 1:
        a = a[:, None]
    if a.ndim != 2 or a.shape[0] != len(u) or np.any(a < 0) or np.any(np.abs(u) > 1+1e-12):
        raise ValueError("Need nonnegative amplitudes on nodes in [-1,1]")
    return u[None, :]**np.arange(degree+1)[:, None] @ a


def transform_moments(m, source, target):
    """Change coordinates, without inventing unknown higher moments."""
    from math import comb
    if target.degree > source.degree:
        raise ValueError("Cannot infer higher moments by a coordinate change")
    if target.lower > source.lower or target.upper < source.upper:
        raise ValueError("The new interval must contain the old interval")
    a = (source.center-target.center)/target.halfwidth
    b = source.halfwidth/target.halfwidth
    return np.array([sum(comb(k,j)*a**(k-j)*b**j*m[j] for j in range(k+1))
                     for k in range(target.degree+1)])


def region_kernel(b, region):
    b = np.asarray(b)
    product = b*region.center
    if not np.isfinite(product).all():
        raise ValueError("Acquisition-rate products overflow; check physical units")
    columns = [np.exp(-product)]
    for k in range(1, region.degree+1):
        columns.append(columns[-1]*(-b*region.halfwidth)/k)
    return np.column_stack(columns)


def remainder_bound(b, region):
    """Absolute kernel error per unit mass, using a Taylor remainder bound."""
    b = np.asarray(b)
    bound = np.zeros_like(b, dtype=float)
    positive = b > 0
    x = b[positive]*region.halfwidth
    log_bound = -b[positive]*region.lower + (region.degree+1)*np.log(x)-gammaln(region.degree+2)
    bound[positive] = np.exp(np.minimum(log_bound, 700))
    return bound


class MomentSpace:
    def __init__(self, regions):
        self.regions = tuple(regions)
        if not self.regions:
            raise ValueError("Need at least one region")
        if any(a.upper != b.lower for a,b in zip(self.regions, self.regions[1:])):
            raise ValueError("Moment regions must be contiguous and ordered")
        self.offsets = np.r_[0, np.cumsum([r.degree+1 for r in regions])].astype(int)
        self.size = int(self.offsets[-1])
        self.mass_indices = self.offsets[:-1]

    def slices(self):
        return [slice(a,b) for a,b in zip(self.offsets[:-1], self.offsets[1:])]

    def kernel(self, b):
        return np.hstack([region_kernel(b,r) for r in self.regions])

    def bounds(self, b):
        return np.column_stack([remainder_bound(b,r) for r in self.regions])

    def from_measure(self, rates, amplitudes):
        rates = np.asarray(rates)
        a = np.asarray(amplitudes)
        if a.ndim == 1:
            a = a[:,None]
        if a.shape[0] != len(rates) or np.any(rates < self.regions[0].lower) or np.any(rates > self.regions[-1].upper):
            raise ValueError("Measure outside the declared moment domain")
        out=[]
        for i,r in enumerate(self.regions):
            selected=(rates>=r.lower)&((rates<r.upper) if i<len(self.regions)-1 else (rates<=r.upper))
            out.append(moment_sequence((rates[selected]-r.center)/r.halfwidth, a[selected], r.degree))
        return np.vstack(out)

    def cone_map(self):
        """Map moments to scaled upper-triangular H and localizing G matrices."""
        rr, cc, vv, sizes, row = [], [], [], [], 0
        for offset,r in zip(self.offsets[:-1],self.regions):
            for localizing,size in ((False,r.degree//2+1),(True,r.degree//2)):
                sizes.append(size)
                for j in range(size):
                    for i in range(j+1):
                        factor=1. if i==j else np.sqrt(2)
                        rr.append(row); cc.append(offset+i+j); vv.append(factor)
                        if localizing:
                            rr.append(row); cc.append(offset+i+j+2); vv.append(-factor)
                        row+=1
        return sparse.csc_matrix((vv,(rr,cc)),shape=(row,self.size)),sizes

    def feasibility(self, moments, caps):
        minimum=0.
        for sl,r in zip(self.slices(),self.regions):
            for m in moments[sl].T:
                s=r.degree//2
                minimum=min(minimum,float(np.linalg.eigvalsh(hankel(m,s+1)).min()),
                            float(np.linalg.eigvalsh(hankel(m,s)-hankel(m,s,2)).min()))
        mass=moments[self.mass_indices].sum(axis=0)
        return dict(min_cone_eigenvalue=minimum, mass_cap_violation=float(np.maximum(mass-caps,0).max()))

    def metadata(self):
        return [dict(lower=r.lower,upper=r.upper,degree=r.degree) for r in self.regions]


def adaptive_space(b, bounds, sigma, caps, config):
    """Increase local order, then subdivide, to meet a declared noise-relative bound.

    Uses training acquisitions/noise only. The common mass cap bounds the
    sum of regional masses, so the worst regional kernel bound suffices.
    """
    edges=np.geomspace(*bounds,config.initial_regions+1)
    regions=[MomentRegion(float(a),float(z),config.min_degree) for a,z in zip(edges[:-1],edges[1:])]
    log=[]
    i=0
    while i<len(regions):
        r=regions[i]
        error=float(np.max(remainder_bound(b,r)[:,None]*caps/sigma))
        if error<=config.model_error_budget:
            i+=1; continue
        if r.degree<config.max_degree:
            regions[i]=MomentRegion(r.lower,r.upper,r.degree+2)
            log.append(dict(operation="increase_order",region=i,degree=r.degree+2,bound_before=error))
        elif len(regions)<config.max_regions:
            mid=float(np.sqrt(r.lower)*np.sqrt(r.upper))
            regions[i:i+1]=[MomentRegion(r.lower,mid,config.min_degree),MomentRegion(mid,r.upper,config.min_degree)]
            log.append(dict(operation="split_for_kernel_accuracy",region=i,bound_before=error))
        else:
            raise ValueError("Moment representation budget exhausted before controlling kernel error")
    return MomentSpace(regions),log


def radau_measure(m, *, tolerance=1e-8):
    """Positive right-Radau quadrature of a finite even moment sequence.

    It is an integration representation, not a chemical-component declaration.
    Return the actual moment mismatch after numerical rank truncation.
    """
    m=np.asarray(m,dtype=float)
    if len(m)%2!=1 or len(m)<3 or not np.isfinite(m).all():
        raise ValueError("Need finite moments of even degree >=2")
    if m[0]<=1e-12:
        return np.empty(0),np.empty(0),float(np.max(np.abs(m)))
    s=(len(m)-1)//2
    transformed=m[:-1]-m[1:]
    h0=hankel(transformed,s)
    h1=hankel(transformed,s,1)
    eigen,vectors=np.linalg.eigh(h0)
    keep=eigen>tolerance*max(float(eigen[-1]),m[0]*1e-6)
    if not np.any(keep):
        nodes,weights=np.array([1.]),np.array([m[0]])
    else:
        basis=vectors[:,keep]/np.sqrt(eigen[keep])
        multiplication=basis.T@h1@basis
        nodes,v=np.linalg.eigh((multiplication+multiplication.T)/2)
        constant=np.sqrt(eigen[keep])*vectors[0,keep]
        transformed_weights=(v.T@constant)**2
        if np.min(nodes)<-1-1e-4 or np.max(nodes)>1+1e-4:
            raise ValueError("Moment quadrature contains nodes outside its interval")
        nodes=np.clip(nodes,-1,1)
        away=(1-nodes)>1e-10
        weights=transformed_weights[away]/(1-nodes[away])
        nodes=nodes[away]
        endpoint=m[0]-weights.sum()
        # Enforce positive mass conservation only within a numerical tolerance.
        if endpoint < -1e-4*m[0]:
            raise ValueError("Negative Radau endpoint mass")
        if endpoint<0:
            weights*=m[0]/weights.sum()
        elif endpoint>0:
            nodes=np.r_[nodes,1.]; weights=np.r_[weights,endpoint]
    mismatch=float(np.max(np.abs(moment_sequence(nodes,weights,len(m)-1)[:,0]-m)))
    return nodes,weights,mismatch
