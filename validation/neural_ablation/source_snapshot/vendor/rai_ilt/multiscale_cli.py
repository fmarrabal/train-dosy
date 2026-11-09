"""MATLAB interchange for experimental multiscale 1D and shared/private DOSY."""
import argparse,json
from pathlib import Path
import numpy as np
from scipy.io import savemat
from .io import load_problem
from .model import DecayProblem
from .multiscale_profiles import MultiscaleConfig,solve_multiscale_single,solve_multiscale_dosy

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=['single','dosy'])
    parser.add_argument('input',type=Path);parser.add_argument('output',type=Path)
    parser.add_argument('--config',type=Path,help='Multiscale settings or benchmark settings.json')
    parser.add_argument('--noise-gate',action='store_true',help='Use the benchmark observed first-three-sample threshold in DOSY')
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError(f'Refusing to overwrite {args.output}')
    p=load_problem(args.input)
    config=json.loads(args.config.read_text(encoding='utf-8')) if args.config else {}
    config=MultiscaleConfig(**config.get('multiscale',config))
    gate=np.ones(p.y.shape[1],bool)
    if args.noise_gate and args.mode=='dosy':
        # Independent observation errors: variance of the first-three average.
        count=min(3,len(p.b));sd=np.sqrt(np.sum(p.sigma[:count]**2,axis=0))/count
        gate=p.y[:count].mean(axis=0)>4*sd
    if not gate.any():raise ValueError('No spectral columns exceed the declared noise gate')
    q=DecayProblem(p.b,p.y[:,gate],p.sigma[:,gate],p.rate_bounds,ppm=p.ppm[gate] if p.ppm is not None else None)
    r=(solve_multiscale_single if args.mode=='single' else solve_multiscale_dosy)(q,config)
    payload={}
    for key in ('masses','independent_masses','shared_masses','private_masses'):
        a=np.zeros((config.bins,p.y.shape[1]));a[:,gate]=getattr(r,key);payload[key]=a
    h=np.zeros_like(p.y);h[:,gate]=r.reconstructed
    amps=np.zeros((r.anchor_amplitudes.shape[0],p.y.shape[1]));amps[:,gate]=r.anchor_amplitudes
    payload.update(b=p.b,Y=p.y,sigma=p.sigma,rate_bounds=p.rate_bounds,rate_edges=r.rate_edges,
        rates=np.sqrt(r.rate_edges[:-1]*r.rate_edges[1:]),reconstructed=h,residual=p.y-h,
        anchor_profiles=r.anchor_profiles,anchor_amplitudes=amps,active_columns=gate,
        diagnostics_json=json.dumps(r.diagnostics),experimental=True)
    if p.ppm is not None:payload['ppm']=p.ppm
    args.output.parent.mkdir(parents=True,exist_ok=True);savemat(args.output,payload,do_compression=True,long_field_names=True)
    print(json.dumps(dict(output=str(args.output.resolve()),converged=r.diagnostics['converged'],
        mode=args.mode,active_columns=int(gate.sum()),experimental=True)))
    if not r.diagnostics['converged']:raise SystemExit(2)

if __name__=='__main__':main()
