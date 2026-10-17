"""MATLAB interchange for the two explicitly experimental continuous solvers."""
import argparse
import json
from pathlib import Path
from scipy.io import loadmat, savemat
from .io import load_problem
from .continuous_unmixing import ContinuousConfig, solve_single_profile, solve_dosy_profiles


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=['single','dosy'])
    p.add_argument('input',type=Path)
    p.add_argument('output',type=Path)
    p.add_argument('--references',type=Path,help='Independent MAT v5 reference file; see method documentation')
    p.add_argument('--reference-source',help='Provenance of independent reference profiles')
    p.add_argument('--smoothness',type=float)
    p.add_argument('--low-rank',type=float,default=1.)
    args=p.parse_args()
    if args.output.exists():
        raise FileExistsError(f'Refusing to overwrite {args.output}')
    problem=load_problem(args.input)
    refs={}
    if args.references:
        d=loadmat(args.references,simplify_cells=True)
        import numpy as np
        if 'b' not in d or not np.array_equal(np.asarray(d['b']).reshape(-1),problem.b):
            raise ValueError('Reference b must exactly match input acquisition order and units')
        refs=dict(nuisance_profiles=d['nuisance_profiles'],nuisance_bounds=d['nuisance_bounds'],reference_source=args.reference_source)
    cfg=ContinuousConfig(smoothness=args.smoothness,low_rank=args.low_rank)
    result=(solve_single_profile if args.mode=='single' else solve_dosy_profiles)(problem,cfg,**refs)
    fields=('rate_edges','rates','masses','density_log_rate','coefficients','reconstructed',
            'residual','cancelled_signal','nuisance_amplitudes','groups')
    data={name:getattr(result,name) for name in fields}
    data.update(b=problem.b,Y=problem.y,sigma=problem.sigma,rate_bounds=problem.rate_bounds,
                diagnostics_json=json.dumps(result.diagnostics),config_json=json.dumps(result.config))
    if problem.ppm is not None:data['ppm']=problem.ppm
    args.output.parent.mkdir(parents=True,exist_ok=True)
    savemat(args.output,data,do_compression=True)
    print(json.dumps(dict(output=str(args.output.resolve()),converged=result.diagnostics['converged'],
                         method=result.diagnostics['method'],status='experimental')))
    if not result.diagnostics['converged']:
        raise SystemExit(2)


if __name__=='__main__':main()
