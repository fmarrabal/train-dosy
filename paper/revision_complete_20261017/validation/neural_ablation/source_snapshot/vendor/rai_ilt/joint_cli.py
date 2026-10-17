"""MAT interchange for the experimental jointly learned profile candidate."""
import argparse,json
from pathlib import Path
import numpy as np
from scipy.io import savemat
from .io import load_problem
from .model import DecayProblem
from .joint_profiles import JointProfileConfig,solve_joint_single,solve_joint_dosy

def export_result(path,result,problem,gate):
    path=Path(path)
    if path.exists():raise FileExistsError(f'Refusing to overwrite {path}')
    payload={}
    for key in ('masses','atomic_masses','continuous_masses','density_log_rate','reconstructed',
                'atomic_spectra','continuous_spectra'):
        a=getattr(result,key);full=np.zeros((a.shape[0],problem.y.shape[1]));full[:,gate]=a;payload[key]=full
    payload.update(b=problem.b,Y=problem.y,sigma=problem.sigma,rate_bounds=problem.rate_bounds,
        rate_edges=result.rate_edges,atomic_rates=result.atomic_rates,
        continuous_profiles=result.continuous_profiles,
        continuous_coefficients=result.continuous_coefficients,
        active_columns=gate,residual=problem.y-payload['reconstructed'],
        config_json=json.dumps(result.config),diagnostics_json=json.dumps(result.diagnostics),
        candidates_json=json.dumps(result.candidates),experimental=True,
        representation='Atomic masses and positive cubic densities in log D; spectra jointly fitted',
        candidate_spectra_column_mapping='Candidate spectra use active_columns in input order')
    if problem.ppm is not None:payload['ppm']=problem.ppm
    path.parent.mkdir(parents=True,exist_ok=True)
    savemat(path,payload,do_compression=True,long_field_names=True)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=['single','dosy']);parser.add_argument('input',type=Path)
    parser.add_argument('output',type=Path);parser.add_argument('--config',type=Path)
    parser.add_argument('--noise-gate',action='store_true')
    args=parser.parse_args()
    if args.output.exists() or args.output.resolve()==args.input.resolve():raise FileExistsError('Output must be a new file')
    p=load_problem(args.input);settings=json.loads(args.config.read_text()) if args.config else {}
    settings=settings.get('candidate',settings);settings=settings.get(args.mode,settings)
    config=JointProfileConfig(**settings);gate=np.ones(p.y.shape[1],bool)
    if args.noise_gate and args.mode=='dosy':
        count=min(3,len(p.b));gate=p.y[:count].mean(axis=0)>4*np.sqrt(np.sum(p.sigma[:count]**2,axis=0))/count
    if not gate.any():raise ValueError('No columns exceed the declared noise gate')
    q=DecayProblem(p.b,p.y[:,gate],p.sigma[:,gate],p.rate_bounds,p.ppm[gate] if p.ppm is not None else None)
    r=(solve_joint_single if args.mode=='single' else solve_joint_dosy)(q,config)
    export_result(args.output,r,p,gate)
    print(json.dumps(dict(output=str(args.output.resolve()),mode=args.mode,
        locally_stationary=r.diagnostics['locally_stationary'],native_success=r.diagnostics['native_success'],experimental=True)))
    if not r.diagnostics['locally_stationary']:raise SystemExit(2)

if __name__=='__main__':main()
