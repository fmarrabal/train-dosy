"""MAT interface for strict discrete RAI; ordinary attenuation, no baseline."""
from pathlib import Path
import argparse,json
from scipy.io import savemat
from .io import load_problem
from .discrete import DiscreteConfig,solve_discrete

def export_result(path,result,problem):
    path=Path(path)
    if path.exists():raise FileExistsError(path)
    payload=dict(b=problem.b,Y=problem.y,sigma=problem.sigma,rate_bounds=problem.rate_bounds,
        D=result.rates,amplitudes=result.amplitudes,reconstructed=result.reconstructed,
        rate_edges=result.rate_edges,masses=result.masses,
        diagnostics_json=json.dumps(result.diagnostics),candidates_json=json.dumps(result.candidates),
        config_json=json.dumps(result.config),experimental=True,
        representation='Finite point amplitudes; masses are display bins, not densities')
    if problem.ppm is not None:payload['ppm']=problem.ppm
    path.parent.mkdir(parents=True,exist_ok=True)
    savemat(path,payload,do_compression=True,long_field_names=True)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input',type=Path);parser.add_argument('output',type=Path)
    parser.add_argument('--config',type=Path)
    parser.add_argument('--known-components',type=int,help='Use only when independently known; otherwise omit')
    args=parser.parse_args()
    if args.output.exists() or args.output.resolve()==args.input.resolve():raise FileExistsError(args.output)
    config=DiscreteConfig(**json.loads(args.config.read_text())) if args.config else DiscreteConfig()
    p=load_problem(args.input);r=solve_discrete(p,config,args.known_components)
    export_result(args.output,r,p)
    print(json.dumps(r.diagnostics))
    if not r.diagnostics['locally_stationary'] or not r.diagnostics['selection_valid']:raise SystemExit(2)

if __name__=='__main__':main()
