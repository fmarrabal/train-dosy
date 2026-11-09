"""Canonical MAT interface for ordinary-attenuation mixed-measure solvers."""
import argparse,json
from pathlib import Path
from scipy.io import savemat
from .io import load_problem
from .mixed_unmixing import MixedConfig,solve_mixed_single,solve_mixed_dosy


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=['single','dosy'])
    parser.add_argument('input',type=Path);parser.add_argument('output',type=Path)
    parser.add_argument('--measure',choices=['mixed','continuous','discrete'],default='mixed',help='Explicit physical prior, never inferred from a file name')
    parser.add_argument('--max-factors',type=int,default=3)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError(f'Refusing to overwrite {args.output}')
    problem=load_problem(args.input)
    config=MixedConfig(max_factors=args.max_factors,measure=args.measure)
    result=(solve_mixed_single if args.mode=='single' else solve_mixed_dosy)(problem,config)
    fields=('centers','median_rates','log_widths','is_continuous','skewness','spectra','rate_edges','continuous_masses','atomic_masses','density_log_rate','profile_masses','reconstructed','residual','fractions','masses')
    data={name:getattr(result,name) for name in fields}
    data.update(b=problem.b,Y=problem.y,sigma=problem.sigma,rate_bounds=problem.rate_bounds,
                diagnostics_json=json.dumps(result.diagnostics),candidates_json=json.dumps(result.candidates),config_json=json.dumps(result.config))
    if problem.ppm is not None:data['ppm']=problem.ppm
    args.output.parent.mkdir(parents=True,exist_ok=True);savemat(args.output,data,do_compression=True)
    print(json.dumps(dict(output=str(args.output.resolve()),status='experimental',method=result.diagnostics['method'],
                         factors=result.diagnostics['rank'],continuous=result.diagnostics['continuous_factors'],atomic=result.diagnostics['atomic_factors'],
                         noise_adequate=result.diagnostics['selection_noise_adequate'],measure_type_ambiguous=result.diagnostics['measure_type_ambiguous'])))
    if not result.diagnostics['selection_noise_adequate']:raise SystemExit(2)


if __name__=='__main__':main()
