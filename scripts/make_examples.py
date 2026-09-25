"""Generate fixed synthetic examples; truth never enters fit requests."""
from pathlib import Path
import sys,json,time
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'benchmarks'))
from synthetic import generate
from train_dosy import fit
from train_dosy.inversion import clean_json,prepare

def write(path,obj):path.write_text(json.dumps(clean_json(obj),allow_nan=False,indent=2),encoding='utf-8')
summary=[]
for i,name in zip((16,17,18),('one_component','two_components','three_components')):
    z=generate(i);folder=ROOT/'examples'/name;folder.mkdir(parents=True,exist_ok=True)
    q=dict(Y=z['Y'][z['train']].tolist(),b=z['b'][z['train']].tolist(),ppm=z['ppm'].tolist(),
           sigma=z['sigma'],method='RAI-S',bins=256,diffusion_bounds=[.1e-9,15e-9])
    q['mask']=prepare(q)[4].tolist()
    write(folder/'input.json',q)
    write(folder/'truth.json',dict(seed=z['seed'],partition='development',index=i,D=z['truth_D'],
        A=z['truth_A'],ppm=z['ppm'],b_test=z['b'][z['test']],Y_test=z['Y'][z['test']],
        Y_clean_test=z['Y_clean'][z['test']],note='Only evaluation reads this file'))
    for method in ['RAI-S','DOME-S']:
        start=time.perf_counter();q['method']=method;out=fit(q)
        write(folder/(method+'.json'),out)
        pred=np.exp(-z['b'][z['test'],None]*np.array(out['D']))@np.array(out['A'])
        rmse=float(np.sqrt(np.mean((pred-z['Y_clean'][z['test']][:,q['mask']])**2))/z['sigma'])
        row=dict(example=name,method=method,true_rank=z['r_true'],selected_rank=out['selected_rank'],
            success=out['success'],kkt=out['kkt'],stationarity=out['stationarity'],
            clean_test_rmse_sigma=rmse,seconds=time.perf_counter()-start)
        summary.append(row);print(row,flush=True)
write(ROOT/'examples/summary.json',summary)
