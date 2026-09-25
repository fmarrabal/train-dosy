"""Refit the 48-case atomic protocol; frozen selection constants."""
from pathlib import Path
import argparse,json,csv
import numpy as np
from synthetic import generate
from evaluate import metrics
from train_dosy._frozen.support import fit_support,native_path
from train_dosy.inversion import clean_json
p=argparse.ArgumentParser();p.add_argument('--count',type=int,default=48);p.add_argument('--output',type=Path,default=Path('benchmark_run'))
a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
rows=[]
for i in range(a.count):
    z=generate(i,'test');tr=z['train'];m=z['mask']
    for method in ['RAI','RAI_S','DOME','DOME_S']:
        if method.endswith('_S'):out=fit_support(z['b'][tr],z['Y'][tr][:,m],z['sigma'],z['ppm'][m],method=method[:-2])
        else:_,out=native_path(z['b'][tr],z['Y'][tr][:,m],z['sigma'],(.1e-9,15e-9),method,4)
        row=metrics(z,out);row.update(method=method,partition='test');rows.append(row)
        (a.output/f"{z['id']}__{method}.json").write_text(json.dumps(clean_json(out),allow_nan=False),encoding='utf-8')
        print(z['id'],method,row['rank'],flush=True)
with (a.output/'metrics.csv').open('w',newline='',encoding='utf-8') as f:
    w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
