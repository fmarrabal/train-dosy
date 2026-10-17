"""Read-only verification of split isolation, normalization, outputs and provenance."""
from pathlib import Path
import json,hashlib,gzip
import numpy as np
from scipy.ndimage import binary_dilation
BASE=Path(__file__).resolve().parent
SEEDS=[172901,172902,172903,172904,172905]
METHODS=['classical_shortlist']+[f'net_seed_{s}' for s in SEEDS]

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    inputs=sorted((BASE/'inputs').glob('case_*.npz'));assert len(inputs)==120
    protocols=json.loads((BASE/'protocol.json').read_text());source=BASE/'source_snapshot'
    all_training_ids=set();training_seeds=set();training_observation_hashes=set()
    for p in (source/'training').glob('trace_*.json.gz'):
        with gzip.open(p,'rt',encoding='utf-8') as f:t=json.load(f)
        all_training_ids.add(t['id']);training_seeds.add(t['seed']);training_observation_hashes.add(t['observation_sha256'])
    cases=[];output_checks=[];ids=[];test_seeds=[]
    for i,p in enumerate(inputs):
        with np.load(p,allow_pickle=False) as z:
            pid=str(z['id']);ids.append(pid);seed=int(z['seed']);test_seeds.append(seed)
            outer=z['train'];val=np.zeros(int(outer.sum()),bool);val[2:-1:4]=True
            fitting=np.flatnonzero(outer)[~val];internal_validation=np.flatnonzero(outer)[val]
            discovery=fitting[:6];sigma=float(z['sigma']);mask=z['mask'];ppm=z['ppm'];y=z['Y']
            expected=binary_dilation(y[discovery].mean(0)>4*sigma/np.sqrt(6),iterations=2)
            if str(z['scenario'])=='null':expected=(ppm>4)&(ppm<4.65)
            observation_hash=hashlib.sha256(y[outer][:,mask].tobytes()).hexdigest()
            checks=dict(index=i,id=pid,seed=seed,domain=str(z['domain']),scenario=str(z['scenario']),
                id_disjoint_training=pid not in all_training_ids,seed_disjoint_training=seed not in training_seeds,
                observation_hash_disjoint_training=observation_hash not in training_observation_hashes,
                external_rows=int(z['test'].sum()),outer_rows=int(outer.sum()),inner_fitting_rows=len(fitting),inner_validation_rows=len(internal_validation),
                discovery_avoids_internal_external=not bool(set(discovery)&(set(internal_validation)|set(np.flatnonzero(z['test'])))),
                discovery_rows_exact=np.array_equal(z['discovery_rows'],discovery),mask_exact=np.array_equal(mask,expected))
            checks['passed']=all(checks[k] for k in ['id_disjoint_training','seed_disjoint_training','observation_hash_disjoint_training','discovery_avoids_internal_external','discovery_rows_exact','mask_exact']) and (len(fitting),len(internal_validation),int(z['test'].sum()))==(18,6,8)
            cases.append(checks)
            expected_scale=np.maximum(np.max(np.abs(y[fitting][:,mask]),axis=0),8*sigma)
            for method in METHODS:
                result=json.loads((BASE/'results'/f'case_{i:03d}__{method}.json').read_text());r=result['metrics']
                if r.get('error'):
                    output_checks.append(dict(index=i,method=method,error=r['error'],passed=False));continue
                a=BASE/'arrays'/f'case_{i:03d}__{method}.npz'
                with np.load(a,allow_pickle=False) as arr:
                    check=dict(index=i,method=method,input_hash_exact=result['input_sha256']==sha(p),array_hash_exact=result['array_sha256']==sha(a),
                        observation_hash_exact=r['observation_sha256']==observation_hash,
                        solver_scale_uses_only18fitting_rows=np.array_equal(np.asarray(result['diagnostics']['training_scale']),expected_scale),
                        finite_nonnegative_masses=bool(np.isfinite(arr['X']).all() and np.min(arr['X'])>=0),
                        shapes_correct=arr['X'].shape==(256,int(mask.sum())) and arr['prediction'].shape==(32,int(mask.sum())),
                        finite_predictions=bool(np.isfinite(arr['prediction']).all()))
                    check['passed']=all(v for k,v in check.items() if k not in ['index','method'])
                    output_checks.append(check)
    expected_seeds=list(range(202610180000,202610180120))
    final=dict(n_test_problems=len(cases),n_training_problems=len(all_training_ids),test_ids_unique=len(set(ids))==120,
        seeds_exact=test_seeds==expected_seeds,n_output_checks=len(output_checks),all_case_checks_passed=all(c['passed'] for c in cases),
        all_output_checks_passed=all(c['passed'] for c in output_checks),cases=cases,output_checks=output_checks,
        snapshot_hashes_unchanged=all(sha(source/Path(p['file'].replace(chr(92),'/')))==p['sha256'] for p in json.loads((source/'manifest.json').read_text())['files']))
    (BASE/'complete_input_audit.json').write_text(json.dumps(final,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in final.items() if k not in ['cases','output_checks']},indent=2))

if __name__=='__main__':main()
