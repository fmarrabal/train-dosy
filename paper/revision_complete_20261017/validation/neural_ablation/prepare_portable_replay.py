"""Create a fresh, portable neural-study run without modifying the recorded study.

The frozen runner receives two path-only substitutions: its archive root and an
unused repository-depth assignment. Numerical functions and settings are untouched.
No numerical function, hyperparameter, seed, split or inference rule is changed.
"""
from pathlib import Path
import argparse,hashlib,json,shutil

BASE=Path(__file__).resolve().parent
ORIGINAL=Path('E:/ARTICULOS-CIENTIFICOS/20240905_OLIGOQUITOSANO/CODIGO_MEJORADO/dosy_support_v2_20260925')

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def snapshot():
    out=BASE/'source_snapshot';files=list((ORIGINAL/'vendor/rai_ilt').glob('*.py'))
    files += [ORIGINAL/'evaluate.py',ORIGINAL/'synthetic.py',ORIGINAL/'vendor_provenance.json']
    files += list((ORIGINAL/'training').glob('trace_*.json.gz'))
    manifest=[]
    for p in files:
        q=out/p.relative_to(ORIGINAL);q.parent.mkdir(parents=True,exist_ok=True)
        if q.exists():assert sha(q)==sha(p)
        else:shutil.copy2(p,q)
        manifest.append(dict(file=str(q.relative_to(out)),sha256=sha(q),original_sha256=sha(p)))
    (out/'manifest.json').write_text(json.dumps(dict(files=manifest,numerical_sources_unmodified=True),indent=2),encoding='utf-8')
    print('snapshot_files',len(files))

def prepare(dest):
    dest=Path(dest).resolve()
    if dest.exists() and any(dest.iterdir()):raise FileExistsError('Replay destination must be absent or empty; existing study will not be overwritten')
    assert (BASE/'source_snapshot/manifest.json').exists(),'Create/archive the source snapshot first'
    dest.mkdir(parents=True,exist_ok=True);shutil.copytree(BASE/'source_snapshot',dest/'source_snapshot')
    source=(BASE/'neural_study.py').read_text(encoding='utf-8')
    original_line="ARCHIVE=Path('E:/ARTICULOS-CIENTIFICOS/20240905_OLIGOQUITOSANO/CODIGO_MEJORADO/dosy_support_v2_20260925')"
    replacement="ARCHIVE=Path(__file__).resolve().parent/'source_snapshot'"
    assert source.count(original_line)==1
    depth_line='REPO=OUT.parents[3]'
    depth_replacement='REPO=OUT  # Unused in the runner; portable destinations may be shallow.'
    assert source.count(depth_line)==1
    portable=source.replace(original_line,replacement).replace(depth_line,depth_replacement)
    (dest/'neural_study.py').write_text(portable,encoding='utf-8')
    helpers=['analyse_study.py','check_reproducibility.py','extra_neural_diagnostics.py',
        'audit_complete_inputs.py','render_neural_report.py']
    for name in helpers:shutil.copy2(BASE/name,dest/name)
    shutil.copy2(BASE.parents[1]/'requirements_revision_neural.txt',dest/'requirements_revision_neural.txt')
    (dest/'replay_adaptation.json').write_text(json.dumps(dict(original_runner_sha256=sha(BASE/'neural_study.py'),
        replay_runner_sha256=sha(dest/'neural_study.py'),source_line_changes=[dict(before=original_line,after=replacement),
        dict(before=depth_line,after=depth_replacement)],copied_helpers={name:sha(dest/name) for name in helpers},
        numerical_settings_changed=False,commands=['python neural_study.py train','python check_reproducibility.py',
        'python analyse_study.py --freeze','python neural_study.py run --workers 4','python analyse_study.py',
        'python extra_neural_diagnostics.py','python audit_complete_inputs.py','python render_neural_report.py']),indent=2),encoding='utf-8')
    print(str(dest))

def main():
    p=argparse.ArgumentParser();p.add_argument('--snapshot-original',action='store_true');p.add_argument('--destination');a=p.parse_args()
    if a.snapshot_original:snapshot()
    if a.destination:prepare(a.destination)
    if not a.snapshot_original and not a.destination:p.error('Choose snapshot-original or destination')

if __name__=='__main__':main()
