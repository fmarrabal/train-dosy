import argparse,json
from pathlib import Path
from . import fit

def main():
    p=argparse.ArgumentParser(description="Whole-spectrum joint DOSY on selected signal frequencies")
    p.add_argument('input',type=Path);p.add_argument('output',type=Path)
    p.add_argument('--method',choices=['RAI-S','DOME-S','MF-AUTO'])
    a=p.parse_args();q=json.loads(a.input.read_text(encoding='utf-8'))
    if a.method:q['method']=a.method
    result=fit(q)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,allow_nan=False,indent=2),encoding='utf-8')
    print(f"{result['method']}: rank={result['selected_rank']}, success={result['success']}, {a.output}")
if __name__=='__main__':main()
