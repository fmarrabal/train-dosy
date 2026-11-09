"""Verify distribution files and solver provenance without running or writing fits.

Run `python scripts/verify.py` from any directory. With --sources-only, verify
the solver/provenance records without requiring the full distribution manifest.
The hashes check files, not numerical convergence or scientific identification.
"""
import argparse
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads((ROOT/path).read_text(encoding='utf-8'))


def verify(records):
    seen=set()
    for record in records:
        name=record['path']
        path=(ROOT/name).resolve()
        if not path.is_relative_to(ROOT) or path in seen or not path.is_file():
            raise ValueError(f'Invalid, duplicate, or missing file: {name}')
        seen.add(path)
        expected_size=record.get('bytes',record.get('size'))
        if expected_size is not None and path.stat().st_size != expected_size:
            raise ValueError(f'File size mismatch: {name}')
        if digest(path) != record['sha256']:
            raise ValueError(f'SHA-256 mismatch: {name}')
    if not seen:
        raise ValueError('The expected file inventory is empty')
    return len(seen)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sources-only',action='store_true')
    args=parser.parse_args()
    source_count=verify(read('provenance/source_hashes.json'))
    mf=read('provenance/train_mf_sources.json')
    mf_count=verify(mf['implementation']+mf['evidence']+mf.get('layout_regressions',[]))
    result={'sources_verified':source_count,'train_mf_records_verified':mf_count,
            'protocol':mf['protocol']}
    if not args.sources_only:
        manifest=read('MANIFEST.json')
        files=manifest['files']
        if isinstance(files,dict):
            files=[{'path':name, **({'sha256':record} if isinstance(record,str) else record)}
                   for name,record in files.items()]
        result['distribution_files_verified']=verify(files)
    result['passed']=True
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
