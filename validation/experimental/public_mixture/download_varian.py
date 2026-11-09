"""Fetch exactly the public raw example; verify official archive before extraction."""
from pathlib import Path
import urllib.request,hashlib,zipfile,json
HERE=Path(__file__).resolve().parent
URL='https://nmr.chemistry.manchester.ac.uk/sites/default/files/datasets_v09.zip'
EXPECTED='1b29d59ff54392ddea05587b25d159442f51f56509c112df80145a89a6b82216'
archive=HERE/'sources/datasets_v09.zip';archive.parent.mkdir(exist_ok=True)
if not archive.exists():
    request=urllib.request.Request(URL,headers={'User-Agent':'Mozilla/5.0 scientific reproducibility'})
    with urllib.request.urlopen(request,timeout=60) as response:
        data=response.read(32*1024*1024+1)
    if len(data)>32*1024*1024:raise RuntimeError('Download exceeds recorded archive size budget')
    if hashlib.sha256(data).hexdigest()!=EXPECTED:raise RuntimeError('Upstream archive changed; original hash required')
    archive.write_bytes(data)
assert hashlib.sha256(archive.read_bytes()).hexdigest()==EXPECTED
base=(HERE/'legacy_data').resolve();base.mkdir(exist_ok=True)
prefix='TestData/DOSY/PropPentSuc_DOSY.fid/'
members=[prefix+name for name in ['fid','procpar','log','text']]
with zipfile.ZipFile(archive) as z:
    for name in members:
        path=(base/name).resolve();assert path.is_relative_to(base)
        if path.exists():
            assert path.read_bytes()==z.read(name),'Existing raw input differs; do not overwrite'
        else:
            path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(z.read(name))
print('Verified official archive and four raw input files; originals preserved.')
