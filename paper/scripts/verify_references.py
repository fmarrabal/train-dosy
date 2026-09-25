"""Archive registered bibliographic metadata. TLS verification stays enabled."""
from pathlib import Path
import json,re,datetime,time
import requests
ROOT=Path(__file__).resolve().parents[1]
dois=re.findall(r'https://doi.org/([^}]+)',(ROOT/'references.tex').read_text(encoding='utf-8'))
def fetch(doi):
    url='https://api.crossref.org/works/'+requests.utils.quote(doi,safe='')
    try:
        r=requests.get(url,timeout=25);r.raise_for_status();m=r.json()['message']
        return dict(doi=doi,status='verified_registered_metadata',title=m.get('title'),author=m.get('author'),container=m.get('container-title'),volume=m.get('volume'),page=m.get('page'),article_number=m.get('article-number'),published=m.get('published'),source=url)
    except Exception as e:return dict(doi=doi,status='not_verified',error=str(e),source=url)
target=ROOT/'evidence/reference_metadata.json'
previous=json.loads(target.read_text(encoding='utf-8')) if target.exists() else {'references':[]}
cache={x['doi']:x for x in previous['references'] if x['status']=='verified_registered_metadata'}
out=[]
for doi in dois:
    if doi in cache:out.append(cache[doi]);continue
    row=fetch(doi);out.append(row)
    if row['status']=='not_verified' and '429' in row.get('error',''):
        # Stop network work on rate limiting; keep all earlier verified records.
        out.extend(cache.get(d,{'doi':d,'status':'not_verified','error':'deferred after rate limit'}) for d in dois[len(out):]);break
    time.sleep(.5)
(ROOT/'evidence/reference_metadata.json').write_text(json.dumps({'checked_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'references':out},ensure_ascii=False,indent=2),encoding='utf-8')
for r in out:print(r['doi'],r['status'],r.get('title'),r.get('volume'),r.get('page'),r.get('article_number'),r.get('published'))
