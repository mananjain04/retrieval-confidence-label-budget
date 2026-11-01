"""Fetch public inputs at recorded revisions and verify exact executed bytes."""
from pathlib import Path
import argparse,hashlib,json,urllib.request
ROOT=Path(__file__).resolve().parent

def download(url,path,expected):
    path.parent.mkdir(parents=True,exist_ok=True)
    if not path.exists() or hashlib.sha256(path.read_bytes()).hexdigest()!=expected:
        urllib.request.urlretrieve(url,path)
    assert hashlib.sha256(path.read_bytes()).hexdigest()==expected,str(path)

def main(data,models):
    for d in json.loads((ROOT/'data_provenance.json').read_text()):
        download(d['url'],data/(d['dataset']+'.zip'),d['sha256'])
    for m in json.loads((ROOT/'static_model_manifest.json').read_text(encoding='utf-8-sig')):
        for f in m['files']:
            download(f"https://huggingface.co/{m['repo']}/resolve/{m['revision']}/{f['name']}",models/m['kind']/f['name'],f['sha256'])
    print('All public inputs match recorded SHA-256 values.')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--models',type=Path,required=True);a=p.parse_args();main(a.data,a.models)
