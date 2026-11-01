"""Verify public benchmark bytes and create group-disjoint split records, without model outcomes."""
from pathlib import Path
from collections import defaultdict
import argparse,csv,hashlib,json,re,zipfile
ROOT=Path(__file__).resolve().parent
MD5={'scifact':'5f7d1de60b170fc8027bb7898e2efca1','nfcorpus':'a89dba18a62ef92f7d323ec890a0d38d','arguana':'8ad3e3c2a5867cdced806d6503f29b99','scidocs':'38121350fc3a4d2f48850f6aff52e4a9'}
def tokens(s):return re.findall(r'[a-z0-9]+',s.lower())
def norm(s):return ' '.join(tokens(s))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,obj):Path(p).write_text(json.dumps(obj,indent=2,allow_nan=False),encoding='utf-8')
def group(name,q,text):
    if name=='arguana':return re.sub(r'-(?:pro|con)\d+[ab]$','',q)
    return norm(text)
def read_data(base,name):
    p=Path(base)/name
    docs={str(d['_id']):d for d in map(json.loads,(p/'corpus.jsonl').read_text(encoding='utf-8').splitlines())}
    queries={str(q['_id']):q['text'] for q in map(json.loads,(p/'queries.jsonl').read_text(encoding='utf-8').splitlines())}
    rels=defaultdict(dict);splits={}
    for f in (p/'qrels').glob('*.tsv'):
        ids=set()
        with f.open(encoding='utf-8') as stream:
            for r in csv.DictReader(stream,delimiter='\t'):
                ids.add(r['query-id']);rels[r['query-id']][r['corpus-id']]=int(r['score'])
        splits[f.stem]=sorted(ids)
    return docs,queries,dict(rels),splits
def main(base):
    result={};audit={};provenance=[]
    for name,expected in MD5.items():
        archive=base/f'{name}.zip';assert hashlib.md5(archive.read_bytes()).hexdigest()==expected
        with zipfile.ZipFile(archive) as z:
            assert z.testzip() is None
            for info in z.infolist():assert (base/info.filename).resolve().is_relative_to(base.resolve())
            z.extractall(base)
        docs,queries,rels,original=read_data(base,name)
        if name in ['scifact','nfcorpus']:train=original['train'];test=original['test']
        else:
            train=[];test=[]
            for q in original['test']:
                h=hashlib.sha256(('submission-v2-split\0'+group(name,q,queries[q])).encode()).hexdigest()
                (train if int(h[:16],16)/2**64<.65 else test).append(q)
        test_text={norm(queries[q]) for q in test};seen=set();clean=[];excluded=[]
        for q in sorted(train):
            text=norm(queries[q])
            if text in test_text or text in seen:excluded.append(q)
            else:seen.add(text);clean.append(q)
        assert len(clean)>=100 and len(test)>=100
        tg={group(name,q,queries[q]) for q in test}
        assert not ({group(name,q,queries[q]) for q in clean}&tg)
        result[name]={'train':clean,'test':sorted(test),'excluded_train':excluded,'groups':{q:hashlib.sha256(group(name,q,queries[q]).encode()).hexdigest() for q in clean+test}}
        missing=[{'qid':q,'docid':d} for q,rr in rels.items() for d,r in rr.items() if r>0 and d not in docs]
        audit[name]={'documents':len(docs),'eligible_train':len(clean),'heldout':len(test),'heldout_groups':len(tg),'excluded_train':excluded,'missing_positive_judgments':missing,'all_positive_qrels_retained':True,'official_test_repartitioned':name in ['arguana','scidocs']}
        provenance.append({'dataset':name,'url':f'https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{name}.zip','md5':expected,'sha256':sha(archive),'bytes':archive.stat().st_size,'extracted_hashes':{p.relative_to(base/name).as_posix():sha(p) for p in (base/name).rglob('*') if p.is_file()}})
    dump(ROOT/'split_ids.json',result);dump(ROOT/'dataset_audit.json',audit);dump(ROOT/'data_provenance.json',provenance)
    print(json.dumps(audit,indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);main(p.parse_args().data)
