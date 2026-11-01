"""Pinned static embeddings: no contextual network or paid/cloud inference."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
os.environ['TOKENIZERS_PARALLELISM']='false'
from pathlib import Path
import argparse,datetime,hashlib,importlib.metadata,json,time
import numpy as np
from prepare_data import ROOT,read_data,sha,dump

def check_pool(model,text):
    backend=model.model
    ids=np.array(backend.tokenizer.encode(text,add_special_tokens=False).ids,dtype=np.int64)
    if backend.unk_token_id is not None:ids=ids[ids!=backend.unk_token_id]
    rows=ids if backend.token_mapping is None else backend.token_mapping[ids]
    vectors=backend.embeddings[rows]
    if backend.weights is not None:vectors=vectors*backend.weights[ids,None]
    expected=vectors.sum(axis=0) if len(ids) else np.zeros(backend.embeddings.shape[1])
    expected/=max(float(np.linalg.norm(expected)),1e-12)
    actual=np.array(list(model.embed([text])))[0]
    error=float(np.max(np.abs(expected-actual)));assert error<2e-5,error
    return error

def main(data,models,cache,kind):
    from fastembed import TextEmbedding
    pinned=json.loads((ROOT/'static_model_manifest.json').read_text(encoding='utf-8-sig'))
    record=next(x for x in pinned if x['kind']==kind)
    path=models/kind
    for f in record['files']:assert sha(path/f['name'])==f['sha256']
    protocol=ROOT/'protocol_v3.json';phash=sha(protocol)
    model=TextEmbedding(model_name=record['repo'],specific_model_path=str(path.resolve()),local_files_only=True,mmap=True,cache_dir=str(cache/'backend'))
    split=json.loads((ROOT/'split_ids.json').read_text());start=time.perf_counter();outputs={};checks={}
    for name in ['scifact','nfcorpus','arguana','scidocs']:
        docs,queries,_,_=read_data(data,name);dids=sorted(docs);qids=split[name]['train']+split[name]['test']
        for label,texts in [('documents',[docs[d].get('title','')+' '+docs[d]['text'] for d in dids]),('queries',[queries[q] for q in qids])]:
            loc=cache/name/kind;loc.mkdir(parents=True,exist_ok=True);target=loc/(label+'.npy')
            fingerprint=hashlib.sha256(json.dumps({'texts':texts,'model':record},sort_keys=True).encode()).hexdigest()
            vectors=np.array(list(model.embed(texts,batch_size=64)),dtype=np.float32)
            assert np.isfinite(vectors).all()
            norms=np.linalg.norm(vectors,axis=1);assert np.all((norms<1e-7)|(np.abs(norms-1)<2e-5))
            np.save(target,vectors)
            info={'input_sha256':fingerprint,'output_sha256':sha(target),'shape':list(vectors.shape),'zero_vectors':int((norms<1e-7).sum())}
            dump(loc/(label+'_input.json'),info);outputs[name+'/'+label]=info
            print(name,kind,label,len(texts),'complete',flush=True)
        checks[name]=check_pool(model,queries[qids[0]])
    assert sha(protocol)==phash
    dump(ROOT/(kind+'_inference_manifest.json'),{'status':'executed','model':record,'protocol_sha256':phash,'max_tokens':512,'query_prefix':'','batch_size':64,'device':'CPU/numpy static pooling','packages':{p:importlib.metadata.version(p) for p in ['fastembed','numpy','tokenizers']},'outputs':outputs,'direct_pool_max_abs_errors':checks,'elapsed_seconds':time.perf_counter()-start,'completed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--models',type=Path,required=True);p.add_argument('--cache',type=Path,required=True);p.add_argument('--kind',choices=['potion_base','potion_retrieval'],required=True);a=p.parse_args();main(a.data,a.models,a.cache,a.kind)
