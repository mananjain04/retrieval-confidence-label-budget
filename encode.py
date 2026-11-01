"""Pinned CPU ONNX inference. Called with a Python environment containing FastEmbed."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
from pathlib import Path
import argparse,datetime,hashlib,importlib.metadata,json,time
import numpy as np
from prepare_data import ROOT,read_data,sha,dump

def embed(model,texts,query,prefix,label):
    # Length sorting reduces padding while restoring exact source order afterward.
    indexed=sorted(enumerate(texts),key=lambda x:len(x[1]));output=np.empty((len(texts),384),np.float32)
    for begin in range(0,len(indexed),32):
        batch=indexed[begin:begin+32];actual=[prefix+s if query else s for _,s in batch]
        vec=np.array(list(model.embed(actual,batch_size=8)),dtype=np.float32)
        vec/=np.linalg.norm(vec,axis=1,keepdims=True)
        assert np.isfinite(vec).all()
        for (i,_),v in zip(batch,vec):output[i]=v
        if begin%1024==0:print(label,begin+len(batch),'/',len(texts),flush=True)
    return output

def direct_pool_check(model,text,kind):
    # Use the backend tokenizer/session directly, independently of FastEmbed's pooling.
    enc=model.model.tokenizer.encode_batch([text])
    ids=np.array([e.ids for e in enc],dtype=np.int64);mask=np.array([e.attention_mask for e in enc],dtype=np.int64);types=np.array([e.type_ids for e in enc],dtype=np.int64)
    feed={'input_ids':ids,'attention_mask':mask,'token_type_ids':types}
    session=model.model.model
    actual_feed={x.name:feed[x.name] for x in session.get_inputs()}
    last=session.run(None,actual_feed)[0]
    expected=last[:,0,:] if kind=='bge_small' else (last*mask[:,:,None]).sum(axis=1)/mask.sum(axis=1,keepdims=True)
    expected=expected[0]/np.linalg.norm(expected[0]);actual=np.array(list(model.embed([text])))[0]
    actual=actual/np.linalg.norm(actual)
    error=float(np.max(np.abs(expected-actual)));assert error<2e-5,(kind,error)
    return error

def main(data,models,cache,kind):
    import truststore;truststore.inject_into_ssl()
    from fastembed import TextEmbedding
    name='BAAI/bge-small-en-v1.5' if kind=='bge_small' else 'sentence-transformers/all-MiniLM-L6-v2'
    path=models/('bge-small' if kind=='bge_small' else 'minilm')
    model=TextEmbedding(model_name=name,specific_model_path=str(path.resolve()),local_files_only=True,threads=4,providers=['CPUExecutionProvider'],cache_dir=str(cache/'backend'),enable_cpu_mem_arena=False)
    model.model.tokenizer.enable_truncation(max_length=512 if kind=='bge_small' else 256)
    prefix='Represent this sentence for searching relevant passages: ' if kind=='bge_small' else ''
    start=time.perf_counter();checks={}
    split=json.loads((ROOT/'split_ids.json').read_text())
    manifest={'kind':kind,'model_name':name,'query_prefix':prefix,'max_tokens':512 if kind=='bge_small' else 256,'providers':['CPUExecutionProvider'],'threads':4,'batch_size':8,'cpu_memory_arena':False,'files':{p.name:{'sha256':sha(p),'bytes':p.stat().st_size} for p in path.glob('*') if p.is_file()},'packages':{p:importlib.metadata.version(p) for p in ['fastembed','onnxruntime','tokenizers','numpy']}}
    for dataset in ['scifact','nfcorpus','arguana','scidocs']:
        docs,queries,_,_=read_data(data,dataset);dids=sorted(docs);qids=split[dataset]['train']+split[dataset]['test']
        dtext=[docs[d].get('title','')+' '+docs[d]['text'] for d in dids];qtext=[queries[q] for q in qids]
        location=cache/dataset/kind;location.mkdir(parents=True,exist_ok=True)
        for label,texts,query in [('documents',dtext,False),('queries',qtext,True)]:
            target=location/(label+'.npy');fingerprint=hashlib.sha256(json.dumps({'texts':texts,'prefix':prefix if query else '', 'files':manifest['files']},sort_keys=True).encode()).hexdigest()
            meta=location/(label+'_input.json')
            if target.exists() and meta.exists() and json.loads(meta.read_text())['input_sha256']==fingerprint:
                print('verified cache',dataset,kind,label,flush=True)
            else:
                vectors=embed(model,texts,query,prefix,dataset+' '+kind+' '+label);np.save(target,vectors);dump(meta,{'input_sha256':fingerprint,'shape':list(vectors.shape),'output_sha256':sha(target)})
        checks[dataset]=direct_pool_check(model,prefix+qtext[0],kind)
    manifest['direct_pool_max_abs_errors']=checks;manifest['wall_seconds']=time.perf_counter()-start;manifest['completed_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
    dump(ROOT/(kind+'_inference_manifest.json'),manifest);print(kind,'CPU inference complete',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--models',type=Path,required=True);p.add_argument('--cache',type=Path,required=True);p.add_argument('--kind',choices=['minilm','bge_small'],required=True);a=p.parse_args();main(a.data,a.models,a.cache,a.kind)
