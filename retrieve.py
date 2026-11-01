import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
from pathlib import Path
from collections import Counter
import argparse,hashlib,json,time
import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer
from prepare_data import ROOT,read_data,tokens,sha,dump
from core import top

class Index:
    def __init__(self,docs):
        self.ids=sorted(docs);self.pos={d:i for i,d in enumerate(self.ids)}
        self.vectorizer=CountVectorizer(tokenizer=tokens,token_pattern=None,lowercase=False,dtype=np.float64)
        counts=self.vectorizer.fit_transform([docs[d].get('title','')+' '+docs[d]['text'] for d in self.ids]).tocsr()
        self.vocab=self.vectorizer.vocabulary_;self.length=np.asarray(counts.sum(axis=1)).ravel();self.cf=np.asarray(counts.sum(axis=0)).ravel();self.total=float(self.cf.sum())
        df=np.asarray((counts>0).sum(axis=0)).ravel();self.idf=np.log1p((len(self.ids)-df+.5)/(df+.5));self.pc=self.cf/self.total
        rows=np.repeat(np.arange(len(self.ids)),np.diff(counts.indptr));weight=counts.copy()
        weight.data=counts.data*2.2/(counts.data+1.2*(.25+.75*self.length[rows]/self.length.mean()))
        self.bm=weight.multiply(self.idf).tocsr()
        ql=counts.copy();ql.data=np.log1p(counts.data/(1000*self.pc[counts.indices]));self.ql=ql.tocsr()
    def batches(self,texts):
        qc=self.vectorizer.transform(texts);bmq=qc.copy();bmq.data[:]=1
        bm=(bmq@self.bm.T).toarray();collection=np.asarray(qc@np.log(self.pc)).ravel();known=np.asarray(qc.sum(axis=1)).ravel()
        ql=(qc@self.ql.T).toarray()+collection[:,None]-known[:,None]*np.log1p(self.length[None,:]/1000)
        return bm,ql,collection,known
    def features(self,text,rank_scores,nqc,wig):
        ss=np.zeros(10);ss[:min(10,len(rank_scores))]=rank_scores[:10];ts=tokens(text);known=[self.vocab[t] for t in ts if t in self.vocab]
        return [float(np.log1p(max(0,ss[0]))),float((ss[0]-ss[1])/max(abs(ss[0]),1e-12)),float(ss.std()/max(abs(ss).mean(),1e-12)),float(np.log1p(len(ts))),float(1-len(known)/max(len(ts),1)) if ts else 0.,float(self.idf[known].mean()) if known else 0.,nqc,wig]

def main(data,cache,kinds):
    split=json.loads((ROOT/'split_ids.json').read_text());location=ROOT/'features';location.mkdir(exist_ok=True);runinfo=[]
    for name in ['scifact','nfcorpus','arguana','scidocs']:
        docs,queries,rels,_=read_data(data,name);ix=Index(docs);qids=split[name]['train']+split[name]['test'];texts=[queries[q] for q in qids]
        for kind in kinds:
            start=time.perf_counter();target=location/f'{name}_{kind}.npz'
            if target.exists():print('existing feature artifact',target.name,flush=True);continue
            if kind!='bm25':
                de=np.load(cache/name/kind/'documents.npy');qe=np.load(cache/name/kind/'queries.npy')
                assert de.shape[0]==len(ix.ids) and qe.shape==(len(qids),de.shape[1])
                assert np.isfinite(de).all() and np.isfinite(qe).all()
            feats=[];rankids=[];rankscores=[];ys=[];retrieval=[]
            for begin in range(0,len(qids),32):
                bm,ql,cs,known=ix.batches(texts[begin:begin+32])
                ds=bm if kind=='bm25' else qe[begin:begin+32]@de.T
                for j,q in enumerate(qids[begin:begin+32]):
                    if q in ix.pos:ds[j,ix.pos[q]]=-np.inf;ql[j,ix.pos[q]]=-np.inf
                    qltop=top(ql[j],ix.ids,name,q,100) if known[j]>0 else []
                    vs=ql[j,qltop]
                    nqc=float(vs.std()/max(abs(cs[j]),1e-12)) if len(vs) else 0.
                    wig=float((vs.mean()-cs[j])/np.sqrt(known[j])) if len(vs) else 0.
                    order=top(ds[j],ix.ids,name,q,100,positive=kind=='bm25');ids=[ix.ids[i] for i in order];ss=[float(ds[j,i]) for i in order]
                    feats.append(ix.features(queries[q],ss,nqc,wig));rankids.append(ids+['']*(100-len(ids)));rankscores.append(ss+[np.nan]*(100-len(ss)))
                    ys.append([int(any(rels[q].get(d,0)>0 for d in ids[:k])) for k in [1,5,10]])
                    gains=np.array([2**rels[q].get(d,0)-1 for d in ids[:10]]);ideal=np.array(sorted([2**r-1 for r in rels[q].values() if r>0],reverse=True)[:10]);den=(ideal/np.log2(np.arange(len(ideal))+2)).sum()
                    retrieval.append([float((gains/np.log2(np.arange(len(gains))+2)).sum()/den) if den else 0.,len({d for d in ids[:5] if rels[q].get(d,0)>0})/max(sum(r>0 for r in rels[q].values()),1)])
            X=np.array(feats);assert np.isfinite(X).all()
            np.savez_compressed(target,qids=np.array(qids),features=X,ranked_ids=np.array(rankids),scores=np.array(rankscores),y=np.array(ys,np.int8),retrieval=np.array(retrieval))
            info={'dataset':name,'retriever':kind,'queries':len(qids),'seconds':time.perf_counter()-start,'artifact_sha256':sha(target),'source_sha256':sha(Path(__file__))};runinfo.append(info);print(json.dumps(info),flush=True)
    previous=json.loads((ROOT/'retrieval_registry.json').read_text()) if (ROOT/'retrieval_registry.json').exists() else []
    dump(ROOT/'retrieval_registry.json',previous+runinfo)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--cache',type=Path,required=True);p.add_argument('--kind',nargs='+',default=['bm25','potion_base','potion_retrieval']);a=p.parse_args();main(a.data,a.cache,a.kind)
