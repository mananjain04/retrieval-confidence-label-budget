"""Slow independent full-corpus reference rankings for one heldout query per system."""
from pathlib import Path
from collections import Counter
import argparse,hashlib,json,math
import numpy as np
from prepare_data import ROOT,read_data,tokens,dump,sha

def main(data,cache):
    split=json.loads((ROOT/'split_ids.json').read_text());checked=[]
    for name,s in split.items():
        docs,queries,_,_=read_data(data,name);ids=sorted(docs);counts=[Counter(tokens(docs[d].get('title','')+' '+docs[d]['text'])) for d in ids]
        lengths=np.array([sum(c.values()) for c in counts]);total=lengths.sum();q=s['test'][0];terms=Counter(tokens(queries[q]));qi=(s['train']+s['test']).index(q)
        df={t:sum(t in c for c in counts) for t in terms};cf={t:sum(c[t] for c in counts) for t in terms};known={t:f for t,f in terms.items() if cf[t]}
        bm=np.zeros(len(ids));ql=np.zeros(len(ids));cs=sum(f*math.log(cf[t]/total) for t,f in known.items());L=sum(known.values())
        for t,f in known.items():
            tf=np.array([c[t] for c in counts]);idf=math.log1p((len(ids)-df[t]+.5)/(df[t]+.5))
            bm+=idf*tf*2.2/(tf+1.2*(.25+.75*lengths/lengths.mean()))
            ql+=f*np.log((tf+1000*cf[t]/total)/(lengths+1000))
        if q in docs:bm[ids.index(q)]=-np.inf;ql[ids.index(q)]=-np.inf
        def order(scores,positive=False):
            candidates=[i for i,v in enumerate(scores) if np.isfinite(v) and (not positive or v>0)]
            return sorted(candidates,key=lambda i:(-float(scores[i]),hashlib.sha256(f'{name}\0{q}\0{ids[i]}'.encode()).hexdigest()))[:100]
        qtop=order(ql);nqc=float(np.std(ql[qtop])/abs(cs)) if L else 0.;wig=float((np.mean(ql[qtop])-cs)/math.sqrt(L)) if L else 0.
        for kind in ['bm25','potion_base','potion_retrieval']:
            a=np.load(ROOT/'features'/f'{name}_{kind}.npz');score=bm
            if kind!='bm25':
                de=np.load(cache/name/kind/'documents.npy',mmap_mode='r');qe=np.load(cache/name/kind/'queries.npy',mmap_mode='r')
                score=np.array([np.dot(v.astype(np.float64),qe[qi].astype(np.float64)) for v in de])
                if q in docs:score[ids.index(q)]=-np.inf
            ix=order(score,kind=='bm25');expected=[ids[i] for i in ix[:10]]
            assert a['ranked_ids'][qi,:len(expected)].tolist()==expected,(name,kind)
            err=float(np.max(np.abs(a['scores'][qi,:len(expected)]-score[ix[:10]]))) if expected else 0.;assert err<2e-6,(name,kind,err)
            assert abs(a['features'][qi,6]-nqc)<1e-10 and abs(a['features'][qi,7]-wig)<1e-10
            checked.append({'dataset':name,'retriever':kind,'qid':q,'top10_exact_match':True,'score_max_abs_error':err,'auxiliary_features_match':True})
    dump(ROOT/'RETRIEVAL_SAMPLE_VALIDATION.json',{'verdict':'PASS','scope':'One heldout query per collection and retriever, full-corpus independent Counter/float64 reference; not an exhaustive ranking reimplementation.','checks':checked,'validator_sha256':sha(Path(__file__))})
    print('Independent full-corpus ranking and QL references passed for12 samples.')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--cache',type=Path,required=True);a=p.parse_args();main(a.data,a.cache)
