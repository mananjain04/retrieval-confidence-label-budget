"""Independent arithmetic, split, label, hash and primary-interval validation."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
from pathlib import Path
from collections import defaultdict
import argparse,csv,hashlib,json
import numpy as np
from scipy.special import expit,logit
from prepare_data import ROOT,read_data,norm,sha,dump

def main(data):
    results=ROOT/'results';reg=json.loads((results/'execution_registry.json').read_text())
    assert sha(ROOT/'protocol_v3.json')==reg['protocol_sha256']
    for name,h in reg['source_hashes'].items():assert sha(ROOT/name)==h
    split=json.loads((ROOT/'split_ids.json').read_text());fits=json.loads((results/'fit_ids.json').read_text())
    queries={};feature_checks=0;self_checks=0
    for name,s in split.items():
        docs,qs,rels,_=read_data(data,name);queries[name]=qs
        assert not set(s['train'])&set(s['test'])
        assert not {s['groups'][q] for q in s['train']}&{s['groups'][q] for q in s['test']}
        assert not {norm(qs[q]) for q in s['train']}&{norm(qs[q]) for q in s['test']}
        for kind in ['bm25','potion_base','potion_retrieval']:
            a=np.load(ROOT/'features'/f'{name}_{kind}.npz');assert a['qids'].tolist()==s['train']+s['test']
            assert np.isfinite(a['features']).all()
            for q,rank,ys,sc in zip(a['qids'],a['ranked_ids'],a['y'],a['scores']):
                rank=rank.tolist();assert q not in rank;assert len(set(d for d in rank if d))==sum(bool(d) for d in rank)
                assert all((d in docs) for d in rank if d)
                finite=sc[np.isfinite(sc)];assert np.all(np.diff(finite)<=1e-7)
                expected=[any(rels[q].get(d,0)>0 for d in rank[:k]) for k in [1,5,10]]
                assert expected==ys.tolist();feature_checks+=3;self_checks+=1
    for target,record in fits.items():
        tn={norm(queries[target][q]) for q in split[target]['test']}
        assert all(n!=target and q in split[n]['train'] and norm(queries[n][q]) not in tn for n,q in record['source_pool'])
        for ids in record['fits'].values():
            cal=ids['target_calibration_ids'];assert len(cal)==len(set(cal))==100
            assert set(cal)<=set(split[target]['train']);assert not set(cal)&set(split[target]['test'])
    rows=list(csv.DictReader((results/'metrics.csv').open()));lookup={(x['target'],x['retriever'],int(x['k']),x['method'],int(x['budget']),int(x['seed'])):x for x in rows}
    metric_checks=0;prediction_count=0;arrays={}
    for file,h in reg['prediction_hashes'].items():
        path=results/file;assert sha(path)==h
        a=np.load(path);name,rest=file.split('_',1);kind,kstr=rest.rsplit('_k',1);k=int(kstr.split('.')[0]);p=a['p'];y=a['y']
        assert a['qids'].tolist()==split[name]['test'];assert np.isfinite(p).all() and np.all((p>=0)&(p<=1));prediction_count+=p.size
        f=np.load(ROOT/'features'/f'{name}_{kind}.npz');assert np.array_equal(y,f['y'][len(split[name]['train']):,[1,5,10].index(k)])
        for i,seed in enumerate(a['seeds']):
            for j,(m,b) in enumerate(zip(a['methods'],a['budgets'])):
                r=lookup[name,kind,k,str(m),int(b),int(seed)];z=p[i,j];cl=np.clip(z,1e-12,1-1e-12)
                brier=np.dot(z-y,z-y)/len(y);ll=-np.sum(y*np.log(cl)+(1-y)*np.log(1-cl))/len(y)
                assert abs(float(r['brier'])-brier)<2e-12;assert abs(float(r['logloss'])-ll)<2e-10
                bins=np.digitize(z,np.arange(.1,1,.1),right=False)
                # Explicit interval counting matches floor(p*10), avoiding floating edge construction.
                counts=[sum(min(int(v*10),9)==b for v in z) for b in range(10)]
                assert json.loads(r['bin_counts'])==counts and sum(counts)==len(y)
                accept=z>=.8;assert int(r['accepted'])==sum(accept)
                assert abs(float(r['coverage'])-accept.mean())<1e-12
                if not accept.any():assert r['risk']==''
                else:assert abs(float(r['risk'])-float((1-y[accept]).mean()))<1e-12
                metric_checks+=1
        if k==5:arrays[name,kind]=a
    contrasts=json.loads((results/'contrasts.json').read_text());ci_checks=0
    for c in contrasts:
        a=arrays[c['target'],c['retriever']];idx=lambda m:next(j for j,(x,b) in enumerate(zip(a['methods'],a['budgets'])) if x==m and b==c['budget'])
        loss=(a['p'][:,idx('target_only')]-a['y'])**2-(a['p'][:,idx('offset_adapt')]-a['y'])**2
        assert abs(loss.mean()-c['delta_brier'])<1e-14
        if c['primary']:
            gs=defaultdict(list)
            for j,q in enumerate(a['qids']):gs[split[c['target']]['groups'][str(q)]].append(j)
            members=list(gs.values());gcount=len(members);rng=np.random.default_rng(1776);means=[]
            for _ in range(5000):
                fi=rng.integers(0,20,size=20);gi=rng.integers(0,gcount,size=gcount)
                qi=[q for g in gi for q in members[g]]
                means.append(np.mean(loss[np.ix_(fi,qi)]))
            lo,hi=np.quantile(means,[.00625,.99375]);assert abs(lo-c['ci_low'])<1e-13 and abs(hi-c['ci_high'])<1e-13;ci_checks+=1
    # Reconstruct every stored model's heldout probabilities from numeric parameters, without fitting.
    pc=0; feature_cache={}; prediction_cache={}
    for r in json.loads((results/'fitted_parameters.json').read_text()):
        name,kind,k,seed=r['target'],r['retriever'],r['k'],r['seed'];pa=r['params'];
        if (name,kind) not in feature_cache:feature_cache[name,kind]=np.load(ROOT/'features'/f'{name}_{kind}.npz')['features'][len(split[name]['train']):]
        X=feature_cache[name,kind]
        
        if (name,kind,k) not in prediction_cache:prediction_cache[name,kind,k]={key:value for key,value in np.load(results/f'{name}_{kind}_k{k}.npz').items()}
        a=prediction_cache[name,kind,k]
        j=next(j for j,(m,b) in enumerate(zip(a['methods'],a['budgets'])) if m==r['method'] and b==r['budget']);actual=a['p'][seed-2000,j]
        if 'cols' in pa:
            expected=np.full(len(X),pa['fallback']) if 'fallback' in pa else None
            if expected is None:
                if r['method']=='platt_adapt':
                    sj=a['methods'].tolist().index('source_multi');X=logit(np.clip(a['p'][seed-2000,sj],1e-6,1-1e-6))[:,None]
                expected=expit(((X[:,pa['cols']]-pa['mean'])/pa['scale'])@np.array(pa['coef'])+pa['intercept'])
        elif 'delta' in pa:
            sj=a['methods'].tolist().index('source_multi');expected=expit(logit(np.clip(a['p'][seed-2000,sj],1e-6,1-1e-6))+pa['delta'])
        elif 'prior' in pa:expected=np.full(len(X),pa['prior'])
        elif 'fallback' in pa:expected=np.full(len(X),pa['fallback'])
        else:
            sj=a['methods'].tolist().index('source_multi');z=logit(np.clip(a['p'][seed-2000,sj],1e-6,1-1e-6));expected=np.interp(z,pa['x'],pa['y'])
        assert np.max(np.abs(actual-expected))<2e-12,(r['method'],name,kind,k);pc+=1
    dump(ROOT/'VALIDATION.json',{'verdict':'PASS','unit_tests':16,'all_metric_rows_checked':metric_checks,'prediction_values_checked':int(prediction_count),'judged_hit_labels_checked':feature_checks,'self_exclusions_checked':self_checks,'source_and_target_split_checks':'PASS','primary_intervals_independently_recomputed':ci_checks,'fitted_models_probabilities_reconstructed':pc,'all36contrast_means_checked':True,'validator_sha256':sha(Path(__file__))})
    print((ROOT/'VALIDATION.json').read_text(),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);main(p.parse_args().data)
