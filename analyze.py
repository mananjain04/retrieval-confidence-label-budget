import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
from pathlib import Path
from collections import defaultdict
import argparse,csv,datetime,hashlib,json,time
import numpy as np
import scipy,sklearn
from scipy.special import expit,logit
from sklearn.isotonic import IsotonicRegression
from prepare_data import ROOT,read_data,norm,sha,dump
from core import fit,predict,prior,offset,metrics,crossed_interval

def balanced_sources(target,split,queries):
    testnorm={norm(queries[target][q]) for q in split[target]['test']};selected=[]
    for dataset in split:
        if dataset==target:continue
        groups=defaultdict(list)
        for q in split[dataset]['train']:
            if norm(queries[dataset][q]) not in testnorm:groups[split[dataset]['groups'][q]].append(q)
        order=sorted(groups,key=lambda g:hashlib.sha256(('source-cap\0'+dataset+'\0'+g).encode()).hexdigest())
        used=[]
        for g in order:
            if len(used)+len(groups[g])<=600:used.extend((dataset,q) for q in groups[g])
        selected.extend(used)
    return selected

def main(data):
    start=time.perf_counter();results=ROOT/'results';results.mkdir(exist_ok=True)
    protocol=json.loads((ROOT/'protocol_v3.json').read_text());phash=sha(ROOT/'protocol_v3.json')
    split=json.loads((ROOT/'split_ids.json').read_text());queries={n:read_data(data,n)[1] for n in split}
    meta=[];contrasts=[];params=[];runids={};pred_hashes={}
    specs=[(m,0) for m in ['source_prior','source_score','source_nqc','source_wig','source_multi']]+[(m,b) for b in [25,50,100] for m in ['offset_adapt','platt_adapt','isotonic_adapt','target_prior','target_only']]
    for target in ['scifact','nfcorpus','arguana','scidocs']:
        train=split[target]['train'];test=split[target]['test'];selected=balanced_sources(target,split,queries)
        sourcegroups=defaultdict(list)
        for i,(n,q) in enumerate(selected):sourcegroups[n,split[n]['groups'][q]].append(i)
        runids[target]={'source_pool':selected,'fits':{}}
        for seed in range(2000,2020):
            rng=np.random.default_rng(seed);boot=[]
            for name in sorted(set(n for n,_ in selected)):
                gs=[v for (n,g),v in sourcegroups.items() if n==name]
                for i in rng.integers(0,len(gs),len(gs)):boot.extend(gs[i])
            cal=np.random.default_rng(seed+991).permutation(len(train))[:100]
            runids[target]['fits'][str(seed)]={'source_bootstrap_indices':boot,'target_calibration_ids':[train[i] for i in cal]}
        for retriever in ['bm25','potion_base','potion_retrieval']:
            sets={n:np.load(ROOT/'features'/f'{n}_{retriever}.npz',allow_pickle=False) for n in split}
            lookup={n:{q:i for i,q in enumerate(sets[n]['qids'].tolist())} for n in split}
            sx=np.array([sets[n]['features'][lookup[n][q]] for n,q in selected]);tx=sets[target]['features'][:len(train)];ex=sets[target]['features'][len(train):]
            for kidx,k in enumerate([1,5,10]):
                sy=np.array([sets[n]['y'][lookup[n][q],kidx] for n,q in selected]);ty=sets[target]['y'][:len(train),kidx];ey=sets[target]['y'][len(train):,kidx]
                predictions=np.empty((20,len(specs),len(test)),dtype=np.float64)
                for repeat,seed in enumerate(range(2000,2020)):
                    ids=runids[target]['fits'][str(seed)];boot=np.array(ids['source_bootstrap_indices']);calindex={q:i for i,q in enumerate(train)}
                    p_source={};train_source_p=None
                    for method,cols in [('source_score',[0]),('source_nqc',[6]),('source_wig',[7]),('source_multi',range(8))]:
                        m,pa=fit(sx[boot],sy[boot],cols);p_source[method]=predict(m,pa,ex)
                        params.append({'target':target,'retriever':retriever,'k':k,'seed':seed,'method':method,'budget':0,'params':pa})
                        if method=='source_multi':train_source_p=predict(m,pa,tx)
                    p_source['source_prior']=np.full(len(ey),prior(sy[boot]))
                    for j,(method,budget) in enumerate(specs):
                        if budget==0:p=p_source[method]
                        else:
                            ix=np.array([calindex[q] for q in ids['target_calibration_ids'][:budget]]);yc=ty[ix];zcal=logit(np.clip(train_source_p[ix],1e-6,1-1e-6));ztest=logit(np.clip(p_source['source_multi'],1e-6,1-1e-6))
                            if method=='offset_adapt':
                                delta=offset(train_source_p[ix],yc);p=expit(ztest+delta);pa={'delta':delta}
                            elif method=='platt_adapt':
                                m,pa=fit(zcal[:,None],yc,[0]);p=predict(m,pa,ztest[:,None])
                            elif method=='isotonic_adapt':
                                if len(np.unique(yc))<2:p=np.full(len(ey),prior(yc));pa={'fallback':prior(yc)}
                                else:
                                    iso=IsotonicRegression(increasing=True,out_of_bounds='clip');iso.fit(zcal,yc);p=iso.predict(ztest);pa={'x':iso.X_thresholds_.tolist(),'y':iso.y_thresholds_.tolist()}
                            elif method=='target_prior':p=np.full(len(ey),prior(yc));pa={'prior':prior(yc),'positive_count':int(yc.sum())}
                            else:m,pa=fit(tx[ix],yc,range(8));p=predict(m,pa,ex)
                            params.append({'target':target,'retriever':retriever,'k':k,'seed':seed,'method':method,'budget':budget,'params':pa})
                        predictions[repeat,j]=p
                        mm=metrics(ey,p);mm['bin_counts']=json.dumps(mm['bin_counts'])
                        meta.append({'target':target,'retriever':retriever,'k':k,'method':method,'budget':budget,'seed':seed,**mm})
                predpath=results/f'{target}_{retriever}_k{k}.npz'
                np.savez_compressed(predpath,p=predictions,y=ey,qids=np.array(test),methods=np.array([m for m,b in specs]),budgets=np.array([b for m,b in specs]),seeds=np.arange(2000,2020))
                pred_hashes[predpath.name]=sha(predpath)
                if k==5:
                    for budget in [25,50,100]:
                        ai=specs.index(('offset_adapt',budget));ti=specs.index(('target_only',budget));loss=(predictions[:,ti]-ey)**2-(predictions[:,ai]-ey)**2
                        groups=[split[target]['groups'][q] for q in test]
                        lo,hi=crossed_interval(loss,groups)
                        contrasts.append({'target':target,'retriever':retriever,'budget':budget,'contrast':'target_only_minus_offset','delta_brier':float(loss.mean()),'ci_low':lo,'ci_high':hi,'ci_level':.9875,'fit_mean_sd':float(loss.mean(axis=1).std(ddof=1)),'n_queries':len(ey),'n_groups':len(set(groups)),'primary':target in ['arguana','scidocs'] and retriever in ['bm25','potion_retrieval'] and budget==100})
                print('analysis complete',target,retriever,k,flush=True)
    with (results/'metrics.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(meta[0]));w.writeheader();w.writerows(meta)
    dump(results/'contrasts.json',contrasts);dump(results/'fitted_parameters.json',params);dump(results/'fit_ids.json',runids)
    dump(results/'execution_registry.json',{'status':'executed','protocol_sha256':phash,'source_hashes':{f:sha(ROOT/f) for f in ['analyze.py','core.py','retrieve.py','prepare_data.py','encode_static.py']},'prediction_hashes':pred_hashes,'elapsed_analysis_seconds':time.perf_counter()-start,'completed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'python':os.sys.version,'numpy':np.__version__,'scipy':scipy.__version__,'sklearn':sklearn.__version__,'metric_rows':len(meta),'prediction_values':sum(int(np.load(results/f)['p'].size) for f in pred_hashes),'primary_tests':4,'failed_fits':[]})
    assert sha(ROOT/'protocol_v3.json')==phash
    print(json.dumps([c for c in contrasts if c['primary']],indent=2),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);main(p.parse_args().data)
