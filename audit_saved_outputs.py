from pathlib import Path
from collections import defaultdict
import csv, hashlib, json, re, sys
import numpy as np

ROOT=Path(__file__).parent
STUDY=Path(sys.argv[1]).resolve()
DATA=Path(sys.argv[2]).resolve()
(ROOT/'qa').mkdir(exist_ok=True)
result={"scope":"Independent recalculation of saved experimental outputs; not a fresh retriever run or human approval","status":"RUNNING"}
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
reg=json.loads((STUDY/'results/execution_registry.json').read_text())
split=json.loads((STUDY/'split_ids.json').read_text())
assert sha(STUDY/'protocol_v3.json')==reg['protocol_sha256']
for name,h in reg['source_hashes'].items(): assert sha(STUDY/name)==h
queries={};rels={};docids={};label_checks=0
norm=lambda t:' '.join(re.findall('[a-z0-9]+',t.lower()))
for name,s in split.items():
 d=DATA/name
 queries[name]={str(q['_id']):q['text'] for q in map(json.loads,(d/'queries.jsonl').read_text(encoding='utf8').splitlines())}
 docids[name]={str(q['_id']) for q in map(json.loads,(d/'corpus.jsonl').read_text(encoding='utf8').splitlines())}
 rr=defaultdict(dict)
 for f in (d/'qrels').glob('*.tsv'):
  for row in csv.DictReader(f.open(encoding='utf8'),delimiter='\t'): rr[row['query-id']][row['corpus-id']]=int(row['score'])
 rels[name]=rr
 assert not set(s['train'])&set(s['test'])
 assert not {s['groups'][q] for q in s['train']}&{s['groups'][q] for q in s['test']}
 assert not {norm(queries[name][q]) for q in s['train']}&{norm(queries[name][q]) for q in s['test']}
 for kind in ['bm25','potion_base','potion_retrieval']:
  a=np.load(STUDY/'features'/f'{name}_{kind}.npz',allow_pickle=False)
  assert a['qids'].tolist()==s['train']+s['test']
  assert np.isfinite(a['features']).all()
  for q,rank,y in zip(a['qids'],a['ranked_ids'],a['y']):
   assert q not in rank
   assert all(d in docids[name] for d in rank if d)
   assert len(set(d for d in rank if d))==sum(bool(d) for d in rank)
   assert [int(any(rr[str(q)].get(d,0)>0 for d in rank[:k])) for k in [1,5,10]]==y.tolist()
   label_checks+=3
fits=json.loads((STUDY/'results/fit_ids.json').read_text())
for target,record in fits.items():
 tn={norm(queries[target][q]) for q in split[target]['test']}
 assert all(n!=target and q in split[n]['train'] and norm(queries[n][q]) not in tn for n,q in record['source_pool'])
 for ids in record['fits'].values():
  cal=ids['target_calibration_ids']; assert len(cal)==len(set(cal))==100
  assert set(cal)<=set(split[target]['train']) and not set(cal)&set(split[target]['test'])
rows=list(csv.DictReader((STUDY/'results/metrics.csv').open()))
lookup={(r['target'],r['retriever'],int(r['k']),r['method'],int(r['budget']),int(r['seed'])):r for r in rows}
arrays={};prediction_count=0;metric_count=0
for file,h in reg['prediction_hashes'].items():
 path=STUDY/'results'/file; assert sha(path)==h
 name,rest=file.split('_',1);kind,ks=rest.rsplit('_k',1);k=int(ks[:-4])
 a={key:value for key,value in np.load(path,allow_pickle=False).items()}; arrays[name,kind,k]=a
 assert a['qids'].tolist()==split[name]['test']
 p,y=a['p'],a['y'];assert np.isfinite(p).all() and ((p>=0)&(p<=1)).all(); prediction_count+=p.size
 f=np.load(STUDY/'features'/f'{name}_{kind}.npz',allow_pickle=False)
 assert np.array_equal(y,f['y'][len(split[name]['train']):,[1,5,10].index(k)])
 for i,seed in enumerate(a['seeds']):
  for j,(method,budget) in enumerate(zip(a['methods'],a['budgets'])):
   row=lookup[name,kind,k,str(method),int(budget),int(seed)];z=p[i,j];cl=np.clip(z,1e-12,1-1e-12)
   assert abs(float(row['brier'])-np.mean((z-y)**2))<2e-12
   assert abs(float(row['logloss'])+np.mean(y*np.log(cl)+(1-y)*np.log1p(-cl)))<2e-10
   accept=z>=.8; assert int(row['accepted'])==sum(accept)
   assert abs(float(row['coverage'])-accept.mean())<1e-12
   assert json.loads(row['bin_counts'])==np.bincount(np.minimum((z*10).astype(int),9),minlength=10).tolist()
   if not accept.any(): assert row['risk']==''
   else: assert abs(float(row['risk'])-float((1-y[accept]).mean()))<1e-12
   metric_count+=1
primary=[]
for c in json.loads((STUDY/'results/contrasts.json').read_text()):
 a=arrays[c['target'],c['retriever'],5]
 ix=lambda m:next(j for j,(x,b) in enumerate(zip(a['methods'],a['budgets'])) if x==m and b==c['budget'])
 loss=(a['p'][:,ix('target_only')]-a['y'])**2-(a['p'][:,ix('offset_adapt')]-a['y'])**2
 assert abs(loss.mean()-c['delta_brier'])<1e-14
 if not c['primary']:continue
 gs=defaultdict(list)
 for j,q in enumerate(a['qids']):gs[split[c['target']]['groups'][str(q)]].append(j)
 members=list(gs.values());rng=np.random.default_rng(1776);means=[]
 for _ in range(5000):
  fi=rng.integers(0,20,size=20);gi=rng.integers(0,len(members),size=len(members));qi=[q for g in gi for q in members[g]]
  means.append(np.mean(loss[np.ix_(fi,qi)]))
 lo,hi=np.quantile(means,[.00625,.99375]); assert abs(lo-c['ci_low'])<1e-13 and abs(hi-c['ci_high'])<1e-13
 primary.append({'target':c['target'],'retriever':c['retriever'],'delta':float(loss.mean()),'ci':[float(lo),float(hi)]})
features={};model_count=0
for r in json.loads((STUDY/'results/fitted_parameters.json').read_text()):
 name,kind,k,seed=r['target'],r['retriever'],r['k'],r['seed'];par=r['params']
 if (name,kind) not in features: features[name,kind]=np.load(STUDY/'features'/f'{name}_{kind}.npz',allow_pickle=False)['features'][len(split[name]['train']):]
 X=features[name,kind];a=arrays[name,kind,k]
 j=next(j for j,(m,b) in enumerate(zip(a['methods'],a['budgets'])) if m==r['method'] and b==r['budget']);actual=a['p'][seed-2000,j]
 sj=a['methods'].tolist().index('source_multi');source=np.clip(a['p'][seed-2000,sj],1e-6,1-1e-6);z=np.log(source/(1-source))
 if 'cols' in par:
  if 'fallback' in par: expected=np.full(len(X),par['fallback'])
  else:
   if r['method']=='platt_adapt': X=z[:,None]
   logit=((X[:,par['cols']]-par['mean'])/par['scale'])@np.array(par['coef'])+par['intercept'];expected=1/(1+np.exp(-logit))
 elif 'delta' in par: expected=1/(1+np.exp(-(z+par['delta'])))
 elif 'prior' in par: expected=np.full(len(X),par['prior'])
 elif 'fallback' in par: expected=np.full(len(X),par['fallback'])
 else:expected=np.interp(z,par['x'],par['y'])
 assert np.max(np.abs(actual-expected))<2e-12
 model_count+=1
result.update(status='PASS',saved_probability_values_recomputed=prediction_count,metric_rows_recomputed=metric_count,judged_hit_labels_checked=label_checks,models_reconstructed=model_count,primary_comparisons=primary,independent_test_queries=sum(len(s['test']) for s in split.values()),human_author_approval='PENDING',fresh_end_to_end_reproduction='NOT_RUN',equation_code_review='Brier, clipped-logit offset objective, unigram NQC/WIG match saved implementation; CV uses absolute mean')
(ROOT/'qa'/'FINAL_SCIENTIFIC_AUDIT.json').write_text(json.dumps(result,indent=2),encoding='utf8')
print(json.dumps(result,indent=2),flush=True)
