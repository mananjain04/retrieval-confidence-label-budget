from collections import defaultdict
import hashlib,math
import numpy as np
from scipy.special import expit,logit
from scipy.optimize import brentq
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score,average_precision_score

def top(scores,ids,name,qid,limit=100,positive=False):
    good=np.flatnonzero(np.isfinite(scores)&((scores>0) if positive else True))
    if len(good)>limit:
        threshold=np.partition(scores[good],-limit)[-limit];good=good[scores[good]>=threshold]
    chosen=sorted(good,key=lambda i:(-float(scores[i]),hashlib.sha256(f'{name}\0{qid}\0{ids[i]}'.encode()).hexdigest()))[:limit]
    return chosen
def prior(y):return float((np.sum(y)+.5)/(len(y)+1))
def fit(X,y,cols):
    cols=list(cols)
    if len(np.unique(y))<2:return None,{'cols':cols,'fallback':prior(y)}
    m=make_pipeline(StandardScaler(),LogisticRegression(C=1,max_iter=2000,solver='lbfgs'))
    m.fit(X[:,cols],y);s,l=m.steps[0][1],m.steps[1][1];assert int(l.n_iter_[0])<2000
    return m,{'cols':cols,'mean':s.mean_.tolist(),'scale':s.scale_.tolist(),'coef':l.coef_[0].tolist(),'intercept':float(l.intercept_[0]),'iterations':int(l.n_iter_[0])}
def predict(m,par,X):return np.full(len(X),par['fallback']) if m is None else m.predict_proba(X[:,par['cols']])[:,1]
def offset(p,y):
    z=logit(np.clip(p,1e-6,1-1e-6));return float(brentq(lambda d:float((expit(z+d)-y).sum()+d),-len(y)-50,len(y)+50,xtol=1e-12))
def metrics(y,p):
    y=np.asarray(y);p=np.asarray(p);assert np.isfinite(p).all() and ((p>=0)&(p<=1)).all()
    clipped=np.clip(p,1e-12,1-1e-12);bins=np.minimum((p*10).astype(int),9);accept=p>=.8
    counts=[int(np.sum(bins==i)) for i in range(10)]
    ece=sum(counts[i]/len(y)*abs(float(p[bins==i].mean()-y[bins==i].mean())) for i in range(10) if counts[i])
    return {'n':len(y),'brier':float(np.mean((p-y)**2)),'logloss':float(-np.mean(y*np.log(clipped)+(1-y)*np.log1p(-clipped))),'hit_rate':float(y.mean()),'auroc':float(roc_auc_score(y,p)) if len(np.unique(y))>1 else None,'ap':float(average_precision_score(y,p)) if len(np.unique(y))>1 else None,'ece10':ece,'coverage':float(accept.mean()),'risk':float(1-y[accept].mean()) if accept.any() else None,'accepted':int(accept.sum()),'bin_counts':counts}
def crossed_interval(loss,groups,replicates=5000,seed=1776,alpha=.0125):
    # Paired crossed resampling: fitted algorithm realizations and test query groups.
    gg=defaultdict(list)
    for i,g in enumerate(groups):gg[g].append(i)
    idx=[np.array(v) for v in gg.values()];rng=np.random.default_rng(seed);samples=[]
    for _ in range(replicates):
        fits=rng.integers(0,len(loss),len(loss));which=rng.integers(0,len(idx),len(idx));queries=np.concatenate([idx[g] for g in which]);samples.append(float(loss[fits][:,queries].mean()))
    lo,hi=np.quantile(samples,[alpha/2,1-alpha/2])
    return float(lo),float(hi)
