import unittest
from pathlib import Path
import numpy as np
from scipy.special import expit,logit
from core import top,fit,predict,offset,metrics,crossed_interval
from retrieve import Index
from prepare_data import group

class ExpandedTests(unittest.TestCase):
    def test_bm25_formula(self):
        ix=Index({'a':{'text':'cat cat dog'},'b':{'text':'dog'}});bm,ql,c,n=ix.batches(['cat']);expected=np.log(2)*2*2.2/(2+1.2*(.25+.75*3/2));self.assertAlmostEqual(bm[0,0],expected,12)
    def test_dirichlet_likelihood(self):
        ix=Index({'a':{'text':'cat cat dog'},'b':{'text':'dog'}});_,ql,c,n=ix.batches(['cat dog']);expected=np.log((2+1000*.5)/(3+1000))+np.log((1+1000*.5)/(3+1000));self.assertAlmostEqual(ql[0,0],expected,12);self.assertAlmostEqual(c[0],2*np.log(.5),12)
    def test_ql_repeated_terms(self):
        ix=Index({'a':{'text':'cat cat dog'},'b':{'text':'dog'}});_,ql,c,n=ix.batches(['cat','cat cat']);np.testing.assert_allclose(ql[1],2*ql[0])
    def test_oov_safe(self):
        ix=Index({'a':{'text':'cat dog'}});_,ql,c,n=ix.batches(['zxy']);self.assertEqual(n[0],0);self.assertTrue(np.isfinite(ix.features('zxy',[],0,0)).all())
    def test_partial_sort_matches_full(self):
        rng=np.random.default_rng(5);scores=rng.random(1000);ids=[str(i) for i in range(1000)];self.assertEqual(top(scores,ids,'d','q',10),np.argsort(-scores)[:10].tolist())
    def test_ties_repeatable(self):
        x=top(np.ones(1000),[str(i) for i in range(1000)],'d','q',10);self.assertEqual(x,top(np.ones(1000),[str(i) for i in range(1000)],'d','q',10));self.assertEqual(len(set(x)),10)
    def test_positive_and_finite(self):self.assertEqual(top(np.array([0,-1,np.inf,np.nan,2]),list('abcde'),'d','q',10,True),[4])
    def test_debate_groups(self):self.assertEqual(group('arguana','test-topic-abc-con03a','x'),group('arguana','test-topic-abc-pro01a','y'))
    def test_source_scaling(self):
        X=np.arange(32).reshape(4,8);m,p=fit(X,np.array([0,0,1,1]),range(8));predict(m,p,np.full((1,8),1e6));np.testing.assert_allclose(m.steps[0][1].mean_,X.mean(axis=0))
    def test_one_class_fallback(self):
        m,p=fit(np.ones((3,8)),np.ones(3),[0]);self.assertAlmostEqual(predict(m,p,np.ones((1,8)))[0],.875)
    def test_offset_gradient(self):
        p=np.array([.2,.4,.8]);y=np.array([1,1,1]);d=offset(p,y);self.assertAlmostEqual(float((expit(logit(p)+d)-y).sum()+d),0,10)
    def test_monotone_offset_rank(self):
        p=np.array([.1,.4,.2,.8]);np.testing.assert_array_equal(np.argsort(p),np.argsort(expit(logit(p)+2)))
    def test_metrics(self):self.assertAlmostEqual(metrics([0,1],[.2,.7])['brier'],.065)
    def test_null_risk(self):self.assertIsNone(metrics([0,1],[.1,.2])['risk'])
    def test_bin_boundaries(self):self.assertEqual(sum(metrics([0,1],[0,1])['bin_counts']),2)
    def test_crossed_constant_effect(self):
        lo,hi=crossed_interval(np.full((3,4),.123),['a','a','b','c'],100);self.assertAlmostEqual(lo,.123);self.assertAlmostEqual(hi,.123)
if __name__=='__main__':unittest.main()
