import unittest
import numpy as np
from scipy.stats import multivariate_normal
import lab as l

class PartialPoolingTests(unittest.TestCase):
    def setUp(self):
        self.rng=np.random.default_rng(771)
        self.cov=l.covariances(2)
        self.chol=np.linalg.cholesky(self.cov)
    def draw(self,m):
        return np.einsum('nij,nkj->nki',self.chol,self.rng.normal(size=(40,m,2)))
    def test_score_against_independent_density(self):
        d=l.differences(self.draw(6))
        exact=np.mean([-2*multivariate_normal.logpdf(d[i],cov=s)-2*np.log(2*np.pi) for i,s in enumerate(self.cov)])
        self.assertAlmostEqual(l.score(self.cov,d),exact,places=11)
    def test_difference_covariance_and_offset_invariance(self):
        z=self.draw(20000);d=l.differences(z)
        empirical=np.einsum('nki,nkj->nij',d,d)/d.shape[1]
        # Whiten to compare scales without a misleading large absolute tolerance.
        whitened=np.linalg.solve(self.chol,empirical)
        whitened=np.linalg.solve(self.chol,whitened.transpose(0,2,1)).transpose(0,2,1)
        np.testing.assert_allclose(whitened,np.broadcast_to(np.eye(2),whitened.shape),atol=.055)
        np.testing.assert_allclose(l.differences(z+np.arange(40)[:,None,None]),d,atol=1e-14)
    def test_convex_endpoints_pd_and_unit_invariance(self):
        a=self.draw(6);b=self.draw(6);e=l.estimate(a);p=l.estimate(a,l.CLASSES)
        np.testing.assert_array_equal(l.mix(e,p,0),e);np.testing.assert_array_equal(l.mix(e,p,1),p)
        for w in l.WEIGHTS:self.assertTrue(np.all(np.linalg.eigvalsh(l.mix(e,p,w))>0))
        s,w,score=l.tune(a,b)
        scale=np.array([1000.,.001]);s2,w2,scores2=l.tune(a*scale,b*scale)
        self.assertEqual(w,w2);np.testing.assert_allclose(s2,s*scale[:,None]*scale[None,:],rtol=1e-11)
        np.testing.assert_allclose(scores2,score,atol=1e-10) # determinant scale=1
    def test_proper_score_population_minimum(self):
        # Independent many-sample validation: true covariance beats wrong scale.
        d=l.differences(self.draw(4000))
        true=l.score(self.cov,d)
        for scale in [.5,2.]:self.assertLess(true,l.score(self.cov*scale,d))
    def test_selection_inputs_and_invalid(self):
        a=self.draw(6);b=self.draw(6);s,w,_=l.tune(a,b)
        shifted=l.tune(a+17,b-9)
        self.assertEqual(w,shifted[1]);np.testing.assert_allclose(s,shifted[0],atol=1e-13)
        for bad in [np.zeros((40,3,2)),np.full((40,6,2),np.nan)]:
            with self.assertRaises(ValueError):l.differences(bad)
        with self.assertRaises(ValueError):l.mix(self.cov,self.cov,1.1)
        with self.assertRaises(ValueError):l.score(-self.cov,l.differences(b))
    def test_independent_seed_end_to_end(self):
        report=l.experiment(trials=30,seed=921)
        self.assertEqual(report['equal_total_pairs'],480)
        self.assertEqual(report['failed_fits'],0)
        for row in report['rows']:
            self.assertEqual(sum(row['selection_counts'].values()),30)
            for summary in row['methods'].values():self.assertTrue(np.isfinite(summary['rmse_ms']))
        # Broad physical sanity: calibrated camera delay remains subsecond.
        self.assertLess(max(r['methods']['tuned_6_plus_6']['rmse_ms'] for r in report['rows']),150)

if __name__=='__main__':unittest.main()
