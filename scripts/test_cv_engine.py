"""Synthetic software-verification fixtures, never experimental shale results."""
import unittest,json
import numpy as np
from cv_engine import *

class TestEngine(unittest.TestCase):
 def test_moments_against_exhaustive_joint_space(self):
  pools=[[.02,.08],[.12,.2,.31],[.01,.04,.18,.4]]
  for n in (1,2,3):
   a=exact_equal_roi_moments(pools,n);b=exact_enumeration(pools,n,max_states=1000)
   self.assertAlmostEqual(a['mean'],b['mean'],places=14)
   self.assertAlmostEqual(a['variance'],b['variance'],places=14)
 def test_between_strata_finite_population(self):
  self.assertAlmostEqual(exact_equal_roi_moments([[.1],[.5],[.9]],1)['variance'],np.var([.1,.5,.9]))
  self.assertEqual(exact_equal_roi_moments([[.1],[.5],[.9]],3)['variance'],0)
 def test_area_weighting_is_different(self):
  pools=[[0.,1.],[.25,.75]];area=[[1.,9.],[2.,8.]]
  a=exact_enumeration(pools,2,max_states=100,valid_area_pools=area)
  self.assertNotAlmostEqual(a['mean'],exact_equal_roi_moments(pools,2)['mean'])
 def test_integral_image_vs_direct_counts(self):
  valid=np.ones((5,6),bool);valid[1,1]=False
  hot=np.zeros((5,6),bool);hot[::2,::2]=True;hot &= valid
  positions=[[(0,0),(1,1),(3,4)],[(0,3),(2,2)]]
  p,a=roi_fractions(valid,hot,side_px=2,stratum_positions=positions,min_valid_fraction=.5)
  for h,ps in enumerate(positions):
   for i,(t,l) in enumerate(ps):
    v=valid[t:t+2,l:l+2].sum();self.assertEqual(a[h][i],v)
    self.assertEqual(p[h][i],hot[t:t+2,l:l+2].sum()/v)
 def test_explicit_monte_carlo_reproducible(self):
  p=[[.02,.08],[.12,.2,.31],[.01,.04,.18,.4]]
  a=monte_carlo_equal_roi(p,2,repeats=30000,seed=20261018,ddof=1)
  b=monte_carlo_equal_roi(p,2,repeats=30000,seed=20261018,ddof=1)
  self.assertEqual(a,b)
  self.assertLess(abs(a['CV']-exact_equal_roi_moments(p,2)['CV']),.02)
 def test_ddof_and_missing_parameters(self):
  self.assertAlmostEqual(repeated_cv([1.,2.,3.],ddof=1),.5)
  with self.assertRaises(TypeError):monte_carlo_equal_roi([[.1]],1)
  with self.assertRaises(ValueError):exact_equal_roi_moments([[]],1)

if __name__=='__main__': unittest.main(verbosity=2)
