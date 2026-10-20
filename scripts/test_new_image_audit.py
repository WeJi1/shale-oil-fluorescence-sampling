"""Independent audit software unit tests; synthetic arrays are NOT shale experimental data."""
import unittest
import numpy as np
from final_image_audit import integrals,boxes,rounded_side,select_grid,get_pools,position_fractions
from cv_engine import exact_equal_roi_moments,exact_enumeration

class ImageAuditTests(unittest.TestCase):
 def test_integral_windows(self):
  a=np.array([[1,0,1,1],[1,1,0,0],[0,1,1,0]],dtype=bool)
  ii=integrals(a)
  for L in (1,2,3):
   got=boxes(ii,L)
   expected=np.array([[a[i:i+L,j:j+L].sum() for j in range(a.shape[1]-L+1)] for i in range(a.shape[0]-L+1)])
   np.testing.assert_array_equal(got,expected)
 def test_valid_fraction(self):
  v=np.ones((4,4),bool);v[0,0]=False
  h=np.zeros((4,4),bool);h[0,1]=True;h[1,0]=True
  dat,ok=position_fractions(integrals(v),integrals(h),2)
  self.assertFalse(ok[0,0]);self.assertTrue(ok[0,1]);self.assertAlmostEqual(dat[0,1],.25)
 def test_grid_contains_N(self):
  for N in range(2,33):
   rows,cols=select_grid(N,250,300)
   self.assertGreaterEqual(rows*cols,N)
   self.assertLessEqual(rows*cols,2*N)
 def test_pools_cover_every_position_once(self):
  data=np.arange(35,dtype=float).reshape(5,7);ok=np.ones_like(data,dtype=bool)
  for rows,cols in [(1,2),(2,3),(3,4)]:
   pools=get_pools(data,ok,rows,cols)
   flat=np.sort(np.concatenate(pools))
   np.testing.assert_array_equal(flat,data.ravel())
 def test_exact_engine_matches_enumeration(self):
  pools=[[.1,.3],[.05,.1],[.9,.4]]
  for N in (1,2,3):
   a=exact_equal_roi_moments(pools,N);b=exact_enumeration(pools,N,max_states=1000)
   self.assertAlmostEqual(a['CV'],b['CV'])
 def test_rounding_rule(self):
  self.assertEqual(rounded_side(15.5),16);self.assertEqual(rounded_side(15.4999),15)
 def test_window_cv_small_bruteforce(self):
  v=np.ones((4,4),bool);h=np.zeros((4,4),bool);h[0:2,0:2]=True
  w,ok=position_fractions(integrals(v),integrals(h),2)
  wanted=np.array([[h[y:y+2,x:x+2].mean() for x in range(3)] for y in range(3)])
  np.testing.assert_allclose(w[ok],wanted.ravel())
 def test_nonoverlap_is_not_imposed(self):
  data=np.arange(16,dtype=float).reshape(4,4);ok=np.ones((4,4),bool)
  p=get_pools(data,ok,2,2)
  self.assertEqual(len(p),4)
  self.assertEqual(sum(map(len,p)),16)

if __name__=='__main__':unittest.main(verbosity=2)
