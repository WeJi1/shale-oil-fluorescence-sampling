import csv,sys,math,unittest,pathlib,numpy as np
from scipy.signal import fftconvolve
ROOT=pathlib.Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from scipy.signal import fftconvolve

def load():
 with (ROOT/'A_Design_Bias_and_CV_by_Map.csv').open(encoding='utf-8-sig') as f:return list(csv.DictReader(f))
class TestDesignBias(unittest.TestCase):
 def test_ht_unbiased_target_hot(self):
  rows=[x for x in load() if x['target']=='hotspot' and x['method']=='HT_STRAT_pixel_exposure']
  self.assertEqual(len(rows),14)
  for r in rows:self.assertAlmostEqual(float(r['expected']),float(r['reference']),places=9)
 def test_ht_unbiased_target_continuous(self):
  rows=[x for x in load() if x['target']=='continuous_proxy' and x['method']=='HT_STRAT_pixel_exposure']
  for r in rows:self.assertAlmostEqual(float(r['expected']),float(r['reference']),places=7)
 def test_coverage_all(self):
  rows=[x for x in load() if x['method']=='HT_STRAT_pixel_exposure']
  self.assertTrue(all(x['coverage_all_valid']=='True' and int(x['uncovered_valid_pixels'])==0 for x in rows))
 def test_design_weights_synthetic(self):
  # 6x8 images, admissible ROI top-left positions, two strata and two sampled ROIs
  H,W,L=6,8,2; good=np.ones((H-L+1,W-L+1),dtype=bool)
  rr,cc=1,2;K=rr*cc
  q=np.zeros_like(good,dtype=float);split=(good.shape[1]//2)
  for sl in [np.s_[:,0:split],np.s_[:,split:]]:
   q[sl]=2/K/good[sl].sum()
  lamb=fftconvolve(q,np.ones((L,L)),mode='full')
  self.assertAlmostEqual(q.sum(),2)
  self.assertEqual(lamb.shape,(H,W))
  self.assertTrue(np.all(lamb>0))
  z=np.arange(H*W,dtype=float).reshape(H,W)
  weights=(z/lamb)
  ii=np.pad(weights,((1,0),(1,0))).cumsum(0).cumsum(1)
  contributions=ii[L:,L:]-ii[:-L,L:]-ii[L:,:-L]+ii[:-L,:-L]
  estimate=np.sum(contributions*q)/(H*W)
  self.assertAlmostEqual(estimate,z.mean(),places=9)
 def test_cached_result_counts(self):
  rows=[x for x in load() if x['target']=='hotspot' and x['method']=='unweighted_STRAT' and ((x['map'],int(x['N'])) in zip('abcdefgh',[8,4,12,2,12,10,11,9]))]
  self.assertEqual(len(rows),8)
  self.assertTrue(all(float(x['CV'])<=.2 for x in rows))
 def test_bias_not_zero_for_original(self):
  rows=[x for x in load() if x['target']=='hotspot' and x['method']=='unweighted_STRAT']
  self.assertGreater(max(abs(float(r['bias_pct'])) for r in rows),15)
if __name__=='__main__':unittest.main(verbosity=2)
