"""Seeded Monte Carlo spot check of the exact formulas, not new empirical CLSM measurements."""
from pathlib import Path
import json,sys,math
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
from final_image_audit import prepare_image,integrals,position_fractions,rounded_side,select_grid,get_pools
from cv_engine import exact_equal_roi_moments
P=Path(__file__).resolve().parent/'inputs'
report=[]
for tag,ix,N in [('a',1,8),('e',5,12)]:
 im=sorted(P.glob('NAV*.png'))[ix-1]
 valid,hot,_,_=prepare_image(im);L=rounded_side(.2*min(valid.shape));vii,hii=integrals(valid),integrals(hot)
 values,ok=position_fractions(vii,hii,L)
 pool=values[ok];rng=np.random.default_rng(49010+ix)
 rr,cc=select_grid(N,*ok.shape);pools=get_pools(values,ok,rr,cc)
 ex=exact_equal_roi_moments(pools,N)
 # SRSWOR analytical variance
 theor_r=pool.var(ddof=0)/N*(len(pool)-N)/(len(pool)-1)
 for method in ('SRS','STRAT'):
  z=[]
  for _ in range(4500):
   if method=='SRS':
    z.append(float(np.mean(pool[rng.choice(len(pool),size=N,replace=False)])))
   else:
    strata=rng.choice(len(pools),size=N,replace=False)
    z.append(float(np.mean([pools[k][rng.integers(len(pools[k]))] for k in strata])))
  z=np.array(z)
  v=theor_r if method=='SRS' else ex['variance']
  error=(z.var(ddof=0)-v)/v
  assert abs(error)<0.11,(tag,method,error)
  report.append({'map':tag,'N':N,'method':method,'mc_replicates':len(z),'seed':49010+ix,'variance_MC':float(z.var(ddof=0)),'variance_exact':float(v),'relative_difference_pct':float(100*error)})
print(json.dumps(report,indent=2))
(Path(__file__).resolve().parent/'results/MonteCarlo_Spot_Check.json').write_text(json.dumps(report,indent=2))
