"""Independent fixed-seed Monte Carlo validation of the NEW explicit design only.
Not historical Monte Carlo run or experimental repetitions.
"""
from pathlib import Path
from PIL import Image
import sys,csv,json
from final_image_audit import prepare_image,integrals,position_fractions,rounded_side,select_grid,get_pools
from cv_engine import exact_equal_roi_moments,monte_carlo_equal_roi
root=Path(__file__).parent
files=sorted((root/'inputs/01_CLSM_navigation_originals').glob('NAV*.png'))
summary=list(csv.DictReader((root/'new_results/New_Image_Summary_8_Maps.csv').open(encoding='utf-8-sig')))
res=[]
for i,f in enumerate(files):
 valid,hot,_,_=prepare_image(f);h,w=valid.shape;L=rounded_side(.2*min(h,w));x,ok=position_fractions(integrals(valid),integrals(hot),L)
 for lbl,N in [('fixed_9',9),('fixed_12',12),('adaptive',int(summary[i]['selected_N']))]:
  rows,cols=select_grid(N,*ok.shape);pools=get_pools(x,ok,rows,cols)
  exact=exact_equal_roi_moments(pools,N)
  mc=monte_carlo_equal_roi(pools,N,repeats=2000,seed=20261019+100*i+N,ddof=0)
  res.append({'map':chr(97+i),'scheme':lbl,'N':N,'exact_CV':exact['CV'],'MC_CV':mc['CV'],'MC_minus_exact':mc['CV']-exact['CV'],'seed':mc['seed'],'repeats':mc['repeats'],'ddof':mc['ddof']})
  print(chr(97+i),lbl,round(exact['CV'],5),round(mc['CV'],5),flush=True)
with (root/'new_results/NEW_MC_Verification_24.csv').open('w',encoding='utf-8-sig',newline='') as f:
 wr=csv.DictWriter(f,fieldnames=list(res[0]));wr.writeheader();wr.writerows(res)
print('MAX abs diff',max(abs(z['MC_minus_exact']) for z in res),'mean abs diff',sum(abs(z['MC_minus_exact']) for z in res)/len(res))
