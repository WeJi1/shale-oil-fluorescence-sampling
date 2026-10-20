"""INDEPENDENT 2026-10-19 proposal-based image audit, not recovered historical algorithm.
This program contains no stored old CV target values and never reads workbook outputs.
Parameters fixed before image-to-workbook comparisons. Details in PARAMS.json.
"""
from __future__ import annotations
import argparse,csv,hashlib,json,math,os,sys
from pathlib import Path
import numpy as np
from PIL import Image
from cv_engine import exact_equal_roi_moments

SCALES=[round(k/10,1) for k in range(1,10)]
RANGE_N=list(range(2,33))
PARAMS={
 'audit_status':'NEW fully specified conditional image algorithm; NOT historical recovery',
 'image_color':'input PNG RGBA, decoded sRGB 8-bit channel values; no color calibration',
 'valid_pixel':'alpha > 0; exclude probable saturated pure-red annotation by R>=180,G<=80,B<=80,R>2G,R>2B; preserve whites and dark minerals',
 'proxy':'max(0,G-B)*(max(R,G,B)/255)**2; no gamma linearization, no denoising or smoothing',
 'hotspot':'rank exactly ceil(0.05*valid_pixel_count) eligible pixels by proxy descending; ties broken by raster index ascending via stable argsort',
 'L0':'min(full_input_image_height,full_input_image_width) before masking',
 'roi_width':'max(1, floor(relative_length * L0+0.5)); same for height (round half up)',
 'valid_area':'ROI accepted iff valid_pixels >= 0.90*total_roi_pixels; hotspot fraction = hotspot_pixels/valid_pixels',
 'top_left_positions':'all integer one-pixel increment top-left positions with full square ROI inside complete image and >=90% valid area',
 'multiscale':'for L/L0=0.1..0.9, exact population CV (ddof 0) of hotspot fractions over all admissible window top-lefts, E > 0',
 'representative':'smallest scale whose CV<=0.20 and all larger evaluated scales also CV<=0.20; else >0.9',
 'strata_grid':'for each N choose nrow from 1..ceil(sqrt(N*H_positions/W_positions))+2 and ncol=ceil(N/nrow), minimize score 0.5*aspect_deviation + 0.5*(K-N)/N with aspect_deviation=abs(log((Hpositions/nrow)/(Wpositions/ncol))), subject K>=N; tie prefer smaller K then smaller nrow',
 'stratum_definition':'regular intervals of the rectangle of possible top-left positions, split by floor(index*size/nparts); a position belongs to one stratum by its TOP-LEFT only; ROI may cross strata',
 'stratum_pick':'choose N of K nonempty equal-probability strata without replacement (SRSWOR); each selected stratum independently draws one allowed position uniformly',
 'overlap':'allowed including shared pixels, no rejection, independent conditional positions; non-nested candidate N',
 'group_estimator':'arithmetic mean of per-ROI hotspot fractions, each ROI weight 1/N (NOT union, NOT pooled area, NOT stratum-area-weighted)',
 'cv_exact':'sqrt(within plus stratum-selection variance) / expected estimate as in cv_engine.py; no Monte Carlo seed',
 'N_domain':'N=2..32, candidate first N in 2..30 for which exact CV at N,N+1,N+2 all<=0.20',
 'fixed_N':'uniform minimum N satisfying same triple CV acceptance across all eight maps; 2..30',
 'source_data':'8 individually archived original navigation PNGs; no historical mask/position/seed recovered',
 'scientific_limitation':'top-5%-by-rank hotspot and near-white/black/annotation assumptions are new definitions; not chemical truth'
}

def integrals(mask):
 a=mask.astype(np.int64)
 return np.pad(a,((1,0),(1,0))).cumsum(axis=0).cumsum(axis=1)
def boxes(ii,L):
 # exact valid counts over all LxL windows indexed by top-left positions
 return ii[L:,L:] - ii[:-L,L:] - ii[L:,:-L] + ii[:-L,:-L]
def rounded_side(x):return max(1,int(math.floor(x+0.5)))

def prepare_image(path):
 rgba=np.asarray(Image.open(path).convert('RGBA'));rgb=rgba[:,:,:3]
 r,g,b=(rgb[:,:,i].astype(float) for i in range(3))
 red=(r>=180)&(g<=80)&(b<=80)&(r>2*g)&(r>2*b)
 valid=(rgba[:,:,3]>0)&~red
 brightness=np.max(rgb,axis=2)/255.
 p=np.maximum(0.,g-b)*brightness**2
 ix=np.flatnonzero(valid.ravel());n=int(math.ceil(.05*len(ix)))
 # stable tie breaking by row-major source order, deterministic
 selected=ix[np.argsort(-p.ravel()[ix],kind='stable')[:n]]
 hot=np.zeros(valid.size,dtype=bool);hot[selected]=True
 return valid,hot.reshape(valid.shape),p,red

def position_fractions(vii,hii,L,threshold=.9):
 v=boxes(vii,L);h=boxes(hii,L)
 ok=(v >= .9*L*L)
 data=np.divide(h,v,out=np.zeros(v.shape,dtype=float),where=v>0)
 return data,ok

def select_grid(N,Hp,Wp):
 opts=[]
 for rows in range(1,N+1):
  cols=math.ceil(N/rows)
  K=rows*cols
  aspect=abs(math.log((Hp/rows)/(Wp/cols)))
  # strong but balanced bias toward a nearly square physical cell and limited empty slots
  score=.5*aspect+.5*(K-N)/N
  opts.append((score,K,rows,cols))
 return sorted(opts)[0][2:4]

def get_pools(data,ok,rows,cols):
 Hp,Wp=ok.shape
 ys=np.floor(np.arange(rows+1)*Hp/rows).astype(int)
 xs=np.floor(np.arange(cols+1)*Wp/cols).astype(int)
 pools=[]
 for iy in range(rows):
  for ix in range(cols):
   y0,y1=ys[iy:iy+2];x0,x1=xs[ix:ix+2]
   z=data[y0:y1,x0:x1][ok[y0:y1,x0:x1]]
   if not len(z):raise ValueError(f'Empty stratum ({iy},{ix}) for grid {rows}x{cols}')
   pools.append(z)
 return pools

def run(paths,out):
 out=Path(out);out.mkdir(parents=True,exist_ok=True)
 with (out/'PARAMS.json').open('w',encoding='utf8') as f:json.dump(PARAMS,f,ensure_ascii=False,indent=2)
 scale_rows=[];nr_rows=[];summary=[];inputs=[]
 for im_id,path in enumerate(paths):
  valid,hot,p,red=prepare_image(path); H,W=valid.shape;L0=min(H,W)
  sha=hashlib.sha256(Path(path).read_bytes()).hexdigest()
  vii,hii=integrals(valid),integrals(hot)
  item={'map':chr(97+im_id),'image_file':Path(path).name,'sha256':sha,'width':W,'height':H,'valid_pixels':int(valid.sum()),'annotation_pixels_excluded':int(red.sum()),'hotspot_pixels':int(hot.sum())}
  print('BEGIN',chr(97+im_id),W,H,'valid',int(valid.sum()),'hot',int(hot.sum()),flush=True)
  inputs.append(item)
  cvsc=[]
  for rel in SCALES:
   L=rounded_side(rel*L0);data,ok=position_fractions(vii,hii,L)
   z=data[ok]
   if not len(z) or z.mean()<=0:cv=float('nan');mu=float('nan')
   else:mu=float(z.mean());cv=float(z.std(ddof=0)/mu)
   scale_rows.append(dict(map=chr(97+im_id),scale=rel,side_px=L,valid_positions=int(len(z)),mean_fraction=mu,CV=cv))
   cvsc.append(cv)
  qualifies=[k for k in range(9) if all(np.isfinite(cvsc[j]) and cvsc[j]<=.20 for j in range(k,9))]
  rep=SCALES[qualifies[0]] if qualifies else '>0.9'
  L=rounded_side(.2*L0);data,ok=position_fractions(vii,hii,L)
  if not ok.any():raise RuntimeError(f'{path.name}: no valid ROI positions')
  curr=[]
  for N in RANGE_N:
   rr,cc=select_grid(N,*ok.shape)
   pools=get_pools(data,ok,rr,cc)
   res=exact_equal_roi_moments(pools,N)
   val=res['CV']
   nr_rows.append(dict(map=chr(97+im_id),N=N,rows=rr,cols=cc,K=rr*cc,roi_side_px=L,available_positions=int(ok.sum()),CV=float(val),design_mean=float(res['mean']),variance=float(res['variance'])))
   curr.append(val)
  req=next((N for N in range(2,31) if max(curr[N-2:N+1])<=.20),None)
  item.update({'representative_L_L0':rep,'roi_size_px':L,'selected_N':req,'CV_at_9':curr[9-2],'CV_at_12':curr[12-2], 'scaleCV_at_0.4':cvsc[3]})
  summary.append(item)
  print('DONE',chr(97+im_id),'rep',rep,'N',req,'CV9',round(item['CV_at_9'],4),'CV12',round(item['CV_at_12'],4),flush=True)
 def dump(name,rows):
  with (out/name).open('w',newline='',encoding='utf-8-sig') as f:
   wr=csv.DictWriter(f,fieldnames=rows[0].keys());wr.writeheader();wr.writerows(rows)
 dump('New_Multiscale_CV_72_Rows.csv',scale_rows)
 dump('New_Exact_CV_N_248_Rows.csv',nr_rows)
 dump('New_Image_Summary_8_Maps.csv',summary)
 with (out/'Inputs_Sha256.json').open('w',encoding='utf8') as f:json.dump(inputs,f,ensure_ascii=False,indent=2)
 # fixed count using one common N and N+1, N+2. no historical workbook dependence
 fixed=next((N for N in range(2,31) if all(max([r['CV'] for r in nr_rows if r['map']==k and r['N'] in (N,N+1,N+2)])<=.20 for k in 'abcdefgh')),None)
 total=sum(z['selected_N'] for z in summary if z['selected_N'] is not None)
 result={'new_fixed_N':fixed,'new_adaptive_N':[s['selected_N'] for s in summary], 'new_total_adaptive_ROIs':total if all(s['selected_N'] is not None for s in summary) else None,'fixed_total_ROIs':fixed*8 if fixed else None,'percent_count_reduction':100*(1-total/(8*fixed)) if fixed and all(s['selected_N'] is not None for s in summary) else None}
 with (out/'NEW_Allocation_Summary.json').open('w',encoding='utf8') as f:json.dump(result,f,indent=2)
 print('RESULT',result,flush=True)
 return result

if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--images',required=True);parser.add_argument('--out',required=True);a=parser.parse_args()
 run(sorted(Path(a.images).glob('NAV*.png')),a.out)
