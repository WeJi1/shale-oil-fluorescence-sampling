"""Independent final fair-budget comparison of archive image-derived hotspot fraction.

Read-only inputs: eight NAV*.png. Conditional retrospective, not a prospective experiment.
Fixed preprocessing and geometrical positions are imported directly from frozen NEW68 image audit.
This comparator does NOT optimize masks, grid, or N using performance comparisons.
"""
import csv, json, math, itertools, hashlib
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
from final_image_audit import prepare_image,integrals,position_fractions,rounded_side,select_grid,get_pools
from cv_engine import exact_equal_roi_moments, monte_carlo_equal_roi

ROOT=Path(__file__).resolve().parent
IMAGES=ROOT/'inputs'
OUTPUT=ROOT/'results'
OUTPUT.mkdir(parents=True,exist_ok=True)
ADAPT=[8,4,12,2,12,10,11,9]
BUDGET=sum(ADAPT)
assert BUDGET==68

def summarize(mu,var,truth):
 bias=mu-truth
 return {'mean':float(mu),'variance':float(var),'CV':float(math.sqrt(max(0.,var))/mu),'bias':float(bias), 'relative_bias_pct':float(bias/truth*100),'RMSE':float(math.sqrt(max(0.,var)+bias*bias)), 'relative_RMSE_pct':float(math.sqrt(max(0.,var)+bias*bias)/truth*100)}

def contrast_random(values, n):
 mu=float(values.mean());v=float(values.var(ddof=0))
 # uniform without replacement from the finite population of allowed top-left positions.
 var=v/n*((len(values)-n)/(len(values)-1))
 return mu,var

def dumps(name, rows):
 with (OUTPUT/name).open('w',newline='',encoding='utf-8-sig') as fp:
  wr=csv.DictWriter(fp,fieldnames=list(rows[0].keys()));wr.writeheader();wr.writerows(rows)

rows=[];permap={}
for i,p in enumerate(sorted(IMAGES.glob('NAV*.png'))):
 tag='abcdefgh'[i];valid,hot,proxy,red=prepare_image(p)
 truth=float(hot.sum()/valid.sum())
 L=rounded_side(.20*min(valid.shape));vii,hii=integrals(valid),integrals(hot)
 val,good=position_fractions(vii,hii,L)
 population=val[good]
 assert population.size>12
 permap[tag]={}
 for n in sorted({ADAPT[i],8,9,12}):
  # Simple random position sampling without replacement. Rectangular ROIs may overlap.
  mu_r,var_r=contrast_random(population,n)
  a=summarize(mu_r,var_r,truth)
  rr,cc=select_grid(n,*good.shape)
  pools=get_pools(val,good,rr,cc)
  exact=exact_equal_roi_moments(pools,n)
  b=summarize(exact['mean'],exact['variance'],truth)
  a.update({'map':tag,'image':p.name,'map_hotspot_reference':truth,'roi_side_px':L,'allowed_positions':int(len(population)),'N':n,'method':'SRS_positions_SRSWOR','grid':'none','strata_K':None})
  b.update({'map':tag,'image':p.name,'map_hotspot_reference':truth,'roi_side_px':L,'allowed_positions':int(len(population)),'N':n,'method':'STRAT_NEW68','grid':f'{rr}x{cc}','strata_K':rr*cc})
  rows.extend([a,b]);permap[tag][(n,'SRS')]=a;permap[tag][(n,'STRAT')]=b
  if n==ADAPT[i]:
   print(tag,'N',n,'ref',round(truth,6),'SRS cv',round(a['CV'],4),'ST cv',round(b['CV'],4),'SRS relRMSE',round(a['relative_RMSE_pct'],2),'ST relRMSE',round(b['relative_RMSE_pct'],2),flush=True)

dumps('Method_Comparisons_By_Map.csv',rows)
# compare pairs at exactly the SAME map-wise N as NEW68 (same total 68 ROIs)
paired=[]
for i,tag in enumerate('abcdefgh'):
 n=ADAPT[i];s=permap[tag][(n,'SRS')];t=permap[tag][(n,'STRAT')]
 paired.append({'map':tag,'N':n,'hotspot_reference':s['map_hotspot_reference'],
                'SRS_CV':s['CV'],'STRAT_CV':t['CV'],'STRAT_minus_SRS_CV':t['CV']-s['CV'],
                'SRS_relative_bias_pct':s['relative_bias_pct'],'STRAT_relative_bias_pct':t['relative_bias_pct'],
                'SRS_RMSE':s['RMSE'],'STRAT_RMSE':t['RMSE'],'STRAT_minus_SRS_RMSE':t['RMSE']-s['RMSE'],
                'SRS_relative_RMSE_pct':s['relative_RMSE_pct'],'STRAT_relative_RMSE_pct':t['relative_RMSE_pct'],
                'SRS_mean':s['mean'],'STRAT_mean':t['mean']})
dumps('Matched_N_SRS_vs_STRAT_8_Maps.csv',paired)

# Balanced equal global budget: 8 maps each start N=8; four of eight get 1 extra.
# All 70 assignments are enumerated equally (prevents selective allocation of extra ROIs).
assignments=[]
for subset in itertools.combinations('abcdefgh',4):
 d={'assignment':''.join(subset)}
 for method in ('SRS','STRAT'):
  selected=[permap[tag][(9 if tag in subset else 8,method)] for tag in 'abcdefgh']
  d[f'{method}_mean_CV']=float(np.mean([x['CV'] for x in selected]))
  d[f'{method}_map_RMSE_root_mean_MSE']=float(math.sqrt(np.mean([x['RMSE']**2 for x in selected])))
  d[f'{method}_mean_RMSE']=float(np.mean([x['RMSE'] for x in selected]))
  d[f'{method}_mean_abs_bias']=float(np.mean([abs(x['bias']) for x in selected]))
  d[f'{method}_pass_CV20']=sum(x['CV']<=.20+1e-12 for x in selected)
 assignments.append(d)
dumps('Balanced_Budget_70_Assignments.csv',assignments)

def aggregate(label,selected):
 return {'strategy':label,'total_ROIs':sum(x['N'] for x in selected),'mean_CV':float(np.mean([x['CV'] for x in selected])),
         'pass_CV20':sum(x['CV']<=0.20+1e-12 for x in selected),'maps':len(selected),
         'mean_signed_bias':float(np.mean([x['bias'] for x in selected])),
         'mean_abs_bias':float(np.mean([abs(x['bias']) for x in selected])),
         'pooled_RMSE':float(math.sqrt(np.mean([x['RMSE']**2 for x in selected]))),
         'mean_map_RMSE':float(np.mean([x['RMSE'] for x in selected])),
         'mean_relative_RMSE_pct':float(np.mean([x['relative_RMSE_pct'] for x in selected]))}
summary=[]
for method in ('SRS','STRAT'):
 summary.append(aggregate(f'{method} mapwise N=NEW68 (68 ROI)',[permap[tag][(ADAPT[i],method)] for i,tag in enumerate('abcdefgh')]))
 summary.append(aggregate(f'{method} fixed N=12 (96 ROI)',[permap[tag][(12,method)] for tag in 'abcdefgh']))
 a=[x for x in assignments]
 summary.append({'strategy':f'{method} balanced 68 (average over 70 budget splits)','total_ROIs':68,'maps':8,
                 'mean_CV':float(np.mean([x[f'{method}_mean_CV'] for x in a])),
                 'pass_CV20':float(np.mean([x[f'{method}_pass_CV20'] for x in a])),
                 'mean_signed_bias':None,
                 'mean_abs_bias':float(np.mean([x[f'{method}_mean_abs_bias'] for x in a])),
                 'pooled_RMSE':float(math.sqrt(np.mean([x[f'{method}_map_RMSE_root_mean_MSE']**2 for x in a]))),
                 'mean_map_RMSE':float(np.mean([x[f'{method}_mean_RMSE'] for x in a])),
                 'mean_relative_RMSE_pct':None})
summary=sorted(summary,key=lambda d:d['strategy'])
dumps('Comparison_Summary.csv',summary)
# Statistical exact formula validation. Recreate previously frozen selected-N/CVs to machine precision.
saved={}
with open(ROOT/'frozen'/'New_Exact_CV_N_248_Rows.csv',encoding='utf-8-sig',newline='') as f:
 for r in csv.DictReader(f): saved[(r['map'],int(r['N']))]=float(r['CV'])
maxdev=max(abs(r['CV']-saved[(r['map'],r['N'])]) for r in rows if r['method']=='STRAT_NEW68')
assert maxdev<1e-11,maxdev
for i,x in enumerate(paired):
 assert x['N']==ADAPT[i]
assert len(assignments)==70
meta={'input_source':'8 NAV PNG files, frozen NEW68 image transformation','proof_status':'Conditional exact finite-design calculations; NOT measured experiment, NOT external validation','estimand':'Image-derived top-5% hotspot fraction from full navigation map','roi_size':'0.2*min(image height,image width), rounded half-up','random_sampling':'SRSWOR allowed top-left positions with ROI overlaps possible','stratified_sampling':'Same NEW68 near-square grids; one position per selected stratum; ROI overlap allowed','within_map_adaptive_definition':'Identical to stratified conditional design at the same N; only N_i changes by map','equal_total_budget_baseline':'Four maps N=9, four N=8; all C(8,4)=70 assignments averaged without cherry-picking','fixed_budget':'68 ROI for main comparison; 96 ROI fixed12 shown contextual only','full_map_reference_limitation':'5% hotspot fraction is defined by image rank, not independent chemical ground truth','max_strat_CV_reproduction_abs_diff':maxdev,
      'sha256_images':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(IMAGES.glob('NAV*.png'))}}
(OUTPUT/'PRESET_METHODS_AND_QC.json').write_text(json.dumps(meta,indent=2,ensure_ascii=False),encoding='utf8')
(OUTPUT/'RESULTS.json').write_text(json.dumps({'paired':paired,'summary':summary,'reproduction_CV_max_error':maxdev},indent=2,ensure_ascii=False),encoding='utf8')
print('SUMMARY')
for x in summary:print(x)
print('Maximum reproduced NEW68 exact CV abs deviation',maxdev)
