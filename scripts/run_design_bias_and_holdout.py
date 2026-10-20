"""2026-10-19 follow-up. Read-only NAV images, explicitly defined NEW68 sampling design.
Purpose: distinguish design precision from accuracy for full-map proxy target; no parameter tuning.
This is analytical/simulated image-based evaluation, not independent microscopy validation.
"""
import sys, csv, pathlib, math, hashlib, json, itertools
import numpy as np
from scipy.signal import fftconvolve
ROOT=pathlib.Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from final_image_audit import prepare_image,integrals,position_fractions,rounded_side,select_grid,get_pools,boxes
from cv_engine import exact_equal_roi_moments, monte_carlo_equal_roi
OUT=pathlib.Path(__file__).resolve().parent
ns=[8,4,12,2,12,10,11,9]

def avg_variance(values,n):
    pop=np.asarray(values)[np.isfinite(values)]
    v=pop.var(ddof=0)*((len(pop)-n)/(len(pop)-1))/n
    return float(pop.mean()),float(v)

def row_stats(mu,var,target):
    bias=mu-target
    return dict(expected=float(mu),CV=float(math.sqrt(max(0,var))/mu) if mu>0 else None,bias=float(bias),bias_pct=float(100*bias/target) if target!=0 else None,RMSE=float(math.sqrt(max(0,var)+bias*bias)))

def float_integral(arr):
    return np.pad(arr.astype(float),((1,0),(1,0))).cumsum(0).cumsum(1)

def weighted_pools(values,mask,rr,cc):
    return get_pools(values,mask,rr,cc)

def geometry(good,L,H,W,N,grid=None):
    Hp,Wp=good.shape
    if grid is None:rr,cc=select_grid(N,Hp,Wp)
    else:rr,cc=grid
    ys=np.floor(np.arange(rr+1)*Hp/rr).astype(int)
    xs=np.floor(np.arange(cc+1)*Wp/cc).astype(int)
    q=np.zeros((Hp,Wp),dtype=float); K=rr*cc
    for iy in range(rr):
        for ix in range(cc):
            sl=np.s_[ys[iy]:ys[iy+1],xs[ix]:xs[ix+1]]
            qty=int(good[sl].sum())
            if not qty: raise ValueError('empty stratum')
            q[sl]=good[sl]*(N/K/qty)
    assert abs(q.sum()-N)<1e-7
    lamb=fftconvolve(q,np.ones((L,L),dtype=float),mode='full')
    # Eliminate FFT tiny negatives in zero region, though coverage is nonzero everywhere in these images.
    lamb=np.maximum(0,lamb)
    if lamb.shape!=(H,W):raise AssertionError('wrong lambda shape')
    return rr,cc,q,lamb

image_rows=[];comparisons=[];variance_tests={};heldout=[];testmap=['c','g']
for i,fp in enumerate(sorted((ROOT/'inputs').glob('NAV*.png'))):
    mapid='abcdefgh'[i]
    valid,hot,proxy,red=prepare_image(fp)
    H,W=valid.shape;V=int(valid.sum());L=rounded_side(.20*min(valid.shape))
    vi,hi=integrals(valid),integrals(hot)
    prop,good=position_fractions(vi,hi,L)
    true_hot=float(hot.sum()/V)
    target_cont=float(proxy[valid].mean())
    p_int=float_integral(proxy*valid)
    den=boxes(vi,L)
    continuous=np.divide(boxes(p_int,L),den,out=np.zeros_like(prop),where=den>0)
    for N in sorted(set([ns[i],12])):
        rr,cc,q,lamb=geometry(good,L,H,W,N)
        cover=valid & (lamb>1e-8)
        uncovered=int((valid&~cover).sum())
        # per-ROI normal old outputs: strat draws equally from K strata, then one top-left in each selected stratum
        for estimand,field,ref in [('hotspot',prop,true_hot),('continuous_proxy',continuous,target_cont)]:
            ps=get_pools(field,good,rr,cc)
            old=exact_equal_roi_moments(ps,N)
            sval=field[good]
            mu_s, var_s=avg_variance(sval,N)
            m={'map':mapid,'N':N,'target':estimand,'reference':ref,'coverage_all_valid':uncovered==0,'uncovered_valid_pixels':uncovered,'grid':f'{rr}x{cc}','pixel_inclusion_min':float(lamb[valid].min()),'pixel_inclusion_p01':float(np.quantile(lamb[valid],.01))}
            for label,mu,var in [('unweighted_STRAT',old['mean'],old['variance']),('unweighted_SRS',mu_s,var_s)]:
                z=m.copy();z['method']=label;z.update(row_stats(mu,var,ref));comparisons.append(z)
            # HT with intensity correction q(pos) per selected ROI expected exposure lambda(pixel): unbiased map mean if all pixels covered.
            inv=np.divide((hot.astype(float) if estimand=='hotspot' else proxy*valid),lamb,
                          out=np.zeros_like(lamb),where=lamb>1e-8)
            # transformed ROI value for use in mean-of-N exact-moments:
            ht_positions=boxes(float_integral(inv),L)*(N/V)
            corrected=exact_equal_roi_moments(get_pools(ht_positions,good,rr,cc),N)
            assert abs(corrected['mean']-ref)<max(1e-6,1e-3*abs(ref)),(mapid,estimand,corrected['mean'],ref)
            z=m.copy();z['method']='HT_STRAT_pixel_exposure';z.update(row_stats(corrected['mean'],corrected['variance'],ref));comparisons.append(z)
            if mapid in testmap and N==12 and estimand=='hotspot':
                heldout.append({'map':mapid,'test_N':N,'preselected_test':'posthoc_choice after prior all-map explorations',
                               'strat_unweighted_CV':old['CV'],'strat_unweighted_RMSE':row_stats(old['mean'],old['variance'],ref)['RMSE'],
                               'srs_unweighted_CV':row_stats(mu_s,var_s,ref)['CV'],
                               'srs_unweighted_RMSE':row_stats(mu_s,var_s,ref)['RMSE'],
                               'HT_strat_CV':corrected['CV'],'HT_strat_RMSE':row_stats(corrected['mean'],corrected['variance'],ref)['RMSE'],
                               'true_reference_hotspot':ref})
        if N==ns[i]:
            image_rows.append({'map':mapid,'N':N,'roi_side_px':L,'valid_pixels':V,'valid_candidate_positions':int(good.sum()),
                               'grid':f'{rr}x{cc}','all_valid_pixels_covered':uncovered==0,
                               'min_expected_pixel_exposure':float(lamb[valid].min()),'1pct_expected_pixel_exposure':float(np.quantile(lamb[valid],.01)),
                               'original_hotspot_mean':float(old['mean']) if estimand=='hotspot' else '',
                               'image_sha256':hashlib.sha256(fp.read_bytes()).hexdigest()})
        # Monte Carlo verify HT analytic moment for c,g; no after-the-fact parameter changes.
        if mapid in testmap and N==12:
            rr,cc,q,lamb=geometry(good,L,H,W,N)
            inv=np.divide(hot.astype(float),lamb,out=np.zeros_like(lamb),where=lamb>1e-8)
            vals=boxes(float_integral(inv),L)*(N/V)
            pp=get_pools(vals,good,rr,cc)
            exact=exact_equal_roi_moments(pp,N)
            mc=monte_carlo_equal_roi(pp,N,repeats=2500,seed=151223+i,ddof=0)
            variance_tests[mapid]={'N':N,'analytical_mean':exact['mean'],'MC_mean':mc['mean'],
                    'analytical_variance':exact['variance'],'MC_variance':mc['variance'],
                    'relative_variance_error_pct':100*(mc['variance']/exact['variance']-1), 'seed':151223+i,'repeats':2500}

def write_csv(fname,rows):
    with (OUT/fname).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0].keys()));w.writeheader();w.writerows(rows)
write_csv('A_Design_Bias_and_CV_by_Map.csv',comparisons)
write_csv('A_Pixel_Coverage_8_Maps.csv',image_rows)
write_csv('C_Holdout_c_g_Fixed12.csv',heldout)
# Fixed allocation using development-only N selection. Frozen code: each training candidate satisfies 3-point rule for ALL training maps.
vals={}
with (ROOT/'frozen'/'New_Exact_CV_N_248_Rows.csv').open(encoding='utf-8-sig') as f:
    for r in csv.DictReader(f): vals[(r['map'],int(r['N']))]=float(r['CV'])
train=[x for x in 'abcdefgh' if x not in testmap]
fix=next((N for N in range(2,31) if all(max(vals[(tag,n)] for n in [N,N+1,N+2])<=.20 for tag in train)),None)
assert fix==12,fix
calculation={
    'bias_definition':'difference between design-expected sample mean and complete-map hotspot fraction (5% by rank)',
    'estimator_caveat':'unweighted ROI mean is stable for ROI-support distribution, not necessarily complete-map pixel average',
    'correction':'HT inverse expected pixel exposure; each ROI contributes sum(y_pixel/lambda_pixel)/V, coverage includes overlaps in expectation',
    'new_budget_allocation':[8,4,12,2,12,10,11,9],
    'fixed12_selected_using_dev6_ONLY':fix,
    'test_maps':testmap,
    'test_caveat':'Posthoc selection: these maps were previously inspected in earlier all-map tuning; not prospective untouched holdout',
    'monte_carlo_crosscheck':variance_tests,
    'image_coverage_8':image_rows,
    'note':'This stage does not change original manuscript figures or any pixel data.'}
(OUT/'A_C_Validation_Notes.json').write_text(json.dumps(calculation,indent=2,ensure_ascii=False),encoding='utf8')
print('Development-only 6-map fixed N:',fix)
for r in heldout:print('heldout',r)
print('MC check',variance_tests)
print('ALL DONE')
