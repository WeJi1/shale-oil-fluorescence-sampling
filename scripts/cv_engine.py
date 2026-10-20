"""NEW audit implementation, NOT recovered historical image-analysis code.
No image segmentation/grid/overlap defaults are supplied.
The caller must supply valid/hotspot masks and explicit per-stratum ROI positions.
Exact moments assume equal ROI weights, uniform SRSWOR strata, independent
uniform position draws conditional on selected strata. See Supplementary_Methods_EN.md.
"""
import itertools, math
import numpy as np

def repeated_cv(estimates, *, ddof):
    a=np.asarray(estimates,dtype=float)
    if a.ndim!=1 or len(a)<=ddof or ddof not in (0,1) or not np.all(np.isfinite(a)) or a.mean()<=0:
        raise ValueError('Need finite 1D estimates, positive mean, explicit ddof=0 or 1')
    return float(a.std(ddof=ddof)/a.mean())

def _pools(pools,N):
    p=[np.asarray(x,dtype=float) for x in pools]
    if not 1<=N<=len(p) or any(x.ndim!=1 or len(x)==0 or not np.all(np.isfinite(x)) for x in p):
        raise ValueError('Invalid N or empty/nonfinite stratum')
    return p

def exact_equal_roi_moments(pools,N):
    """Finite uniform positions, N strata sampled without replacement from K.
    Xhat = arithmetic mean of N ROI fractions. Not a pooled-area ratio.
    Between-stratum term includes randomness in stratum selection.
    """
    p=_pools(pools,N);K=len(p)
    mu=np.array([x.mean() for x in p]);v=np.array([x.var(ddof=0) for x in p])
    mean=float(mu.mean())
    within=float(v.mean()/N)
    between=0. if K==1 else float((K-N)/(N*(K-1))*mu.var(ddof=0))
    var=within+between
    return dict(mean=mean,variance=var,within_variance=within,between_variance=between,
                CV=float(np.sqrt(var)/mean) if mean>0 else None)

def monte_carlo_equal_roi(pools,N,*,repeats,seed,ddof):
    """Explicit NEW simulation settings; not historical seeds or run lengths."""
    p=_pools(pools,N)
    if not isinstance(repeats,int) or repeats<2:raise ValueError('repeats >= 2 required')
    if seed is None:raise ValueError('Explicit audit seed required')
    rng=np.random.Generator(np.random.PCG64(seed));out=np.empty(repeats)
    for b in range(repeats):
        selected=rng.choice(len(p),size=N,replace=False)
        out[b]=np.mean([p[h][rng.integers(len(p[h]))] for h in selected])
    return dict(mean=float(out.mean()),variance=float(out.var(ddof=ddof)),
                CV=repeated_cv(out,ddof=ddof),repeats=repeats,seed=seed,
                generator='NumPy PCG64',ddof=ddof)

def exact_enumeration(pools,N,*,max_states,valid_area_pools=None):
    """Small-space independent cross-check.
    If valid_area_pools is given, enumerate sum(A*X)/sum(A) rather than mean(X).
    Each subset has probability 1/C(K,N); conditional locations are uniform.
    Unequal pool sizes must NOT give subsets unequal probability.
    """
    p=_pools(pools,N);K=len(p);subs=list(itertools.combinations(range(K),N))
    states=sum(math.prod(len(p[h]) for h in S) for S in subs)
    if states>max_states:raise ValueError('Exact enumeration exceeds explicit state limit')
    areas=None
    if valid_area_pools is not None:
        areas=[np.asarray(x,float) for x in valid_area_pools]
        if len(areas)!=K or any(a.shape!=x.shape or np.any(a<=0) or not np.all(np.isfinite(a)) for a,x in zip(areas,p)):
            raise ValueError('Positive valid areas must match each pool')
    m1=m2=0.
    for S in subs:
        prob=1/(len(subs)*math.prod(len(p[h]) for h in S))
        for ids in itertools.product(*(range(len(p[h])) for h in S)):
            xs=np.array([p[h][i] for h,i in zip(S,ids)])
            if areas is None:z=float(xs.mean())
            else:
                aa=np.array([areas[h][i] for h,i in zip(S,ids)])
                z=float(np.dot(xs,aa)/aa.sum())
            m1+=prob*z;m2+=prob*z*z
    var=max(0.,m2-m1*m1)
    return dict(mean=m1,variance=var,CV=math.sqrt(var)/m1 if m1>0 else None,states=states)

def roi_fractions(valid,hotspot,*,side_px,stratum_positions,min_valid_fraction):
    """Read supplied boolean masks; no masking, thresholding or grids are invented.
    positions: one explicit list of (top,left) integer coordinates per stratum.
    No overlap rejection is performed: caller's position law must permit independent
    draws. If non-overlap is mandatory, use a joint-design implementation instead.
    """
    valid=np.asarray(valid);hotspot=np.asarray(hotspot)
    if valid.dtype!=bool or hotspot.dtype!=bool or valid.shape!=hotspot.shape or valid.ndim!=2:
        raise ValueError('Explicit same-size 2D bool masks required')
    if np.any(hotspot & ~valid):raise ValueError('Hotspots outside valid mask')
    if not isinstance(side_px,int) or not 1<=side_px<=min(valid.shape):raise ValueError('Explicit integer ROI side required')
    if not 0<min_valid_fraction<=1:raise ValueError('Explicit valid-area fraction required')
    def integral(a):return np.pad(a.astype(np.int64),((1,0),(1,0))).cumsum(0).cumsum(1)
    vi,hi=integral(valid),integral(hotspot);L=side_px;H,W=valid.shape
    def total(ii,t,l):return int(ii[t+L,l+L]-ii[t,l+L]-ii[t+L,l]+ii[t,l])
    pools=[];areas=[]
    for positions in stratum_positions:
        pos=[tuple(q) for q in positions]
        if len(set(pos))!=len(pos):raise ValueError('Duplicate positions would change sampling probabilities')
        v=[];a=[]
        for t,l in pos:
            if not isinstance(t,(int,np.integer)) or not isinstance(l,(int,np.integer)) or not(0<=t<=H-L and 0<=l<=W-L):raise ValueError('Invalid top-left position')
            den=total(vi,t,l)
            if den/(L*L)<min_valid_fraction:continue
            a.append(den);v.append(total(hi,t,l)/den)
        if not v:raise ValueError('Stratum has no admissible ROI: do not silently delete or reweight')
        pools.append(np.array(v));areas.append(np.array(a))
    return pools,areas
