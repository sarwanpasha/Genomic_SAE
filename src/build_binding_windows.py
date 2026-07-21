#!/usr/bin/env python
"""
Build labeled window sets for the bound-vs-unbound binding test.

For each TF, produce 3 classes of genomic windows (each WINDOW bp):
  bound          : centered on a ChIP-seq peak that contains a strong motif match
  unbound_motif  : strong motif match NOT in any peak
  background     : random windows (no strong motif requirement)

Writes binding_windows_<TF>.tsv with columns: class, chrom, start, end
CPU-only (genome motif scan). Run on login node or CPU job.

Env: NPER (windows per class, default 5000)  WINDOW (default 200)  SEED (0)
"""
import os, json, random, numpy as np
from pyfaidx import Fasta

ROOT = os.environ.get("GENOMIC_SAE_ROOT", os.path.dirname(os.path.abspath(__file__)))
NPER=int(os.environ.get("NPER",5000)); WINDOW=int(os.environ.get("WINDOW",200))
SEED=int(os.environ.get("SEED",0))
random.seed(SEED); np.random.seed(SEED)

TFS={
 "CTCF": dict(db=f"{ROOT}/jaspar2024_core_vertebrates.txt", motif="CTCF",
              peaks=f"{ROOT}/ctcf_gm12878_idr.bed"),
 "GATA1":dict(db=f"{ROOT}/jaspar2024_core_vertebrates.txt", motif="GATA1",
              peaks=f"{ROOT}/gata1_k562_idr.bed"),
}
MAIN=[f"chr{c}" for c in list(range(1,23))+["X","Y"]]
fa=Fasta(f"{ROOT}/hg38.fa")
chrom_len={c:len(fa[c]) for c in MAIN if c in fa}
B2I={"A":0,"C":1,"G":2,"T":3}
def gc_frac(s):
    s=s.upper(); n=len(s)
    return (s.count('G')+s.count('C'))/n if n else 0.0

def load_pwm(path,name):
    blocks=[];hdr=None;rows=[]
    for ln in open(path):
        ln=ln.rstrip("\n")
        if ln.startswith(">"):
            if hdr: blocks.append((hdr,rows))
            hdr=ln;rows=[]
        elif ln.strip():
            nums=ln.replace("A","").replace("C","").replace("G","")\
                   .replace("T","").replace("[","").replace("]","").split()
            rows.append([float(x) for x in nums])
    if hdr: blocks.append((hdr,rows))
    for h,r in blocks:
        if name.upper() in h.upper():
            pfm=np.array(r,float)
            if pfm.shape[0]==4: pfm=pfm.T
            pfm=pfm+0.25; ppm=pfm/pfm.sum(1,keepdims=True)
            return np.log2(ppm/0.25)
    raise RuntimeError(f"{name} not found")

def pwm_max_threshold(pwm, q=0.85):
    """A 'strong' match = score above q fraction of the max possible score."""
    return q * pwm.max(1).sum()

def load_peaks(path):
    P={}
    for ln in open(path):
        p=ln.split("\t"); P.setdefault(p[0],[]).append((int(p[1]),int(p[2])))
    for c in P: P[c].sort()
    return P
import bisect
def in_peak(P,chrom,pos):
    arr=P.get(chrom,[])
    if not arr: return False
    starts=[a[0] for a in arr]
    i=bisect.bisect_right(starts,pos)-1
    return i>=0 and arr[i][0]<=pos<=arr[i][1]

def _encode(seq):
    return np.frombuffer(seq.upper().encode().translate(
        bytes.maketrans(b"ACGT", bytes([0,1,2,3]))), dtype=np.uint8)

def has_motif(seq, pwm, thr):
    """True if any window position scores >= thr (+ strand). Vectorized."""
    L=pwm.shape[0]; arr=_encode(seq); n=len(arr)
    if n<L: return False
    idx=np.arange(n-L+1)[:,None]+np.arange(L)[None,:]
    W=arr[idx]
    ok=(W<=3).all(1)
    if not ok.any(): return False
    scores=pwm[np.arange(L)[None,:], W].sum(1)
    scores=np.where(ok, scores, -1e9)
    return bool((scores>=thr).any())

for tf,info in TFS.items():
    if not os.path.exists(info["peaks"]):
        print(f"[skip] {tf}: no peaks file"); continue
    pwm=load_pwm(info["db"], info["motif"])
    thr_bound=pwm_max_threshold(pwm,0.80)
    thr_unbound=pwm_max_threshold(pwm, 0.50 if tf in ("REST",) else 0.65)
    P=load_peaks(info["peaks"])
    print(f"[{tf}] PWM len {pwm.shape[0]} thr_b={thr_bound:.2f} thr_u={thr_unbound:.2f}; peaks={sum(len(v) for v in P.values())}",
          flush=True)

    # ---- bound: peak centers that contain a strong motif ----
    bound=[]
    peak_list=[(c,s,e) for c in P for (s,e) in P[c] if c in chrom_len]
    random.shuffle(peak_list)
    for c,s,e in peak_list:
        if len(bound)>=NPER: break
        center=(s+e)//2
        a=center-WINDOW//2; b=center+WINDOW//2
        if a<0 or b>chrom_len[c]: continue
        sub=str(fa[c][a:b]).upper()
        if set(sub)<=set("ACGT") and len(sub)==WINDOW:
            # require a strong motif somewhere in the window
            hits=has_motif(sub, pwm, thr_bound)
            if hits: bound.append((c,a,b))
    print(f"[{tf}] bound windows: {len(bound)}", flush=True)

    # ---- bound GC distribution (target for matching) ----
    GCB=np.linspace(0,1,21)
    bound_gc=np.array([gc_frac(str(fa[c][a:b])) for c,a,b in bound]) if bound else np.array([0.5])
    bound_hist,_=np.histogram(bound_gc, bins=GCB, density=True)
    bound_hist=bound_hist/(bound_hist.max()+1e-9)

    # ---- unbound_motif: strong motif NOT in peak, GC-matched to bound ----
    unbound=[]
    tries=0
    while len(unbound)<NPER and tries<NPER*400:
        tries+=1
        c=random.choice(list(chrom_len))
        a=random.randint(0, chrom_len[c]-WINDOW); b=a+WINDOW
        center=(a+b)//2
        if in_peak(P,c,center): continue
        sub=str(fa[c][a:b]).upper()
        if not (set(sub)<=set("ACGT") and len(sub)==WINDOW): continue
        if not has_motif(sub, pwm, thr_unbound): continue
        gb=min(np.digitize(gc_frac(sub), GCB)-1, len(bound_hist)-1)
        if random.random() <= bound_hist[max(0,gb)]:
            unbound.append((c,a,b))
    ug=np.array([gc_frac(str(fa[c][a:b])) for c,a,b in unbound]) if unbound else np.array([0])
    print(f"[{tf}] unbound-motif windows: {len(unbound)} (tries={tries}) "
          f"GC bound={bound_gc.mean():.3f} unbound={ug.mean():.3f}", flush=True)

    # ---- background: random, no motif requirement ----
    bg=[]
    while len(bg)<NPER:
        c=random.choice(list(chrom_len))
        a=random.randint(0, chrom_len[c]-WINDOW); b=a+WINDOW
        sub=str(fa[c][a:b]).upper()
        if set(sub)<=set("ACGT") and len(sub)==WINDOW:
            bg.append((c,a,b))
    print(f"[{tf}] background windows: {len(bg)}", flush=True)

    out=f"{ROOT}/binding_windows_{tf}.tsv"
    with open(out,"w") as f:
        f.write("class\tchrom\tstart\tend\n")
        for c,a,b in bound:   f.write(f"bound\t{c}\t{a}\t{b}\n")
        for c,a,b in unbound: f.write(f"unbound_motif\t{c}\t{a}\t{b}\n")
        for c,a,b in bg:      f.write(f"background\t{c}\t{a}\t{b}\n")
    print(f"[{tf}] wrote {out}\n", flush=True)

print("[done]")
