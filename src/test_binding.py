#!/usr/bin/env python
"""
Bound-vs-unbound binding test on SAE features.
Loads bindingfeat_<MODEL>_<TF>_layer<LAYER>.npz and, per feature, tests:
  A) bound vs unbound_motif   (does feature encode binding beyond motif?)
  B) unbound_motif vs background (is feature motif-selective?)
Reports feature categories and the top binding-sensitive features.

Env: MODEL LAYER TF
"""
import os, json, numpy as np
from scipy.stats import mannwhitneyu
ROOT = os.environ.get("GENOMIC_SAE_ROOT", os.path.dirname(os.path.abspath(__file__)))
MODEL=os.environ["MODEL"]; LAYER=int(os.environ["LAYER"]); TF=os.environ["TF"]

d=np.load(f"{ROOT}/bindingfeat_{MODEL}_{TF}_layer{LAYER}.npz")
F=d["feats"]; y=d["labels"]   # 0 bound, 1 unbound_motif, 2 background

from pyfaidx import Fasta
fa=Fasta(f"{ROOT}/hg38.fa")
def _gc(s): s=s.upper(); return (s.count("G")+s.count("C"))/len(s) if s else 0.0
wrows=[l.strip().split("\t") for l in open(f"{ROOT}/binding_windows_{TF}.tsv")][1:]
gc=np.array([_gc(str(fa[c][int(s):int(e)])) for _,c,s,e in wrows])
assert len(gc)==len(y), f"gc/window mismatch {len(gc)} vs {len(y)}"
nb=(y==0).sum(); nu=(y==1).sum(); ng=(y==2).sum()
print(f"[data] {TF}: bound={nb} unbound={nu} background={ng}, feats={F.shape[1]}",flush=True)

Fb=F[y==0]; Fu=F[y==1]; Fg=F[y==2]
n_feats=F.shape[1]
def auc_from_u(u,n1,n2): return u/(n1*n2)

res=[]
for fi in range(n_feats):
    b=Fb[:,fi]; u=Fu[:,fi]; g=Fg[:,fi]
    # skip dead features (never active anywhere)
    if max(b.max(),u.max(),g.max())<=0: continue
    # A) bound vs unbound
    try:
        uA,pA=mannwhitneyu(b,u,alternative="greater"); aucA=auc_from_u(uA,len(b),len(u))
    except ValueError: pA,aucA=1.0,0.5
    # B) unbound vs background
    try:
        uB,pB=mannwhitneyu(u,g,alternative="greater"); aucB=auc_from_u(uB,len(u),len(g))
    except ValueError: pB,aucB=1.0,0.5
    res.append(dict(feature=int(fi),
                    auc_bound_vs_unbound=float(aucA), p_bound_vs_unbound=float(pA),
                    auc_motif_vs_bg=float(aucB), p_motif_vs_bg=float(pB),
                    mean_bound=float(b.mean()), mean_unbound=float(u.mean()),
                    mean_bg=float(g.mean()),
                    gc_corr=float(np.corrcoef(F[:,fi], gc)[0,1])))

nt=len(res); bonf=0.05/max(1,nt)
binding_sensitive=[r for r in res if r["p_bound_vs_unbound"]<bonf and r["auc_bound_vs_unbound"]>0.6]
motif_selective  =[r for r in res if r["p_motif_vs_bg"]<bonf and r["auc_motif_vs_bg"]>0.6]
both=[r for r in res if r in binding_sensitive and r in motif_selective]

print(f"\n[{TF}] tested {nt} live features (Bonferroni p<{bonf:.1e})")
print(f"  motif-selective (unbound>bg)      : {len(motif_selective)}")
print(f"  binding-sensitive (bound>unbound) : {len(binding_sensitive)}")
print(f"  BOTH motif-selective & binding-sensitive: {len(both)}")
clean=[r for r in binding_sensitive if abs(r["gc_corr"])<0.2]
print(f"  binding-sensitive AND GC-robust (|gcCorr|<0.2): {len(clean)}")

res.sort(key=lambda r:-r["auc_bound_vs_unbound"])
print(f"\nTop 12 binding-sensitive features (bound vs unbound motif):")
print(f"{'feat':>6} {'AUC_b/u':>8} {'p_b/u':>10} {'AUC_m/bg':>9} {'mBound':>7} {'mUnbnd':>7} {'mBg':>7}")
for r in res[:12]:
    print(f"{r['feature']:6d} {r['auc_bound_vs_unbound']:8.3f} "
          f"{r['p_bound_vs_unbound']:10.1e} {r['auc_motif_vs_bg']:9.3f} "
          f"{r.get('gc_corr',0):7.2f} {r['mean_bound']:7.2f} {r['mean_unbound']:7.2f}")

json.dump(res, open(f"{ROOT}/bindingtest_{MODEL}_{TF}_layer{LAYER}.json","w"))
print(f"\n[done] saved bindingtest_{MODEL}_{TF}_layer{LAYER}.json")
