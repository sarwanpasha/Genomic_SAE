#!/usr/bin/env python
"""
Publication-quality figures for the genomic-SAE paper.
All numbers are the real experimental results collected on the cluster.
Outputs high-resolution PNGs into Figures/ with large, bold, readable text.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# ---------------- global style: large, bold, readable ----------------
plt.rcParams.update({
    "font.size": 15,
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],
    "axes.titlesize": 17,
    "axes.titleweight": "bold",
    "axes.labelsize": 16,
    "axes.labelweight": "bold",
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 13,
    "axes.linewidth": 1.4,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
    "savefig.dpi": 400,
})
NT_C   = "#2c6fbb"   # NT blue
DB_C   = "#d1495b"   # DNABERT-2 red
GREY   = "#9aa0a6"
GREEN  = "#3a8a4f"
OUT = "Figures"

# ============================================================
# REAL DATA (from cluster runs)
# ============================================================
# SAE layer sweep: var_explained, frac_alive
sae = {
    "NT":       {"depth":[7,14,22,29], "rel":[0.25,0.50,0.75,1.0],
                 "ve":[0.722,0.739,0.662,0.720], "alive":[0.988,0.927,0.960,0.680]},
    "DNABERT-2":{"depth":[3,6,9,12],  "rel":[0.25,0.50,0.75,1.0],
                 "ve":[0.662,0.757,0.605,0.773], "alive":[1.00,1.00,0.904,0.998]},
}

# Causal validation matrix: (significant_targets, n_targets); controls all 0/15
causal = {
    # rows                NT(sig/15)   DB2(sig/15)
    "CTCF":        {"NT":(7,15),  "DNABERT-2":(10,15)},
    "GATA1":       {"NT":(9,15),  "DNABERT-2":(13,15)},
    "REST":        {"NT":(7,15),  "DNABERT-2":(14,15)},
    "SCRAMBLE":    {"NT":(0,0),   "DNABERT-2":(0,0)},   # negative: no features
    "GATA1-SCRAM": {"NT":(0,0),   "DNABERT-2":(0,0)},   # negative: no features
}
ROW_ORDER = ["CTCF","GATA1","REST","SCRAMBLE","GATA1-SCRAM"]

# Causal AUC distributions (per target feature) — real AUCs from causalpop JSONs
auc_targets = {
 "NT": {
   "CTCF":[0.600,0.612,0.592,0.483,0.568,0.487,0.564,0.469,0.545,0.544,0.482,0.546,0.548,0.561,0.569],
   "GATA1":[0.550,0.587,0.598,0.565,0.579,0.533,0.540,0.585,0.573,0.556,0.523,0.571,0.534,0.536,0.574],
   "REST":[0.609,0.558,0.559,0.559,0.566,0.596,0.523,0.574,0.555,0.582,0.563,0.575,0.553,0.569,0.594],
 },
 "DNABERT-2": {
   "CTCF":[0.707,0.631,0.598,0.538,0.502,0.441,0.570,0.611,0.583,0.581,0.490,0.516,0.573,0.569,0.587],
   "GATA1":[0.611,0.584,0.577,0.611,0.588,0.604,0.581,0.591,0.592,0.573,0.576,0.574,0.512,0.583,0.550],
   "REST":[0.503,0.633,0.605,0.615,0.605,0.613,0.599,0.608,0.601,0.582,0.586,0.590,0.594,0.607,0.582],
 },
}
auc_random = {  # pooled random-control AUCs (real)
 "NT":[0.489,0.486,0.492,0.486,0.484,0.497,0.482,0.512,0.505,0.487,0.451,0.486,0.470,0.536,0.479,
       0.509,0.502,0.504,0.504,0.499,0.489,0.491,0.461,0.506,0.499,0.495,0.498,0.508,0.511,0.498],
 "DNABERT-2":[0.442,0.453,0.478,0.493,0.460,0.474,0.457,0.513,0.470,0.460,0.477,0.485,0.467,0.477,0.496,
       0.491,0.506,0.521,0.516,0.504,0.491,0.476,0.496,0.492,0.507,0.492,0.494,0.502,0.496,0.497],
}

# Single-feature causal exemplar (NT feature 8087) and its controls
exemplar = {
 "labels":["binding\nfeature\n(8087)","random\nfeature","motif-only\nfeature"],
 "kl_bound":[1.247e-05, 6.40e-08, 1.259e-04],
 "kl_unbound":[6.885e-06, 2.35e-08, 1.040e-04],
 "auc":[0.626,0.511,0.514],
}

# DNABERT-2 CTCF feature 857 — bound/unbound/background mean activation (real)
feat857 = {"bound":3.60,"unbound":0.66,"bg":0.18}
feat8087 = {"bound":1.97,"unbound":1.24,"bg":0.77}   # NT CTCF 8087

# motif-selective counts (real)
motif_counts = {"NT":{"CTCF":265,"GATA1":3,"REST":367},
                "DNABERT-2":{"CTCF":88,"GATA1":0,"REST":135}}

# ============================================================
# FIGURE 1 — causal validation matrix (headline)
# ============================================================
fig, ax = plt.subplots(figsize=(7.2,5.4))
mat = np.full((len(ROW_ORDER),2), np.nan)
annot = [["" for _ in range(2)] for _ in ROW_ORDER]
for i,tf in enumerate(ROW_ORDER):
    for j,m in enumerate(["NT","DNABERT-2"]):
        st,nt = causal[tf][m]
        if nt==0:
            mat[i,j]=0.0; annot[i][j]="0 feat."
        else:
            mat[i,j]=st/nt; annot[i][j]=f"{st}/{nt}"
im=ax.imshow(mat,cmap="Blues",vmin=0,vmax=1,aspect="auto")
ax.set_xticks([0,1]); ax.set_xticklabels(["NT\n(6-mer)","DNABERT-2\n(BPE)"],fontweight="bold")
ax.set_yticks(range(len(ROW_ORDER)))
ax.set_yticklabels(ROW_ORDER,fontweight="bold")
for i in range(len(ROW_ORDER)):
    for j in range(2):
        v=mat[i,j]
        col="white" if v>0.5 else "black"
        ax.text(j,i,annot[i][j],ha="center",va="center",color=col,
                fontweight="bold",fontsize=15)
ax.axhline(2.5,color="black",lw=2.5)
ax.text(1.62,1.0,"positives",rotation=90,va="center",ha="center",
        fontsize=13,fontweight="bold",color=GREEN)
ax.text(1.62,3.5,"negatives",rotation=90,va="center",ha="center",
        fontsize=13,fontweight="bold",color=GREY)
ax.set_xlim(-0.5,1.9)
cb=plt.colorbar(im,ax=ax,shrink=0.8,pad=0.13)
cb.set_label("fraction of features causally validated",fontweight="bold",fontsize=14)
ax.set_title("Causally-validated binding features (of 15 tested)\nrandom-feature controls: 0/15 in every cell",
             fontsize=15)
plt.tight_layout()
plt.savefig(f"{OUT}/fig1_causal_matrix.png",bbox_inches="tight")
plt.close()
print("fig1 done")

# ============================================================
# FIGURE 2 — causal AUC distributions: targets vs controls
# ============================================================
fig,axes=plt.subplots(1,2,figsize=(13,5.2),sharey=True)
for ax,m in zip(axes,["NT","DNABERT-2"]):
    series=[]; labels=[]; colors=[]
    c = NT_C if m=="NT" else DB_C
    for tf in ["CTCF","GATA1","REST"]:
        series.append(auc_targets[m][tf]); labels.append(tf); colors.append(c)
    series.append(auc_random[m]); labels.append("random\ncontrol"); colors.append(GREY)
    positions=range(1,len(series)+1)
    bp=ax.boxplot(series,positions=positions,widths=0.62,patch_artist=True,
                  showmeans=False,medianprops=dict(color="black",linewidth=2))
    for patch,c2 in zip(bp["boxes"],colors):
        patch.set_facecolor(c2); patch.set_alpha(0.65); patch.set_edgecolor("black")
    # jittered points
    rng=np.random.default_rng(0)
    for p,s,c2 in zip(positions,series,colors):
        xs=p+rng.uniform(-0.16,0.16,size=len(s))
        ax.scatter(xs,s,s=22,color=c2,edgecolor="black",linewidth=0.4,zorder=3,alpha=0.9)
    ax.axhline(0.5,ls="--",color="grey",lw=1.3,label="chance (0.5)")
    ax.axhline(0.55,ls=":",color="red",lw=1.6,label="significance threshold (0.55)")
    ax.set_xticks(list(positions)); ax.set_xticklabels(labels,fontweight="bold")
    ax.set_title(f"{m}",fontsize=17)
    ax.set_ylim(0.40,0.75)
    if m=="NT":
        ax.set_ylabel("causal effect AUC\n(bound vs. unbound motif)",fontweight="bold")
        ax.legend(loc="upper right",frameon=True,fontsize=12)
plt.suptitle("Feature ablation: binding-specific causal effect, targets vs. random controls",
             fontsize=16,fontweight="bold",y=1.02)
plt.tight_layout()
plt.savefig(f"{OUT}/fig2_auc_distributions.png",bbox_inches="tight")
plt.close()
print("fig2 done")

# ============================================================
# FIGURE 3 — SAE reconstruction by layer
# ============================================================
fig,ax=plt.subplots(figsize=(7.4,5.2))
for m,col,mk in [("NT",NT_C,"o"),("DNABERT-2",DB_C,"s")]:
    d=sae[m]
    ax.plot(d["rel"],d["ve"],"-",marker=mk,color=col,markersize=11,linewidth=2.6,
            label=f"{m}: variance explained")
    ax.plot(d["rel"],d["alive"],"--",marker=mk,color=col,markersize=10,linewidth=2.2,
            alpha=0.55,markerfacecolor="white",label=f"{m}: fraction alive")
ax.axvspan(0.45,0.55,color=GREEN,alpha=0.10)
ax.text(0.50,0.55,"middle-layer\noptimum",ha="center",va="center",
        fontsize=12,fontweight="bold",color=GREEN)
ax.set_xlabel("relative layer depth")
ax.set_ylabel("metric value")
ax.set_ylim(0.55,1.03)
ax.set_title("Sparse-autoencoder reconstruction across layers")
ax.legend(loc="lower center",frameon=True,fontsize=12,ncol=1)
plt.tight_layout()
plt.savefig(f"{OUT}/fig3_sae_layers.png",bbox_inches="tight")
plt.close()
print("fig3 done")

# ============================================================
# FIGURE 4 — exemplar feature: activation gradient + causal effect
# ============================================================
fig,axes=plt.subplots(1,2,figsize=(12,5.0))

# (a) activation gradient bg < unbound < bound for two exemplar features
ax=axes[0]
cats=["background","unbound\nmotif","bound\n(ChIP-seq)"]
x=np.arange(3); w=0.36
v857=[feat857["bg"],feat857["unbound"],feat857["bound"]]
v8087=[feat8087["bg"],feat8087["unbound"],feat8087["bound"]]
ax.bar(x-w/2,v857,w,color=DB_C,edgecolor="black",label="DNABERT-2 feat. 857")
ax.bar(x+w/2,v8087,w,color=NT_C,edgecolor="black",label="NT feat. 8087")
ax.set_xticks(x); ax.set_xticklabels(cats,fontweight="bold")
ax.set_ylabel("mean feature activation",fontweight="bold")
ax.set_title("(a) CTCF binding-feature activation gradient")
ax.legend(frameon=True,fontsize=12)

# (b) single-feature causal KL, target vs controls
ax=axes[1]
xb=np.arange(3); w=0.36
klb=np.array(exemplar["kl_bound"])*1e5
klu=np.array(exemplar["kl_unbound"])*1e5
b1=ax.bar(xb-w/2,klb,w,color=NT_C,edgecolor="black",label="bound sites")
b2=ax.bar(xb+w/2,klu,w,color=GREY,edgecolor="black",label="unbound motif sites")
ax.set_xticks(xb); ax.set_xticklabels(exemplar["labels"],fontweight="bold",fontsize=12)
ax.set_ylabel(r"prediction shift  KL $(\times 10^{-5})$",fontweight="bold")
ax.set_title("(b) Causal effect of ablation (NT)")
ax.legend(frameon=True,fontsize=12)
for i,a in enumerate(exemplar["auc"]):
    ax.text(i,max(klb[i],klu[i])+0.6,f"AUC={a:.2f}",ha="center",
            fontsize=12,fontweight="bold")
ax.set_ylim(0,15)
plt.tight_layout()
plt.savefig(f"{OUT}/fig4_exemplar_feature.png",bbox_inches="tight")
plt.close()
print("fig4 done")

# ============================================================
# FIGURE 5 — motif-selective feature counts (supporting)
# ============================================================
fig,ax=plt.subplots(figsize=(7.0,4.8))
tfs=["CTCF","GATA1","REST"]; x=np.arange(3); w=0.36
ntv=[motif_counts["NT"][t] for t in tfs]
dbv=[motif_counts["DNABERT-2"][t] for t in tfs]
ax.bar(x-w/2,ntv,w,color=NT_C,edgecolor="black",label="NT (6-mer)")
ax.bar(x+w/2,dbv,w,color=DB_C,edgecolor="black",label="DNABERT-2 (BPE)")
for i,v in enumerate(ntv): ax.text(i-w/2,v+4,str(v),ha="center",fontweight="bold",fontsize=12)
for i,v in enumerate(dbv): ax.text(i+w/2,v+4,str(v),ha="center",fontweight="bold",fontsize=12)
ax.set_xticks(x); ax.set_xticklabels(tfs,fontweight="bold")
ax.set_ylabel("motif-selective features",fontweight="bold")
ax.set_title("Motif-selective SAE features per transcription factor")
ax.legend(frameon=True,fontsize=12)
plt.tight_layout()
plt.savefig(f"{OUT}/fig5_motif_counts.png",bbox_inches="tight")
plt.close()
print("fig5 done")
print("ALL FIGURES DONE")
