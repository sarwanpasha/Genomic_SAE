#!/usr/bin/env python
"""
Harvest SAE feature activations at labeled binding windows for the
bound-vs-unbound test.

For each window in binding_windows_<TF>.tsv:
  - run sequence through the FM, take layer-LAYER activations
  - encode every token with the SAE -> feature activations
  - aggregate per window: MAX activation of each feature across the window's
    tokens (a feature "fires" on the window if it activates on any token)
Saves bindingfeat_<MODEL>_<TF>_layer<LAYER>.npz with:
  feats [n_windows, n_features] (max-pooled), labels [n_windows]

Env: MODEL LAYER TF  K(32) EXP_FACTOR(16)
GPU job.
"""
import os, json, numpy as np, torch
os.environ.setdefault("HF_HOME","/projects/bhkb/sali4/genomic_sae/hf_cache")
from pyfaidx import Fasta
from model_loader import load_model

ROOT = os.environ.get("GENOMIC_SAE_ROOT", os.path.dirname(os.path.abspath(__file__)))
MODEL=os.environ["MODEL"]; LAYER=int(os.environ["LAYER"]); TF=os.environ["TF"]
K=int(os.environ.get("K",32)); EXP=int(os.environ.get("EXP_FACTOR",16))
device="cuda" if torch.cuda.is_available() else "cpu"
path={"nt":f"{ROOT}/model_nt_500m","dnabert2":f"{ROOT}/model_dnabert2",
      "caduceus":f"{ROOT}/model_caduceus"}[MODEL]

# ---- load SAE ----
ckpt=torch.load(f"{ROOT}/sae_{MODEL}_full_layer{LAYER}_k{K}_x{EXP}.pt",
                map_location=device, weights_only=False)
W_enc=ckpt["W_enc"].to(device); b_pre=ckpt["b_pre"].to(device)
x_mean=ckpt["x_mean"].to(device); scale=ckpt["scale"].to(device)
n_feats=W_enc.shape[0]

# ---- load model + hook the target layer ----
tok, model = load_model(MODEL, path, device)
import torch.nn as nn
layers=None
for name,mod in model.named_modules():
    if isinstance(mod,nn.ModuleList) and len(mod)>=4:
        if layers is None or len(mod)>len(layers): layers=mod
cap={}
def hook(m,i,o):
    h=o[0] if isinstance(o,(tuple,list)) else o
    h=h.detach().float()
    if h.ndim==3: h=h[0]
    cap["h"]=h
layers[LAYER-1].register_forward_hook(hook)

# ---- read windows ----
fa=Fasta(f"{ROOT}/hg38.fa")
rows=[ln.strip().split("\t") for ln in open(f"{ROOT}/binding_windows_{TF}.tsv")][1:]
print(f"[cfg] {MODEL} layer{LAYER} {TF}: {len(rows)} windows, {n_feats} feats", flush=True)

label_map={"bound":0,"unbound_motif":1,"background":2}
feats=np.zeros((len(rows), n_feats), dtype=np.float32)
labels=np.zeros(len(rows), dtype=np.int64)

with torch.no_grad():
    for wi,(cls,chrom,s,e) in enumerate(rows):
        seq=str(fa[chrom][int(s):int(e)]).upper()
        labels[wi]=label_map[cls]
        enc=tok(seq, return_tensors="pt")
        _=model(**{k:v.to(device) for k,v in enc.items()})
        h=cap["h"]                                  # [n_tok, d]
        h=(h - x_mean)*scale
        pre=torch.relu((h - b_pre) @ W_enc.t())     # [n_tok, n_feats]
        feats[wi]=pre.max(0).values.cpu().numpy()   # max-pool over tokens
        if (wi+1)%2000==0: print(f"  {wi+1}/{len(rows)}", flush=True)

out=f"{ROOT}/bindingfeat_{MODEL}_{TF}_layer{LAYER}.npz"
np.savez_compressed(out, feats=feats, labels=labels,
                    label_names=np.array(["bound","unbound_motif","background"]))
print(f"[done] saved {out}  feats{feats.shape}", flush=True)
