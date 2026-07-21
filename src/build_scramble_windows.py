#!/usr/bin/env python
"""
Negative-control window sets:
  SCRAMBLE      : random genomic windows, randomly labeled bound/unbound (no real signal)
  <TF>_SCRAMBLE : real TF motif-containing windows, but bound/unbound assigned RANDOMLY
                  (same sequences/motif as real TF; only the binding LABEL is scrambled)
Env: NPER(5000) WINDOW(200) SEED(0) TF(GATA1 for motif-matched scramble)
"""
import os, random, numpy as np
from pyfaidx import Fasta
ROOT = os.environ.get("GENOMIC_SAE_ROOT", os.path.dirname(os.path.abspath(__file__)))
NPER=int(os.environ.get("NPER",5000)); WINDOW=int(os.environ.get("WINDOW",200))
SEED=int(os.environ.get("SEED",0)); TF=os.environ.get("TF","GATA1")
random.seed(SEED); np.random.seed(SEED)
fa=Fasta(f"{ROOT}/hg38.fa")
MAIN=[f"chr{c}" for c in list(range(1,23))+["X","Y"]]
clen={c:len(fa[c]) for c in MAIN if c in fa}

# ---- (1) fully scrambled: random windows, random labels ----
wins=[]
while len(wins)<NPER*2:
    c=random.choice(list(clen)); a=random.randint(0,clen[c]-WINDOW); b=a+WINDOW
    s=str(fa[c][a:b]).upper()
    if set(s)<=set("ACGT") and len(s)==WINDOW: wins.append((c,a,b))
random.shuffle(wins)
with open(f"{ROOT}/binding_windows_SCRAMBLE.tsv","w") as f:
    f.write("class\tchrom\tstart\tend\n")
    for i,(c,a,b) in enumerate(wins):
        cls="bound" if i<NPER else "unbound_motif"   # FAKE labels
        f.write(f"{cls}\t{c}\t{a}\t{b}\n")
    for c,a,b in wins[:NPER]:
        f.write(f"background\t{c}\t{a}\t{b}\n")
print(f"[SCRAMBLE] wrote {2*NPER} windows, randomly labeled", flush=True)

# ---- (2) motif-matched scramble: real TF windows, randomized labels ----
src=f"{ROOT}/binding_windows_{TF}.tsv"
if os.path.exists(src):
    rows=[l.strip().split("\t") for l in open(src)][1:]
    motif_wins=[(c,s,e) for cls,c,s,e in rows if cls in ("bound","unbound_motif")]
    random.shuffle(motif_wins)
    half=len(motif_wins)//2
    bg=[(c,s,e) for cls,c,s,e in rows if cls=="background"]
    with open(f"{ROOT}/binding_windows_{TF}SCRAM.tsv","w") as f:
        f.write("class\tchrom\tstart\tend\n")
        for i,(c,s,e) in enumerate(motif_wins):
            cls="bound" if i<half else "unbound_motif"   # FAKE labels on real motif windows
            f.write(f"{cls}\t{c}\t{s}\t{e}\n")
        for c,s,e in bg:
            f.write(f"background\t{c}\t{s}\t{e}\n")
    print(f"[{TF}SCRAM] wrote {len(motif_wins)} motif windows with RANDOM labels", flush=True)
else:
    print(f"[skip] {src} not found")
print("[done]")
