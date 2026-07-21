#!/usr/bin/env python
"""
Train a top-k Sparse Autoencoder (SAE) on harvested genomic-FM activations.

Top-k SAE (Gao et al. / Anthropic formulation):
    f   = TopK( W_enc (x - b_pre) )          # exactly k active features
    x_h = W_dec f + b_pre                     # reconstruction
Loss = ||x - x_h||^2  (+ auxiliary loss reviving dead features)

Memory-maps the activation .npy so we never load the full matrix into RAM.

Env vars:
  MODEL      = nt | dnabert2                 (required)
  LAYER      = layer index, e.g. 14          (required)
  EXP_FACTOR = dictionary expansion factor   (default 16)
  K          = active features per token     (default 32)
  BATCH      = minibatch size                (default 4096)
  EPOCHS     = passes over the data          (default 1)
  LR         = learning rate                 (default 4e-4)
  OUTTAG     = tag for the run               (default "full")
"""
import os, json, time
os.environ.setdefault("HF_HOME", "/projects/bhkb/sali4/genomic_sae/hf_cache")
import numpy as np
import torch
import torch.nn as nn

ROOT = os.environ.get("GENOMIC_SAE_ROOT", os.path.dirname(os.path.abspath(__file__)))
MODEL = os.environ["MODEL"]
LAYER = int(os.environ["LAYER"])
EXP_FACTOR = int(os.environ.get("EXP_FACTOR", 16))
K = int(os.environ.get("K", 32))
BATCH = int(os.environ.get("BATCH", 4096))
EPOCHS = int(os.environ.get("EPOCHS", 1))
LR = float(os.environ.get("LR", 4e-4))
OUTTAG = os.environ.get("OUTTAG", "full")
AUX_K = 512          # aux loss uses top-AUX_K dead features
AUX_COEF = 1.0 / 16  # weight on auxiliary (dead-feature revival) loss
DEAD_STEPS = 50     # a feature is "dead" if unfired for this many steps

device = "cuda" if torch.cuda.is_available() else "cpu"
act_path = f"{ROOT}/acts_{MODEL}_{OUTTAG}_layer{LAYER}.npy"
meta_path = f"{ROOT}/acts_{MODEL}_{OUTTAG}_meta.npy"
print(f"[cfg] MODEL={MODEL} LAYER={LAYER} exp={EXP_FACTOR} k={K} batch={BATCH} "
      f"epochs={EPOCHS} lr={LR} device={device}", flush=True)
print(f"[cfg] activations: {act_path}", flush=True)

# ---------- load activations (memory-mapped) ----------
X = np.load(act_path, mmap_mode="r")           # [N, d], float16 on disk
N, d = X.shape
n_feats = d * EXP_FACTOR
print(f"[data] N={N} d={d} -> dict size={n_feats}", flush=True)

# Exclude special-token rows (bp_start == -1) from training.
meta = np.load(meta_path)                       # [N, 5]
keep = meta[:, 3] >= 0                           # bp_start >= 0
keep_idx = np.nonzero(keep)[0]
print(f"[data] keeping {len(keep_idx)}/{N} non-special token rows", flush=True)

# Estimate normalization stats from a random sample (don't read all N into RAM).
rng = np.random.default_rng(0)
samp = rng.choice(keep_idx, size=min(200_000, len(keep_idx)), replace=False)
samp.sort()
Xs = torch.from_numpy(np.asarray(X[samp], dtype=np.float32))
x_mean = Xs.mean(0)
# normalize so the average squared L2 norm == d (standard SAE preconditioning)
x_norm = (Xs - x_mean).pow(2).sum(1).mean().sqrt()
scale = (d ** 0.5) / x_norm
print(f"[norm] mean|.|={x_mean.norm():.3f}  rms_norm={x_norm:.3f}  scale={scale:.4f}",
      flush=True)
x_mean = x_mean.to(device); scale = scale.to(device)

# ---------- model ----------
class TopKSAE(nn.Module):
    def __init__(self, d, n_feats, k):
        super().__init__()
        self.k = k
        self.b_pre = nn.Parameter(torch.zeros(d))
        self.W_enc = nn.Parameter(torch.empty(n_feats, d))
        self.W_dec = nn.Parameter(torch.empty(d, n_feats))
        nn.init.kaiming_uniform_(self.W_enc, a=5 ** 0.5)
        # init decoder as transpose of encoder, then unit-norm columns
        with torch.no_grad():
            self.W_dec.copy_(self.W_enc.t())
            self._normalize_decoder()

    def _normalize_decoder(self):
        with torch.no_grad():
            self.W_dec.div_(self.W_dec.norm(dim=0, keepdim=True) + 1e-8)

    def encode_pre(self, x):
        return (x - self.b_pre) @ self.W_enc.t()      # [B, n_feats]

    def forward(self, x):
        pre = self.encode_pre(x)
        # top-k activation
        topv, topi = pre.topk(self.k, dim=1)
        topv = torch.relu(topv)
        f = torch.zeros_like(pre)
        f.scatter_(1, topi, topv)
        x_hat = f @ self.W_dec.t() + self.b_pre
        return x_hat, f, pre

sae = TopKSAE(d, n_feats, K).to(device)
sae.b_pre.data = (x_mean * scale).clone()
# data-driven init: seed each dictionary atom with a normalized real activation
with torch.no_grad():
    seed_idx = rng.choice(keep_idx, size=n_feats, replace=True); seed_idx.sort()
    seed = torch.from_numpy(np.asarray(X[seed_idx], dtype=np.float32)).to(device)
    seed = (seed - x_mean) * scale
    seed = seed / (seed.norm(dim=1, keepdim=True) + 1e-8)
    sae.W_enc.data.copy_(seed)
    sae.W_dec.data.copy_(seed.t())
    sae._normalize_decoder()
print("[init] seeded dictionary from data directions", flush=True)  # init pre-bias at data mean
opt = torch.optim.Adam(sae.parameters(), lr=LR)

# ---------- training ----------
steps_since_fire = torch.zeros(n_feats, device=device)
order = keep_idx.copy()
global_step = 0
t0 = time.time()

def get_batch(idx_batch):
    # idx_batch: sorted int array; mmap gather then to GPU, normalized
    arr = np.asarray(X[idx_batch], dtype=np.float32)
    x = torch.from_numpy(arr).to(device)
    return (x - x_mean) * scale

for epoch in range(EPOCHS):
    rng.shuffle(order)
    for bstart in range(0, len(order) - BATCH, BATCH):
        idx_batch = np.sort(order[bstart:bstart + BATCH])
        x = get_batch(idx_batch)
        x_hat, f, pre = sae(x)

        resid = x - x_hat
        recon_loss = resid.pow(2).sum(1).mean()

        # track firing
        fired = (f > 0).any(0)
        steps_since_fire[fired] = 0
        steps_since_fire[~fired] += 1
        dead = steps_since_fire > DEAD_STEPS

        # auxiliary loss: let dead features reconstruct the residual
        aux_loss = torch.tensor(0.0, device=device)
        if dead.any():
            pre_dead = pre.clone()
            pre_dead[:, ~dead] = -1e9            # mask live features
            kk = min(AUX_K, int(dead.sum()))
            av, ai = pre_dead.topk(kk, dim=1)
            av = torch.relu(av)
            f_aux = torch.zeros_like(pre)
            f_aux.scatter_(1, ai, av)
            resid_hat = f_aux @ sae.W_dec.t()
            aux_loss = (resid.detach() - resid_hat).pow(2).sum(1).mean()

        loss = recon_loss + AUX_COEF * aux_loss
        opt.zero_grad()
        loss.backward()
        opt.step()
        sae._normalize_decoder()

        if global_step % 100 == 0:
            with torch.no_grad():
                var = (x - x.mean(0)).pow(2).sum(1).mean()
                fvu = (recon_loss / var).item()       # fraction of variance unexplained
                l0 = (f > 0).float().sum(1).mean().item()
            rate = (global_step + 1) * BATCH / (time.time() - t0)
            print(f"  step {global_step:6d} | FVU {fvu:.4f} | L0 {l0:.1f} | "
                  f"dead {int(dead.sum()):5d} | {rate/1e3:.1f}k tok/s", flush=True)
        global_step += 1

# ---------- final metrics + save ----------
with torch.no_grad():
    # evaluate on a held-out random sample
    ev = rng.choice(keep_idx, size=min(100_000, len(keep_idx)), replace=False)
    ev.sort()
    xe = get_batch(ev)
    xhe, fe, _ = sae(xe)
    var = (xe - xe.mean(0)).pow(2).sum(1).mean()
    fvu = ((xe - xhe).pow(2).sum(1).mean() / var).item()
    l0 = (fe > 0).float().sum(1).mean().item()
    alive = int((steps_since_fire <= DEAD_STEPS).sum())
print(f"[eval] FVU={fvu:.4f}  var_explained={1-fvu:.4f}  L0={l0:.1f}  "
      f"alive_feats={alive}/{n_feats}", flush=True)

out = f"{ROOT}/sae_{MODEL}_{OUTTAG}_layer{LAYER}_k{K}_x{EXP_FACTOR}"
torch.save({"W_enc": sae.W_enc.detach().cpu(),
            "W_dec": sae.W_dec.detach().cpu(),
            "b_pre": sae.b_pre.detach().cpu(),
            "x_mean": x_mean.cpu(), "scale": scale.cpu(),
            "config": dict(d=d, n_feats=n_feats, k=K, model=MODEL, layer=LAYER,
                           exp_factor=EXP_FACTOR)}, out + ".pt")
with open(out + "_metrics.json", "w") as f:
    json.dump(dict(model=MODEL, layer=LAYER, k=K, exp_factor=EXP_FACTOR,
                   n_feats=n_feats, fvu=fvu, var_explained=1 - fvu,
                   l0=l0, alive_feats=alive, n_train=len(keep_idx)), f, indent=2)
print(f"[done] saved {out}.pt", flush=True)
