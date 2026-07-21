# discover layer count from a dummy pass
with torch.no_grad():
    d = tok(records[0][4], return_tensors="pt").to(device)
    hs = model(**d).hidden_states
n_layers = len(hs) - 1  # hidden_states[0] is embeddings
hidden_dim = hs[-1].shape[-1]
# 4 evenly spaced layers across 25/50/75/100% depth
sel_layers = sorted(set(max(1, round(f * n_layers)) for f in (0.25, 0.5, 0.75, 1.0)))
print(f"[model] n_layers={n_layers} hidden_dim={hidden_dim} sel_layers={sel_layers}", flush=True)

# ---------- per-token -> bp offset mapping ----------
def token_bp_spans(model_name, token_strings, special_mask, window):
    """Return list of (bp_start, bp_end) per token; (-1,-1) for special tokens."""
    spans = []
    if model_name == "caduceus":
        pos = 0
        for t, sp in zip(token_strings, special_mask):
            if sp: spans.append((-1, -1))
            else:  spans.append((pos, pos + 1)); pos += 1
    elif model_name == "nt":
        pos = 0
        for t, sp in zip(token_strings, special_mask):
            if sp: spans.append((-1, -1)); continue
            L = len(t)  # 6 normally; final token may be shorter
            spans.append((pos, pos + L)); pos += L
    elif model_name == "dnabert2":
        pos = 0
        for t, sp in zip(token_strings, special_mask):
            if sp: spans.append((-1, -1)); continue
            core = t.replace("##", "").replace("\u2581", "")  # strip BPE markers
            L = len(core)
            spans.append((pos, pos + L)); pos += L
    else:
        raise ValueError(model_name)
    return spans

# ---------- harvest ----------
all_acts = {L: [] for L in sel_layers}   # layer -> list of [n_tok, hidden]
meta = []  # one dict per token row: which seq, which token, bp span, label
seq_index = []  # per-sequence metadata

with torch.no_grad():
    for si, (label, chrom, a, b, seq) in enumerate(records):
        enc = tok(seq, return_tensors="pt")
        ids = enc["input_ids"][0].tolist()
        toks = tok.convert_ids_to_tokens(ids)
        special = [1 if t in tok.all_special_tokens else 0 for t in toks]
        spans = token_bp_spans(MODEL, toks, special, WINDOW)
        out = model(**{k: v.to(device) for k, v in enc.items()})
        for L in sel_layers:
            h = out.hidden_states[L][0].float().cpu().numpy()  # [n_tok, hidden]
            all_acts[L].append(h)
        for ti, (t, sp, (bs, be)) in enumerate(zip(toks, special, spans)):
            meta.append((si, ti, sp, bs, be))
        seq_index.append(dict(seq_i=si, label=label, chrom=chrom,
                              start=a, end=b, n_tok=len(ids)))
        if (si + 1) % 50 == 0:
            print(f"  [{si+1}/{len(records)}]", flush=True)

# ---------- save ----------
out_prefix = f"{ROOT}/acts_{MODEL}_{OUTTAG}"
meta_arr = np.array(meta, dtype=np.int32)  # [N_tokens, 5]
