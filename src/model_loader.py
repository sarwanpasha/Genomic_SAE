"""
Robust loader for the three genomic FMs. Import load_model(MODEL, path, device).
Handles each model's loading quirk WITHOUT editing on-disk remote code
(so transformers re-downloads can't revert our fixes).
"""
import importlib
import torch
from transformers import AutoTokenizer, AutoConfig, AutoModel


def load_model(MODEL, path, device):
    if MODEL == "nt":
        # NT-v2 ships a custom EsmForMaskedLM with gated GLU FFN. The AutoModel
        # mapping isn't registered, so import the dynamic module's base model
        # class directly and instantiate from config.
        tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        cfg = AutoConfig.from_pretrained(path, trust_remote_code=True)
        cfg.output_hidden_states = True
        # get_class_from_dynamic_module loads the remote class object
        from transformers.dynamic_module_utils import get_class_from_dynamic_module
        # NT repo's auto_map points at modeling_esm.EsmModel (base encoder)
        automap = getattr(cfg, "auto_map", {}) or {}
        ref = automap.get("AutoModel") or automap.get("AutoModelForMaskedLM")
        if ref is None:
            raise RuntimeError(f"NT config has no usable auto_map: {automap}")
        # If it's a MaskedLM ref, we still load it and read .esm submodule's hidden states
        ModelClass = get_class_from_dynamic_module(ref, path)
        model = ModelClass.from_pretrained(path, config=cfg, trust_remote_code=True)
        return tok, model.to(device).eval(), ("nt", ref)

    elif MODEL == "dnabert2":
        # DNABERT-2's bundled Triton flash kernel is incompatible with modern
        # Triton. Load first, then monkeypatch the imported module so the model
        # uses its standard-attention fallback (flash func -> None), IN MEMORY.
        tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        cfg = AutoConfig.from_pretrained(path, trust_remote_code=True)
        cfg.output_hidden_states = True
        model = AutoModel.from_pretrained(path, config=cfg, trust_remote_code=True)
        # find the dynamically-imported bert_layers module and null the flash fn
        patched = []
        for name, mod in list(importlib.sys.modules.items()):
            if name.endswith("bert_layers") and hasattr(mod, "flash_attn_qkvpacked_func"):
                mod.flash_attn_qkvpacked_func = None
                patched.append(name)
        if not patched:
            raise RuntimeError("could not find bert_layers module to patch")
        print(f"[loader] patched flash_attn -> None in: {patched}", flush=True)
        return tok, model.to(device).eval(), ("dnabert2", patched)

    elif MODEL == "caduceus":
        tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        cfg = AutoConfig.from_pretrained(path, trust_remote_code=True)
        cfg.output_hidden_states = True
        model = AutoModel.from_pretrained(path, config=cfg, trust_remote_code=True)
        return tok, model.to(device).eval(), ("caduceus", None)

    else:
        raise ValueError(MODEL)
