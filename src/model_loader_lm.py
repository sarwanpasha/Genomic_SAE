"""Loader variant that returns models WITH the MaskedLM head (for logit readout)."""
import importlib, torch
from transformers import AutoTokenizer, AutoConfig, AutoModelForMaskedLM

def load_lm_model(MODEL, path, device):
    tok=AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    cfg=AutoConfig.from_pretrained(path, trust_remote_code=True)
    cfg.output_hidden_states=True
    if MODEL=="nt":
        from transformers.dynamic_module_utils import get_class_from_dynamic_module
        am=getattr(cfg,"auto_map",{}) or {}
        ref=am.get("AutoModelForMaskedLM") or am.get("AutoModel")
        ModelClass=get_class_from_dynamic_module(ref, path)
        model=ModelClass.from_pretrained(path, config=cfg, trust_remote_code=True)
    else:
        model=AutoModelForMaskedLM.from_pretrained(path, config=cfg, trust_remote_code=True)
    if MODEL=="dnabert2":
        for n,mod in list(importlib.sys.modules.items()):
            if n.endswith("bert_layers") and hasattr(mod,"flash_attn_qkvpacked_func"):
                mod.flash_attn_qkvpacked_func=None
    return tok, model.to(device).eval()
