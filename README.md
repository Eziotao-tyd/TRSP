# TRSP / SBA·Tri

[![arXiv](https://img.shields.io/badge/arXiv-TBD-b31b1b.svg)](https://arxiv.org/abs/TBD)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)

Official implementation of

> **The Devil is in the Spectrum: Mitigating Representation Collapse in LLMs via Topologically Regularized Side-Path**
> Yiheng Tao\*, Kaiwen Cheng\*, Yao Lu, Chang Liu, Jie Chen
> *ICML 2026 (camera-ready)* &nbsp;·&nbsp; \* equal contribution

**TL;DR.** A parameter-free, plug-in side-path that enforces *spectral
balance* — high mixing efficiency **and** high information capacity —
and mitigates representation collapse in long-context Transformers.

```python
from trsp import attach_sba_tri_to_llama

model = attach_sba_tri_to_llama(model, max_L_ctx=8192)   # that's it
```

---

## 1. Why Spectral Balance?

Long-context Transformers tend to drift into one of two pathological
extremes:

- **Homogenization collapse** — representations become indistinguishable
  (rank deficiency, attention sinks, over-mixing).
- **Isolation collapse** — representations fail to mix globally (local
  attention, disconnected context).

From a spectral viewpoint, this corresponds to an intrinsic trade-off
between two quantities:

- **Mixing Efficiency** — spectral gap; speed of global propagation.
- **Information Capacity** — effective / stable rank; richness of
  non-degenerate features.

**Goal.** Keep both high simultaneously — *spectral balance*.

## 2. Method: TRSP (triBox + Long-Context Gate)

TRSP augments each decoder layer with a **parallel side-path** added to
the residual stream:

$$
X_{\text{out}} = X_{\text{in}} + \mathrm{Attn}(X_{\text{in}}) + \gamma_\ell(L_{\text{ctx}}) \cdot \mathrm{triBox}_{b_\ell}(X_{\text{in}})
$$

### 2.1 triBox: parameter-free topology regularizer (O(T))

`triBox` applies a **causal triangular kernel** efficiently via two
cascaded causal box filters (moving averages).  The window expands
exponentially with depth, creating multi-scale topology:

- $b_\ell = \min(2^\ell,\, L_{\text{ctx}})$
  (shallow layers: proximal coupling; deep layers: distal propagation).

### 2.2 Long-Context Gate: tiny length-aware controller

A lightweight gate scales the side-path strength purely from the
context-length ratio:

- $r = b_\ell / L_{\text{ctx}}$
- $\gamma_\ell = \sigma(\mathrm{MLP}(\log_2 r)) \cdot r^{\varphi}$,
  with $\varphi = \mathrm{softplus}(\kappa)$ learned globally.

The gate is **input-agnostic** (depends only on length); it acts as a
structural safety net for long contexts. A single MLP and a single
$\kappa$ are **shared across all layers**, so total added parameters are
≈ O(1).

---

## 3. Installation

```bash
git clone https://github.com/Eziotao-tyd/TRSP.git
cd TRSP
```

The core `trsp/` package only depends on **PyTorch**; the example
training scripts additionally use `pytorch-lightning`, `transformers`,
and `peft`.

---

## 4. Usage

### 4.1 With a HuggingFace Llama (with or without LoRA)

```python
from transformers import AutoModelForCausalLM
from trsp import attach_sba_tri_to_llama

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B")
model = attach_sba_tri_to_llama(model, max_L_ctx=8192)

# Combine with PEFT LoRA if desired:
# from peft import get_peft_model, LoraConfig
# model = get_peft_model(model, LoraConfig(...))
# model = attach_sba_tri_to_llama(model, max_L_ctx=8192)
```

`attach_sba_tri_to_llama` works on both bare `LlamaForCausalLM` and
PEFT-wrapped LoRA models. It locates `LlamaModel.layers` and replaces
each `LlamaDecoderLayer` with an `SbaTriLlamaDecoderLayer` that shares a
single `GammaMLP` and `kappa` across layers.

### 4.2 With a custom decoder Transformer

If your block exposes `ln1 / attn / ln2 / mlp` submodules:

```python
from trsp import attach_sba_tri_to_softmax_model

model = attach_sba_tri_to_softmax_model(model, max_L_ctx=8192)
```

### 4.3 Public API

```python
from trsp import (
    box_filter_causal,                  # O(T) causal moving average
    triangular_filter_causal,           # cascaded box -> Fejer kernel
    GammaMLP,                           # length-aware gate MLP
    SbaTriBlock,                        # generic Transformer wrapper
    SbaTriLlamaDecoderLayer,            # HF Llama wrapper
    attach_sba_tri_to_softmax_model,    # one-call injector (custom)
    attach_sba_tri_to_llama,            # one-call injector (HF Llama)
)
```

---

## 5. Reproducing the Paper

The `examples/` directory contains the reference scripts used in our
experiments.

| Script | Setting |
|---|---|
| `examples/train_tiny_softmax_sba.py` | Pre-training a ~100M decoder LM (RoPE) with TRSP, NoLiMa-Alpaca data |
| `examples/train_llama32_lora_sba.py` | Post-training Llama-3.2-1B via LoRA + TRSP, evaluated on MMLU |
| `examples/eval_spectral_metrics.py` | Per-layer spectral diagnostics: Stable Rank / SPR / Anisotropy / Spectral Flatness |

```bash
# (a) From-scratch pretraining on NoLiMa-Alpaca
python examples/train_tiny_softmax_sba.py \
  --data_dir <jsonl_dir> \
  --llama_tokenizer_path <tokenizer_path> \
  --use_sba_tri

# (b) LoRA post-training on Llama-3.2-1B
python examples/train_llama32_lora_sba.py \
  --data_path <alpaca_json> \
  --val_data_path <mmlu_jsonl> \
  --model_name <llama-3.2-1b-path>

# (c) Spectral diagnostics (per-layer CSV)
python examples/eval_spectral_metrics.py \
  --data_path <eval_jsonl> \
  --ckpt_path <checkpoint> \
  --llama_tokenizer_path <tokenizer_path>
```

> The reference scripts enable themselves as a third-party Transformer
> by inserting the repository root into `sys.path` at the top of each
> file, so they can be invoked directly without packaging.

### Data

- **NoLiMa** (long-context needle-in-haystack): build via the
  preprocessing recipe described in the paper (Appendix A).
- **Alpaca / Alpagasus**: standard public release.
- **MMLU**: standard public release; format as JSONL with
  `instruction / input / output` keys.

Pre-trained checkpoints will be released at:
`https://huggingface.co/<TBD>` *(coming soon)*.

---

## 6. Repository Layout

```
TRSP/
├── trsp/                         core method (this is the package)
│   ├── __init__.py
│   └── sba_tri.py                triBox, GammaMLP, wrappers, injectors
├── examples/                     reference experiment scripts
│   ├── train_tiny_softmax_sba.py
│   ├── train_llama32_lora_sba.py
│   └── eval_spectral_metrics.py
├── LICENSE
└── README.md
```

---

## 7. Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{tao2026trsp,
  title     = {The Devil is in the Spectrum: Mitigating Representation
               Collapse in LLMs via Topologically Regularized Side-Path},
  author    = {Tao, Yiheng and Cheng, Kaiwen and Lu, Yao and
               Liu, Chang and Chen, Jie},
  booktitle = {Proceedings of the 43rd International Conference on
               Machine Learning (ICML)},
  year      = {2026},
  url       = {https://arxiv.org/abs/TBD}
}
```

*(arXiv ID will be filled in once the preprint is online.)*

---

## 8. License

Released under the [Apache License 2.0](LICENSE).
