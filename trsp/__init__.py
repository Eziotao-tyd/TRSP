# -*- coding: utf-8 -*-
"""TRSP / SBA.Tri: Topologically Regularized Side-Path for long-context LLMs.

A non-invasive, parameter-light plug-in that mitigates representation
collapse in Transformers by enforcing *spectral balance* between mixing
efficiency and information capacity.

Example (HuggingFace Llama, with or without LoRA):

    >>> from transformers import AutoModelForCausalLM
    >>> from trsp import attach_sba_tri_to_llama
    >>> model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B")
    >>> model = attach_sba_tri_to_llama(model, max_L_ctx=8192)

Example (custom Transformer with ``ln1 / attn / ln2 / mlp`` blocks):

    >>> from trsp import attach_sba_tri_to_softmax_model
    >>> model = attach_sba_tri_to_softmax_model(model)
"""

from .sba_tri import (
    # Core operators.
    box_filter_causal,
    triangular_filter_causal,
    # Modules.
    GammaMLP,
    SbaTriBlock,
    SbaTriLlamaDecoderLayer,
    # One-call injectors.
    attach_sba_tri_to_softmax_model,
    attach_sba_tri_to_llama,
)

__all__ = [
    "box_filter_causal",
    "triangular_filter_causal",
    "GammaMLP",
    "SbaTriBlock",
    "SbaTriLlamaDecoderLayer",
    "attach_sba_tri_to_softmax_model",
    "attach_sba_tri_to_llama",
]

__version__ = "0.1.0"
