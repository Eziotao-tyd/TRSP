# -*- coding: utf-8 -*-
"""
TRSP / SBA·Tri core module.

This file contains the *method-only* implementation of the
Topologically Regularized Side-Path (TRSP), realized as **SBA·Tri**:

    X_out = X_in + Attn(X_in) + gamma_ell(L_ctx) * triBox(X_in)

Components:
    - box_filter_causal              : O(T) causal moving average via cumsum.
    - triangular_filter_causal       : Two cascaded box filters (Fejer kernel).
    - GammaMLP                       : Tiny scalar gate MLP shared across layers.
    - SbaTriBlock                    : Wrapper for a generic Transformer block
                                       (used by the from-scratch TinySoftmaxLM).
    - SbaTriLlamaDecoderLayer        : Wrapper for HuggingFace LlamaDecoderLayer
                                       (compatible with LoRA / PEFT).
    - attach_sba_tri_to_softmax_model: One-call injector for the toy model.
    - attach_sba_tri_to_llama        : One-call injector for HF Llama models
                                       (works for both base and PEFT-wrapped
                                       models).
"""

from typing import Optional
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Causal triangular kernel  (parameter-free, O(T))
# ============================================================

def box_filter_causal(x: torch.Tensor, L: int) -> torch.Tensor:
    """Causal box filter (window length L, left-padded with zeros).

    Args:
        x: tensor of shape ``[B, T, D]``.
        L: window length.  ``L <= 1`` is a no-op.

    Returns:
        tensor of shape ``[B, T, D]``.
    """
    if L <= 1:
        return x
    B, T, D = x.shape
    device, dtype = x.device, x.dtype

    # Pad a 0 in front of time dimension for cumsum sliding window.
    pad = torch.zeros(B, 1, D, device=device, dtype=dtype)
    x_pad = torch.cat([pad, x], dim=1)      # [B, T+1, D]
    c = x_pad.cumsum(dim=1)                 # prefix sum

    t_idx = torch.arange(T, device=device) + 1        # 1..T
    left_idx = (t_idx - L).clamp_min(0)               # max(0, t+1-L)
    right_idx = t_idx                                 # t+1

    c_right = c[:, right_idx, :]                      # [B, T, D]
    c_left  = c[:, left_idx, :]                       # [B, T, D]
    s = c_right - c_left                              # [B, T, D]

    return s / float(L)


def triangular_filter_causal(x: torch.Tensor, L: int) -> torch.Tensor:
    """Two cascaded causal box filters -> causal triangular (Fejer) kernel.

    Total cost is still O(T * D).
    """
    if L <= 1:
        return x
    return box_filter_causal(box_filter_causal(x, L), L)


# ============================================================
# Length-aware scalar gate
# ============================================================

class GammaMLP(nn.Module):
    """Scalar conditional network.

    Maps ``log2(r)`` (with ``r = b_ell / L_ctx``) to a scalar logit; the final
    gate value is ``sigmoid(logit) * r ** phi`` where ``phi = softplus(kappa)``
    is a globally-shared positive scalar.

    According to the SBA.Tri setting, **a single GammaMLP is shared across all
    layers**; layer-specific behaviour comes purely from the doubling
    bandwidth ``b_ell = min(2 ** ell, L_ctx)``.
    """

    def __init__(self, hidden_dim: int = 16):
        super().__init__()
        self.fc1 = nn.Linear(1, hidden_dim)
        self.act = nn.Tanh()
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, log_r: torch.Tensor) -> torch.Tensor:
        # log_r: [B, 1] or [1, 1]
        x = self.fc1(log_r)
        x = self.act(x)
        x = self.fc2(x)
        return x


def _compute_gamma(
    x: torch.Tensor,
    layer_idx: int,
    L_ctx: int,
    gamma_mlp: GammaMLP,
    kappa: nn.Parameter,
):
    """Shared helper: returns ``(gamma, L_ctx, b)``.

    ``gamma`` is shaped ``[1, 1, 1]`` so it broadcasts to ``[B, T, D]``.
    """
    L_ctx = max(int(L_ctx), 1)

    # Doubling bandwidth: b_ell = min(2 ** ell, L_ctx).
    b = min(1 << layer_idx, L_ctx)

    # Scale ratio r in (0, 1].
    r = b / float(L_ctx)
    r_clamped = max(r, 1e-6)
    log_r = math.log2(r_clamped)

    # Build log_r tensor with matching dtype / device.
    log_r_tensor = x.new_full((1, 1), log_r)

    phi = F.softplus(kappa)                            # > 0
    gamma_logit = gamma_mlp(log_r_tensor)              # [1, 1]
    gamma = torch.sigmoid(gamma_logit) * (r_clamped ** phi)
    gamma = gamma.view(1, 1, 1)
    return gamma, L_ctx, b


# ============================================================
# Wrapper 1: generic Transformer block (from-scratch reference)
# ============================================================

class SbaTriBlock(nn.Module):
    """Wrap a vanilla Transformer block with SBA.Tri.

    The wrapper reuses the original block's submodules verbatim:
        ``ln1`` / ``attn`` / ``ln2`` / ``mlp``.

    Only the attention residual is modified::

        x = residual + Attn(ln1(residual)) + gamma_ell(L_ctx) * Tri(Tri(residual))

    The MLP residual remains unchanged.
    """

    def __init__(
        self,
        orig_block: nn.Module,
        layer_idx: int,
        gamma_mlp: GammaMLP,
        kappa: nn.Parameter,
    ):
        super().__init__()
        # Reuse submodules from the original block.
        self.ln1 = orig_block.ln1
        self.attn = orig_block.attn
        self.ln2 = orig_block.ln2
        self.mlp = orig_block.mlp

        # 0-based layer index (drives doubling bandwidth).
        self.layer_idx = layer_idx

        # Shared Gamma and kappa (phi = softplus(kappa)).
        self.gamma_mlp = gamma_mlp
        self.kappa = kappa

    def _compute_gate_and_bandwidth(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor],
    ):
        if attn_mask is not None:
            # Each sample's valid length (non-pad token count); take batch max.
            L_ctx = int(attn_mask.sum(dim=1).max().item())
        else:
            L_ctx = x.size(1)
        return _compute_gamma(x, self.layer_idx, L_ctx, self.gamma_mlp, self.kappa)

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
        # ---- Self-attention + SBA.Tri ----
        residual = x
        x_norm = self.ln1(x)
        attn_out = self.attn(x_norm, attn_mask)        # original softmax attention

        gamma, L_ctx, b = self._compute_gate_and_bandwidth(residual, attn_mask)

        if b > 0:
            yB = triangular_filter_causal(residual, L=b + 1)   # Tri(Tri(residual))
            x = residual + attn_out + gamma * yB
        else:
            x = residual + attn_out

        # ---- MLP residual (unchanged) ----
        residual2 = x
        x_norm2 = self.ln2(x)
        mlp_out = self.mlp(x_norm2)
        x = residual2 + mlp_out
        return x


# ============================================================
# Wrapper 2: HuggingFace LlamaDecoderLayer (LoRA-compatible)
# ============================================================

class SbaTriLlamaDecoderLayer(nn.Module):
    """Wrap a HuggingFace ``LlamaDecoderLayer`` with SBA.Tri.

    All original submodules are reused verbatim, including LoRA-injected
    ``q_proj`` / ``k_proj`` / ``v_proj`` / ``o_proj`` / MLP linears.  The
    forward signature matches the official one (``hidden_states``,
    ``attention_mask``, ``position_ids``, ``past_key_values``, ``use_cache``,
    ``cache_position``, ``position_embeddings``, ``**kwargs``) and returns a
    single hidden-state tensor.
    """

    def __init__(
        self,
        orig_layer: nn.Module,
        layer_idx: int,
        gamma_mlp: GammaMLP,
        kappa: nn.Parameter,
    ):
        super().__init__()
        # Reuse all submodules (including LoRA q/k/v/o etc.).
        self.self_attn = orig_layer.self_attn
        self.mlp = orig_layer.mlp
        self.input_layernorm = orig_layer.input_layernorm
        self.post_attention_layernorm = orig_layer.post_attention_layernorm

        # Keep config (in case it is queried elsewhere).
        self.config = getattr(orig_layer, "config", None)

        self.layer_idx = layer_idx
        self.gamma_mlp = gamma_mlp
        self.kappa = kappa

    def _compute_gate_and_bandwidth(
        self,
        x: torch.Tensor,
        cache_position: Optional[torch.Tensor],
    ):
        # Use cache_position to infer L_ctx (consistent with HF Llama).
        if cache_position is not None:
            # cache_position: [seq_len], global position, starts from 0.
            L_ctx = int(cache_position[-1].item()) + 1
        else:
            # Degenerate case: treat L_ctx = current sequence length.
            L_ctx = x.size(1)
        return _compute_gamma(x, self.layer_idx, L_ctx, self.gamma_mlp, self.kappa)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        use_cache: Optional[bool] = False,
        cache_position=None,
        position_embeddings: Optional["tuple[torch.Tensor, torch.Tensor]"] = None,
        **kwargs,
    ):
        # ---- Self-attention + SBA.Tri ----
        residual = hidden_states
        x_for_tri = residual

        hidden_states = self.input_layernorm(hidden_states)

        # Self-attention (consistent with the official implementation).
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )

        gamma, L_ctx, b = self._compute_gate_and_bandwidth(x_for_tri, cache_position)

        if b > 0:
            yB = triangular_filter_causal(x_for_tri, L=b + 1)
            hidden_states = residual + hidden_states + gamma * yB
        else:
            hidden_states = residual + hidden_states

        # ---- MLP residual (unchanged) ----
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        # Match the official signature: return a tensor only.
        return hidden_states


# ============================================================
# One-call injectors
# ============================================================

def attach_sba_tri_to_softmax_model(
    model: nn.Module,
    max_L_ctx: int = 32768,
    gamma_hidden_dim: int = 16,
):
    """Inject SBA.Tri into a vanilla decoder model.

    Assumes ``model.blocks`` is an ``nn.ModuleList`` of Transformer blocks,
    each exposing ``ln1`` / ``attn`` / ``ln2`` / ``mlp`` submodules (this is
    the contract of the reference ``TinySoftmaxLM``).
    """
    layers = model.blocks
    num_layers = len(layers)

    # Shared Gamma and kappa.
    gamma_mlp = GammaMLP(hidden_dim=gamma_hidden_dim)
    kappa = nn.Parameter(torch.zeros(1))

    # Attach for downstream regularization access.
    model.sba_gamma = gamma_mlp
    model.sba_kappa = kappa
    model.sba_num_layers = num_layers
    model.sba_max_L_ctx = max_L_ctx

    # Register so the optimizer sees them.
    model.add_module("sba_gamma", gamma_mlp)
    model.register_parameter("sba_kappa", kappa)

    new_layers = []
    for ell, blk in enumerate(layers):
        wrapped = SbaTriBlock(
            orig_block=blk,
            layer_idx=ell,
            gamma_mlp=gamma_mlp,
            kappa=kappa,
        )
        new_layers.append(wrapped)
    model.blocks = nn.ModuleList(new_layers)

    print(f"[SBA.Tri] Injected on {num_layers} TransformerBlocks")
    return model


def attach_sba_tri_to_llama(
    llama_or_peft_model: nn.Module,
    max_L_ctx: int = 32768,
    gamma_hidden_dim: int = 16,
):
    """Inject SBA.Tri into a HuggingFace Llama model.

    Works for both:
        * a bare ``LlamaForCausalLM``;
        * a PEFT-wrapped LoRA model (``peft_model``) - in this case the
          base model is fetched via ``get_base_model()`` and LoRA-injected
          ``q/k/v/o_proj`` are preserved verbatim.

    The function locates ``LlamaModel.layers`` and replaces each
    ``LlamaDecoderLayer`` with an :class:`SbaTriLlamaDecoderLayer` that
    shares a single ``GammaMLP`` and ``kappa`` across all layers.
    """
    # 1) Get the base model (ForCausalLM).
    if hasattr(llama_or_peft_model, "get_base_model"):
        base_model = llama_or_peft_model.get_base_model()
    elif hasattr(llama_or_peft_model, "base_model"):
        base_model = llama_or_peft_model.base_model
    else:
        base_model = llama_or_peft_model

    # 2) Find the decoder layer list.
    llama_model = None
    if hasattr(base_model, "model") and hasattr(base_model.model, "layers"):
        # Common: LlamaForCausalLM.model -> LlamaModel
        llama_model = base_model.model
    elif (
        hasattr(base_model, "model")
        and hasattr(base_model.model, "decoder")
        and hasattr(base_model.model.decoder, "layers")
    ):
        llama_model = base_model.model.decoder
    elif hasattr(base_model, "layers"):
        llama_model = base_model
    else:
        raise ValueError(
            "[SBA.Tri] Cannot find Llama decoder layers; "
            "please check the model structure."
        )

    layers = llama_model.layers
    num_layers = len(layers)

    # 3) Shared Gamma and kappa.
    gamma_mlp = GammaMLP(hidden_dim=gamma_hidden_dim)
    kappa = nn.Parameter(torch.zeros(1))

    # Attach to the *outer* model (peft or base) for regularization access.
    llama_or_peft_model.sba_gamma = gamma_mlp
    llama_or_peft_model.sba_kappa = kappa
    llama_or_peft_model.sba_num_layers = num_layers
    llama_or_peft_model.sba_max_L_ctx = max_L_ctx

    # 4) Replace each layer with the wrapper.
    new_layers = []
    for ell, layer in enumerate(layers):
        wrapped = SbaTriLlamaDecoderLayer(
            orig_layer=layer,
            layer_idx=ell,
            gamma_mlp=gamma_mlp,
            kappa=kappa,
        )
        new_layers.append(wrapped)
    llama_model.layers = nn.ModuleList(new_layers)

    print(f"[SBA.Tri] Injected on {num_layers} Llama decoder layers.")
    return llama_or_peft_model
