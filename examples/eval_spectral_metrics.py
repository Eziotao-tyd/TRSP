# -*- coding: utf-8 -*-
"""
eval_spectral_metrics.py

Spectral diagnostics for TinySoftmaxLM + TRSP / SBA.Tri.

Computes 4 per-layer metrics on the first ``--k`` valid samples:
    * StableRank       - Stable / Effective Rank (information capacity)
    * SPR              - Signal Propagation Rate (mixing efficiency proxy)
    * Anisotropy       - Representation Anisotropy (collapse indicator)
    * SpectralFlatness - Spectral Flatness (conditioning proxy)

Each sample is forwarded twice (clean prompt + embedding-noise-perturbed
prompt) so SPR can be estimated as a noise-contraction rate.

Output: a wide-table CSV with one row per layer (layer 0 = embedding).
"""

import os
import json
import csv
import argparse
import pathlib
import math
from typing import Optional, Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

import pytorch_lightning as pl
from transformers import AutoTokenizer

# ---- TRSP / SBA.Tri + reference TinySoftmaxLM ----
# Make the repo root importable so ``import trsp`` and the sibling example
# script work when this file is executed directly.
import sys
_THIS_DIR = pathlib.Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from trsp import attach_sba_tri_to_softmax_model  # noqa: E402
from train_tiny_softmax_sba import TinySoftmaxLM, LitTinyLM  # noqa: E402


# ===================== Data: Alpaca JSONL =====================

class AlpacaJsonlDataset(Dataset):
    """Read Alpaca format JSONL (instruction/input/output)"""
    def __init__(self, path: str, tokenizer, max_seq_len: int = 2048):
        self.path = pathlib.Path(path)
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        assert self.path.exists(), f"Data file does not exist: {self.path}"
        self.samples = []
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                instr = obj.get("instruction", "") or ""
                inp   = obj.get("input", "") or ""
                out   = obj.get("output", "") or ""
                prefix = (instr.strip() + "\n").strip() if instr else ""
                src = (prefix + inp.rstrip()).strip()
                tgt = " " + out.strip()
                self.samples.append((src, tgt, out.strip()))
        print(f"[Info] Loaded: {len(self.samples)}  from {self.path}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx):
        src, tgt, gold_answer = self.samples[idx]
        tok = self.tokenizer
        src_ids = tok.encode(src, add_special_tokens=False)
        tgt_ids = tok.encode(tgt + tok.eos_token, add_special_tokens=False)

        keep_src = min(len(src_ids), self.max_seq_len - 1 - min(len(tgt_ids), 64))
        input_ids = src_ids[:keep_src] + tgt_ids
        input_ids = input_ids[: self.max_seq_len - 1] + [tok.eos_token_id]

        labels = [-100] * keep_src + tgt_ids
        labels = labels[: self.max_seq_len - 1] + [-100]

        attn_mask = [1] * len(input_ids)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attn_mask, dtype=torch.long),
            "gold_answer": gold_answer,
            "src_text": src,
        }


def pad_collate(batch, pad_id):
    max_len = max(len(x["input_ids"]) for x in batch)
    input_ids, labels, attn = [], [], []
    src_texts = []

    for x in batch:
        L = len(x["input_ids"])
        pad_len = max_len - L
        input_ids.append(F.pad(x["input_ids"], (0, pad_len), value=pad_id))
        labels.append(F.pad(x["labels"], (0, pad_len), value=-100))
        attn.append(F.pad(x["attention_mask"], (0, pad_len), value=0))
        src_texts.append(x.get("src_text", ""))

    return {
        "input_ids": torch.stack(input_ids, dim=0),
        "labels": torch.stack(labels, dim=0),
        "attention_mask": torch.stack(attn, dim=0),
        "src_texts": src_texts,
    }


# ===================== Metric Calculation (including embedding layer layer=0) =====================

def _center_over_BT(X: torch.Tensor) -> torch.Tensor:
    # X: [1, T, D] (or [B,T,D]), take mean over (B,T), center each dimension
    mu = X.mean(dim=(0, 1), keepdim=True)
    return X - mu


@torch.no_grad()
def compute_spectral_stats_fast(X: torch.Tensor, eps: float = 1e-12):
    """
    Quickly compute singular values using X^T X, return Stable Rank and Spectral Flatness.

    X: [B, T, D] -> reshape to [N, D]
    """
    # 1. Flatten
    X_flat = X.reshape(-1, X.size(-1)).float()  # [N, D]
    N, D = X_flat.shape

    # 2. Compute correlation matrix (D x D), very fast
    # Note: if N < D (very short text), should use X @ X.T, but usually N >> D
    if N >= D:
        cov = X_flat.t() @ X_flat  # [D, D]
        # Symmetric matrix eigenvalue decomposition (eigenvalues), result is square of singular values
        eigvals = torch.linalg.eigvalsh(cov)
    else:
        # Rare case N < D
        cov = X_flat @ X_flat.t()
        eigvals = torch.linalg.eigvalsh(cov)

    # 3. Convert to singular values sigma
    # eigvalsh returns in ascending order, may have tiny negative values (numerical error), need clamp
    eigvals = torch.clamp(eigvals, min=0.0)
    sigmas = torch.sqrt(eigvals + eps)  # [D]

    # --- Metric 1: Stable Rank ---
    # r_eff = (sum sigma^2) / (max sigma)^2
    # sum sigma^2 is sum of eigvals (Frobenius norm squared)
    sigma_max = sigmas.max()
    fro_sq = eigvals.sum()
    m_rank = fro_sq / (sigma_max**2 + eps)

    # --- Metric 2: Spectral Flatness ---
    # SpectralFlatness = GMean(sigma) / AMean(sigma)
    # AMean = mean(sigma)
    amean = sigmas.mean()
    # GMean = exp(mean(log(sigma)))
    gmean = torch.exp(torch.mean(torch.log(sigmas + eps)))
    xi = gmean / (amean + eps)

    return float(m_rank.item()), float(xi.item())


@torch.no_grad()
def compute_spr_tokenwise(
    X_prev: torch.Tensor,
    X_cur: torch.Tensor,
    X_prev_p: torch.Tensor,
    X_cur_p: torch.Tensor,
    denom_eps: float = 1e-12,
    min_delta_in: float = 1e-6,
) -> float:
    """
    SPR_l = mean_t ||δ_out(t)|| / (||δ_in(t)|| + eps)
      δ_in(t)  = X_prev(t) - X_prev'(t)
      δ_out(t) = X_cur(t)  - X_cur'(t)

    X_*: [1, T, D]
    """
    din = torch.linalg.norm((X_prev - X_prev_p).float(), dim=-1).squeeze(0)   # [T]
    dout = torch.linalg.norm((X_cur  - X_cur_p ).float(), dim=-1).squeeze(0)  # [T]

    mask = din > min_delta_in
    if mask.sum().item() == 0:
        return float("nan")

    r = dout[mask] / (din[mask] + denom_eps)
    return float(r.mean().item())


@torch.no_grad()
def compute_anisotropy(x: torch.Tensor, num_samples: int = 1000) -> float:
    """
    Compute representation anisotropy (Representation Anisotropy).

    Measures SB-2 (Non-degeneracy).

    Principle:
    - Sink Collapse -> all x_i point to Sink -> Anisotropy high
    - Over-smoothing -> all x_i point to Mean -> Anisotropy high
    - SBA -> x_i retains local features -> Anisotropy low (closer to isotropic)

    x: [B, T, D]
    For speed, randomly sample num_samples point pairs to compute, instead of full O(T^2).
    """
    B, T, D = x.shape
    x_flat = x.view(-1, D).float()  # [N, D]

    # Randomly sample indices to speed up computation (especially for long text)
    if x_flat.size(0) > num_samples:
        idx = torch.randperm(x_flat.size(0), device=x.device)[:num_samples]
        x_sub = x_flat[idx]
    else:
        x_sub = x_flat

    # Normalize
    x_norm = F.normalize(x_sub, p=2, dim=-1)

    # Compute Cosine Similarity Matrix: [M, M]
    cos_sim = x_norm @ x_norm.t()

    # Remove diagonal (self-similarity always 1)
    mask = ~torch.eye(x_sub.size(0), device=x.device, dtype=torch.bool)
    avg_sim = cos_sim[mask].mean()

    return float(avg_sim.item())




# ===================== Utility Functions =====================

@torch.no_grad()
def forward_hidden_states(
    model: TinySoftmaxLM,
    input_ids: torch.Tensor = None,
    inputs_embeds: torch.Tensor = None,
    attention_mask: torch.Tensor = None,
):
    """
    Manually collect TinySoftmaxLM's hidden_states
    Returns tuple hidden_states (len = L+1), each [1,T,D]
    - layer 0: embedding output
    - layer 1..L: each TransformerBlock output (wrapped by SBA·Tri)
    """
    hidden_states = []
    
    # Layer 0: embedding
    if inputs_embeds is not None:
        x = inputs_embeds
    else:
        x = model.embed(input_ids)  # [B, T, D]
    hidden_states.append(x)
    
    # Layer 1..L: TransformerBlocks (may be SbaTriBlock)
    for blk in model.blocks:
        x = blk(x, attention_mask)
        hidden_states.append(x)
    
    return tuple(hidden_states)


def build_prefix_ids(tokenizer, input_ids_row, labels_row, src_text: str, device, max_seq_len: int):
    # Find target start position s
    idx = (labels_row != -100).nonzero(as_tuple=False)
    if len(idx) == 0:
        return None
    s = idx[0].item()
    
    prefix_ids = input_ids_row[:s].unsqueeze(0).to(device)
    
    if prefix_ids.size(1) > max_seq_len:
        prefix_ids = prefix_ids[:, -max_seq_len:]
    return prefix_ids


def mean_std(xs: List[float]):
    # Ignore NaN
    t = torch.tensor([x for x in xs if x == x], dtype=torch.float32)  # x==x filters NaN
    if t.numel() == 0:
        return float("nan"), float("nan"), 0
    return float(t.mean().item()), float(t.std(unbiased=True).item() if t.numel() > 1 else 0.0), int(t.numel())


# ===================== Main Process =====================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True,
                        help="Alpaca JSONL data path")
    parser.add_argument("--ckpt_path", type=str, required=True,
                        help="TinySoftmaxLM + SBA·Tri checkpoint path")
    parser.add_argument("--llama_tokenizer_path", type=str, required=True,
                        help="Llama-3.2-1B tokenizer path")
    parser.add_argument("--max_seq_len", type=int, default=65536)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=2)

    parser.add_argument("--precision", type=str, default="bf16-mixed",
                        choices=["bf16-mixed", "16-mixed", "32"])
    parser.add_argument("--devices", type=int, default=1, help="Single GPU test")

    parser.add_argument("--k", type=int, default=100, help="Only count first K valid samples")
    parser.add_argument("--eps_noise", type=float, default=1e-3, help="embedding noise strength ε")
    parser.add_argument("--seed", type=int, default=42)


    parser.add_argument("--out_csv", type=str, default="tiny_softmax_sba_sb_metrics.csv")
    
    # Model structure parameters (must be consistent with training)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--n_layer", type=int, default=14)
    parser.add_argument("--n_head", type=int, default=8)
    parser.add_argument("--mlp_ratio", type=float, default=4.0)
    
    # SBA·Tri parameters (must be consistent with training)
    parser.add_argument("--gamma_hidden_dim", type=int, default=16,
                        help="GammaMLP hidden dimension")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Info] device = {device}")

    print(f"[Info] Loading tokenizer: {args.llama_tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.llama_tokenizer_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    actual_vocab_size = tokenizer.vocab_size + len(tokenizer.added_tokens_decoder)
    print(f"[Info] Tokenizer: vocab_size={tokenizer.vocab_size}, actual={actual_vocab_size}")

    # Build model
    model = TinySoftmaxLM(
        vocab_size=actual_vocab_size,
        d_model=args.d_model,
        n_layer=args.n_layer,
        n_head=args.n_head,
        mlp_ratio=args.mlp_ratio,
        max_seq_len=args.max_seq_len,
    )
    
    # Inject SBA·Tri (must be before loading checkpoint)
    print(f"[Info] Injecting SBA·Tri: gamma_hidden_dim={args.gamma_hidden_dim}")
    model = attach_sba_tri_to_softmax_model(
        model,
        max_L_ctx=args.max_seq_len,
        gamma_hidden_dim=args.gamma_hidden_dim,
    )
    
    # Load from checkpoint
    print(f"[Info] Loading from checkpoint: {args.ckpt_path}")
    lit = LitTinyLM.load_from_checkpoint(
        args.ckpt_path,
        model=model,
        tokenizer=tokenizer,
        use_sba_tri=True,
        strict=False,
    )
    model = lit.model
    model.to(device)
    model.eval()
    
    # Output parameter count
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[Info] Model params: {total_params/1e6:.3f} M")

    dataset = AlpacaJsonlDataset(args.data_path, tokenizer, max_seq_len=args.max_seq_len)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=lambda b: pad_collate(b, pad_id=tokenizer.pad_token_id),
        pin_memory=(device.type == "cuda"),
    )

    # metrics[layer][metric] -> list of values
    metrics: Dict[int, Dict[str, List[float]]] = {}
    n_layers_total: Optional[int] = None

    # noise rng
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    saved = 0
    pbar = tqdm(dataloader, desc=f"Compute metrics (first {args.k})")

    for batch in pbar:
        if saved >= args.k:
            break

        input_ids = batch["input_ids"]
        labels = batch["labels"]
        src_texts = batch.get("src_texts", [""] * input_ids.size(0))

        B = input_ids.size(0)
        for i in range(B):
            if saved >= args.k:
                break

            prefix_ids = build_prefix_ids(
                tokenizer=tokenizer,
                input_ids_row=input_ids[i],
                labels_row=labels[i],
                src_text=src_texts[i] if i < len(src_texts) else "",
                device=device,
                max_seq_len=args.max_seq_len,
            )
            if prefix_ids is None:
                continue

            T = prefix_ids.size(1)
            attn = torch.ones((1, T), dtype=torch.long, device=device)

            # 1) Original forward
            hs = forward_hidden_states(
                model=model,
                input_ids=prefix_ids,
                inputs_embeds=None,
                attention_mask=attn,
            )  # tuple len=L+1, each [1,T,D]

            # Initialize layer count and containers
            if n_layers_total is None:
                n_layers_total = len(hs)  # L+1 (including embedding)
                for l in range(n_layers_total):
                    metrics[l] = {"StableRank": [], "SPR": [], "Anisotropy": [], "SpectralFlatness": []}
                print(f"[Info] Found layers (including embedding) = {n_layers_total} => transformer blocks L = {n_layers_total-1}")

            # 2) Forward after adding noise to embedding (inputs_embeds)
            emb = model.embed(prefix_ids)  # [1,T,D]
            emb_f = emb.float()
            # For reproducibility: use different seed for each sample
            g = torch.Generator(device=device)
            g.manual_seed(args.seed + saved)
            noise = torch.randn(emb_f.shape, generator=g, device=device, dtype=torch.float32)
            emb_noisy = (emb_f + args.eps_noise * noise).to(dtype=emb.dtype)

            hs_p = forward_hidden_states(
                model=model,
                input_ids=None,
                inputs_embeds=emb_noisy,
                attention_mask=attn,
            )

            # 3) per-layer metrics
            for l in range(n_layers_total):
                Xl = hs[l]      # [1,T,D]
                # Use fast method to compute StableRank and SpectralFlatness simultaneously
                m_rank, flatness = compute_spectral_stats_fast(Xl)
                metrics[l]["StableRank"].append(m_rank)
                metrics[l]["SpectralFlatness"].append(flatness)
                
                metrics[l]["Anisotropy"].append(
                    compute_anisotropy(Xl)
                )

                # SPR: from l-1 to l (layer0 undefined)
                if l == 0:
                    metrics[l]["SPR"].append(float("nan"))
                else:
                    metrics[l]["SPR"].append(
                        compute_spr_tokenwise(
                            X_prev=hs[l-1], X_cur=hs[l],
                            X_prev_p=hs_p[l-1], X_cur_p=hs_p[l],
                        )
                    )

            saved += 1
            pbar.set_postfix({"saved": saved, "T": int(T)})

            # Release memory peak
            del hs, hs_p, emb, emb_f, noise, emb_noisy
            if device.type == "cuda" and (saved % 8 == 0):
                torch.cuda.empty_cache()

    if n_layers_total is None or saved == 0:
        raise RuntimeError("No samples processed successfully")

    # Aggregate mean/std -> CSV (wide table, one row per layer)
    length = args.data_path.split("/")[-1].split(".")[0]
    args.out_csv = args.out_csv.replace(".csv", f"_{length}.csv")
    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "layer",
            "StableRank_mean", "StableRank_std",
            "SPR_mean", "SPR_std",
            "Anisotropy_mean",  "Anisotropy_std",
            "SpectralFlatness_mean", "SpectralFlatness_std",
            "n",
        ])
        for l in range(n_layers_total):
            mr_mu, mr_sd, n1 = mean_std(metrics[l]["StableRank"])
            spr_mu, spr_sd, n2 = mean_std(metrics[l]["SPR"])
            la_mu, la_sd, n3 = mean_std(metrics[l]["Anisotropy"])
            flatness_mu, flatness_sd, n4 = mean_std(metrics[l]["SpectralFlatness"])
            n_eff = max(n1, n2, n3, n4)
            writer.writerow([
                l,
                mr_mu, mr_sd,
                spr_mu, spr_sd,
                la_mu, la_sd,
                flatness_mu, flatness_sd,
                n_eff,
            ])

    print(f"[Info] Completed: sample count = {saved}")
    print(f"[Info] CSV written to: {args.out_csv}")


if __name__ == "__main__":
    main()
