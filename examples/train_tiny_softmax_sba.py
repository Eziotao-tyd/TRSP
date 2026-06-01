# -*- coding: utf-8 -*-
"""
Reference example: TinySoftmaxLM (~100M, RoPE) + TRSP / SBA.Tri.

This script is provided as a *reference implementation* of how to plug
SBA.Tri into a from-scratch decoder Transformer.  The TRSP method itself
lives in the ``trsp`` package; this file only contains the toy model,
data pipeline, and Lightning training loop used in the paper.

- Data: Alpaca JSONL (instruction/input/output), NoLiMa-Alpaca variant
- Tokenization: reuse Llama-3.2-1B tokenizer
- Training: standard next-token cross-entropy (time-shift)
- Evaluation: full-answer exact-match accuracy (generation-based)
"""

import os, json, math, random, argparse, pathlib, re, string
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import pytorch_lightning as pl
from transformers import AutoTokenizer

# ---- TRSP / SBA.Tri ----
# Make the repo root importable so ``import trsp`` works when this script
# is executed directly from the ``examples/`` directory.
import sys
_THIS_DIR = pathlib.Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from trsp import attach_sba_tri_to_softmax_model  # noqa: E402


# ======== Data: Alpaca JSONL ========

class AlpacaJsonlDataset(Dataset):
    """
    Read Alpaca format JSONL (each line contains instruction/input/output)
    Training target: only supervise output; set prompt = (instruction? + input + "Answer:"),
    labels only cover answer tokens, other positions set to -100 (not counted in loss).
    """
    def __init__(self, path, tokenizer, max_seq_len=2048, answer_prefix="Answer:"):
        self.path = pathlib.Path(path)
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.answer_prefix = answer_prefix

        # Read all samples
        self.samples = []
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                instr = obj.get("instruction", "") or ""
                inp   = obj.get("input", "") or ""
                out   = obj.get("output", "") or ""
                # Construct English prompt
                prefix = (instr.strip() + "\n").strip() if instr else ""
                src = (prefix + inp.rstrip() + f"\n{self.answer_prefix}").strip()
                tgt = " " + out.strip()
                self.samples.append((src, tgt))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        src, tgt = self.samples[idx]
        tok = self.tokenizer

        # Encode: concatenate input and output; labels only supervise output part
        src_ids = tok.encode(src, add_special_tokens=False)
        tgt_ids = tok.encode(tgt + tok.eos_token, add_special_tokens=False)

        # Truncate: limit total length; ensure at least some tokens for answer
        keep_src = min(len(src_ids), self.max_seq_len - 1 - min(len(tgt_ids), 64))
        input_ids = src_ids[:keep_src] + tgt_ids
        input_ids = input_ids[: self.max_seq_len - 1] + [tok.eos_token_id]

        # Padding to max_seq_len
        pad_len = self.max_seq_len - len(input_ids)
        if pad_len > 0:
            input_ids.extend([tok.pad_token_id] * pad_len)

        # labels: src part all -100, only supervise tgt
        labels = [-100] * keep_src + tgt_ids
        labels = labels[: self.max_seq_len - 1] + [-100]  # EOS not supervised
        labels.extend([-100] * pad_len)

        # attention_mask: 1 valid, 0 padding
        attn_mask = [1] * (self.max_seq_len - pad_len) + [0] * pad_len

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attn_mask, dtype=torch.long),
        }


def pad_collate(batch, pad_id):
    # dataset already padded to fixed max_seq_len, protect alignment here
    max_len = max(len(x["input_ids"]) for x in batch)
    input_ids, labels, attn = [], [], []
    for x in batch:
        L = len(x["input_ids"])
        pad_len = max_len - L
        input_ids.append(F.pad(x["input_ids"], (0, pad_len), value=pad_id))
        labels.append(F.pad(x["labels"], (0, pad_len), value=-100))
        attn.append(F.pad(x["attention_mask"], (0, pad_len), value=0))
    return {
        "input_ids": torch.stack(input_ids, dim=0),
        "labels": torch.stack(labels, dim=0),
        "attention_mask": torch.stack(attn, dim=0)
    }


# ======== Model: RoPE Positional Encoding Multi-Head Attention ========

class CausalSelfAttention(nn.Module):
    """ Standard Multi-Head Softmax Attention + RoPE positional encoding, supports long sequences """
    def __init__(self, d_model, n_head, attn_pdrop=0.0, resid_pdrop=0.0, max_seq_len=4096):
        super().__init__()
        assert d_model % n_head == 0
        self.d_model = d_model
        self.n_head = n_head
        self.d_head = d_model // n_head

        # RoPE requires d_head to be even
        assert self.d_head % 2 == 0, f"RoPE requires d_head to be even, current d_head={self.d_head}"

        self.Wqkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.resid_drop = nn.Dropout(resid_pdrop)

        # RoPE parameters
        inv_freq = 1.0 / (10000 ** (torch.arange(0, self.d_head, 2).float() / self.d_head))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def _build_rope_cache(self, T: int, device, dtype):
        pos = torch.arange(T, device=device, dtype=torch.float32)  # (T,)
        freqs = torch.einsum("i,j->ij", pos, self.inv_freq)        # (T, d_head/2)
        emb = torch.cat([freqs, freqs], dim=-1)                    # (T, d_head)
        cos = emb.cos()
        sin = emb.sin()
        return cos.to(dtype), sin.to(dtype)

    def _apply_rotary_pos_emb(self, x, cos, sin):
        # x: (B, h, T, d), cos/sin: (T, d)
        cos = cos.unsqueeze(0).unsqueeze(0)  # (1,1,T,d)
        sin = sin.unsqueeze(0).unsqueeze(0)
        rotate_half = lambda x: torch.cat([-x[..., x.shape[-1]//2:], x[..., :x.shape[-1]//2]], dim=-1)
        return x * cos + rotate_half(x) * sin

    def forward(self, x, attn_mask=None):
        B, T, C = x.shape
        qkv = self.Wqkv(x)  # (B,T,3C)
        q, k, v = qkv.chunk(3, dim=-1)

        # -> (B,h,T,d)
        q = q.view(B, T, self.n_head, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.d_head).transpose(1, 2)

        # RoPE
        cos, sin = self._build_rope_cache(T, x.device, q.dtype)
        q = self._apply_rotary_pos_emb(q, cos, sin)
        k = self._apply_rotary_pos_emb(k, cos, sin)

        # Attention
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)  # (B,h,T,T)

        # Causal mask
        causal = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool))
        att = att.masked_fill(~causal, float("-inf"))
        if attn_mask is not None:
            # (B,1,1,T)
            mask2d = attn_mask[:, None, None, :].bool()
            att = att.masked_fill(~mask2d, float("-inf"))

        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v  # (B,h,T,d)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_drop(self.proj(y))
        return y


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_head, mlp_ratio=4.0, attn_pdrop=0.0, resid_pdrop=0.0, max_seq_len=4096):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(
            d_model, n_head, attn_pdrop, resid_pdrop, max_seq_len=max_seq_len
        )
        self.ln2 = nn.LayerNorm(d_model)
        hidden = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Linear(hidden, d_model),
            nn.Dropout(resid_pdrop),
        )

    def forward(self, x, attn_mask=None):
        x = x + self.attn(self.ln1(x), attn_mask)
        x = x + self.mlp(self.ln2(x))
        return x


class TinySoftmaxLM(nn.Module):
    """
    Small decoder language model (word embedding and LM head weight sharing)
    - Training target: standard next-token cross-entropy (time-shift)
    - By adjusting d_model / n_layer, total parameters can range from ~1M to ~100M+
    """
    def __init__(self, vocab_size=512, d_model=128, n_layer=3, n_head=4, mlp_ratio=4.0, max_seq_len=4096):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.embed = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_head, mlp_ratio, max_seq_len=max_seq_len)
            for _ in range(n_layer)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        # Weight sharing
        self.lm_head.weight = self.embed.weight

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias)
        if isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, input_ids, attention_mask=None, labels=None):
        x = self.embed(input_ids)  # (B,T,C)
        for blk in self.blocks:
            x = blk(x, attention_mask)
        x = self.ln_f(x)
        logits = self.lm_head(x)  # (B,T,V)

        loss = None
        if labels is not None:
            # ==== Standard next-token LM: time-shift ====
            # Use position t logits to predict position t+1 label
            shift_logits = logits[:, :-1, :].contiguous()   # (B, T-1, V)
            shift_labels = labels[:, 1:].contiguous()       # (B, T-1)
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100
            )
        return logits, loss


# ======== Lightning Module (full-answer EM, generation-based) ========

class LitTinyLM(pl.LightningModule):
    """
    Lightning wrapper:
    - Training: next-token cross-entropy
    - Validation/test: autoregressive generation
        * full_answer_em_gen: full answer EM (exact match after normalization)
    """
    def __init__(self, model: TinySoftmaxLM, tokenizer, lr=3e-4, weight_decay=0.01,
                 max_new_tokens=8, answer_prefix="Answer:",
                 use_sba_tri: bool = False):
        super().__init__()
        self.save_hyperparameters(ignore=["model", "tokenizer"])
        self.model = model
        self.tok = tokenizer
        self.answer_prefix = answer_prefix

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay
        )
        t_max = getattr(self.trainer, "max_epochs", 1)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(t_max, 1))
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }

    def training_step(self, batch, batch_idx):
        logits, loss = self.model(
            batch["input_ids"], batch["attention_mask"], batch["labels"]
        )

        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    @staticmethod
    def _normalize_text(s: str):
        s = re.sub(r"\s+", " ", s.strip().lower())
        return s.strip(string.punctuation)

    @torch.no_grad()
    def _greedy_generate_one(self, prefix_ids: torch.Tensor, max_new_tokens: int = 8):
        """
        Autoregressive generation:
        prefix_ids: (T,) only contains prefix up to "Answer:"
        Returns: newly generated token sequence (without prefix), shape (L_new,)
        """
        self.model.eval()
        out = prefix_ids.unsqueeze(0).to(self.device)  # (1,T)
        for _ in range(max_new_tokens):
            logits, _ = self.model(out, attention_mask=None, labels=None)
            next_id = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)  # (1,1)
            out = torch.cat([out, next_id], dim=1)
            if next_id.item() == self.tok.eos_token_id:
                break
        return out[0, prefix_ids.size(0):]  # (L_new,)

    @torch.no_grad()
    def _eval_full_answer_em_gen(self, batch) -> Tuple[int, int]:
        """
        Metric 2: full answer EM (generation-based)

        For each sample:
        - Use labels to find answer start s and end e:
            s = first position where label != -100
            e = first position where label == -100 after s, or T
        - gold_ids = labels[s:e]
        - prefix = input_ids[:s]
        - Generate gen_ids, length at least len(gold_ids) + 2
        - Take first len(gold_ids) tokens for comparison (prevent long answer tail deviation)
        - Exact match after normalized decoding (EM)
        """
        input_ids = batch["input_ids"].to(self.device)       # (B,T)
        labels    = batch["labels"].to(self.device)          # (B,T)
        B, T = labels.shape

        em_hit, total = 0, 0
        for i in range(B):
            lab = labels[i]
            idx = (lab != -100).nonzero(as_tuple=False)
            if len(idx) == 0:
                continue
            s = idx[0].item()
            # Find answer segment end position e
            e = s
            while e < T and lab[e].item() != -100:
                e += 1
            gold_ids = lab[s:e]  # (L_gold,)

            if gold_ids.numel() == 0:
                continue

            prefix_ids = input_ids[i, :s]
            need_len = max(self.hparams.max_new_tokens, gold_ids.numel() + 2)
            gen_ids = self._greedy_generate_one(prefix_ids, max_new_tokens=need_len)

            if gen_ids.numel() == 0:
                continue

            # Only align first L_gold tokens for EM (avoid extra generated part polluting alignment)
            gen_slice = gen_ids[:gold_ids.numel()].cpu().tolist()
            gold_list = gold_ids.cpu().tolist()

            pred = self._normalize_text(
                self.tok.decode(gen_slice, skip_special_tokens=True)
            )
            gold = self._normalize_text(
                self.tok.decode(gold_list, skip_special_tokens=True)
            )

            em_hit += int(pred == gold)
            total += 1

        return em_hit, max(total, 1)

    def validation_step(self, batch, batch_idx):
        logits, loss = self.model(
            batch["input_ids"], batch["attention_mask"], batch["labels"]
        )
        hit_em, total_em = self._eval_full_answer_em_gen(batch)
        acc_em = hit_em / max(total_em, 1)

        self.log("val/loss", loss, prog_bar=True, on_epoch=True, sync_dist=True)
        self.log("val/full_answer_em_gen", acc_em, prog_bar=True,
                 on_epoch=True, sync_dist=True)

    def test_step(self, batch, batch_idx):
        logits, loss = self.model(
            batch["input_ids"], batch["attention_mask"], batch["labels"]
        )
        hit_em, total_em = self._eval_full_answer_em_gen(batch)
        acc_em = hit_em / max(total_em, 1)

        self.log("test/loss", loss, prog_bar=True, on_epoch=True, sync_dist=True)
        self.log("test/full_answer_em_gen", acc_em, prog_bar=True,
                 on_epoch=True, sync_dist=True)


# ======== CLI & Entry ========

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Alpaca JSONL directory (contains train_len1024 / eval_len1024 / test_len2048)")
    parser.add_argument("--llama_tokenizer_path", type=str, required=True,
                        help="Llama-3.2-1B tokenizer path")
    parser.add_argument("--max_seq_len", type=int, default=4096)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--accumulate_grad_batches", type=int, default=1)
    parser.add_argument("--precision", type=str, default="bf16-mixed",
                        help="Optional: 16-mixed / bf16-mixed / 32")
    parser.add_argument("--devices", type=int, default=1,
                        help="Number of GPUs")
    parser.add_argument("--answer_prefix", type=str, default="Answer:")
    parser.add_argument("--max_new_tokens", type=int, default=8)

    # SBA·Tri related
    parser.add_argument("--use_sba_tri", action="store_true",
                        help="Whether to inject SBA·Tri into softmax model")

    args = parser.parse_args()

    data_dir = pathlib.Path(args.data_dir)
    train_path = data_dir / "train_len1024.jsonl"
    eval_path  = data_dir / "eval_len1024.jsonl"
    test_path  = data_dir / "test_len1024.jsonl"
    assert train_path.exists() and eval_path.exists() and test_path.exists(), \
        "Missing train_len1024 / eval_len1024 / test_len1024.jsonl"

    # Load Llama-3.2-1B tokenizer
    tok = AutoTokenizer.from_pretrained(args.llama_tokenizer_path, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    actual_vocab_size = tok.vocab_size + len(tok.added_tokens_decoder)
    print(f"[Info] Tokenizer: reported vocab_size={tok.vocab_size}, "
          f"actual with added_tokens={actual_vocab_size}")

    # Dataset and DataLoader
    train_set = AlpacaJsonlDataset(train_path, tok, args.max_seq_len, args.answer_prefix)
    eval_set  = AlpacaJsonlDataset(eval_path,  tok, args.max_seq_len, args.answer_prefix)
    test_set  = AlpacaJsonlDataset(test_path,  tok, args.max_seq_len, args.answer_prefix)

    collate = lambda batch: pad_collate(batch, pad_id=tok.pad_token_id)
    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=collate, pin_memory=True
    )
    eval_loader = DataLoader(
        eval_set, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate, pin_memory=True
    )
    test_loader = DataLoader(
        test_set, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate, pin_memory=True
    )

    # Build ~100M level model
    model = TinySoftmaxLM(
        vocab_size=actual_vocab_size,
        d_model=512,
        n_layer=14,
        n_head=8,      # 512 / 8 = 64
        mlp_ratio=4.0,
        max_seq_len=args.max_seq_len,
    )

    if args.use_sba_tri:
        model = attach_sba_tri_to_softmax_model(
            model,
            max_L_ctx=args.max_seq_len,
            gamma_hidden_dim=16,
        )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"[Info] Model params: {total_params/1e6:.3f} M (vocab={actual_vocab_size})")

    lit = LitTinyLM(
        model, tok,
        lr=args.lr,
        weight_decay=args.weight_decay,
        max_new_tokens=args.max_new_tokens,
        answer_prefix=args.answer_prefix,
        use_sba_tri=args.use_sba_tri,
    )

    ckpt_name = "checkpoints_tiny_softmax"
    if args.use_sba_tri:
        ckpt_name += "_sba"
    ckpt_dir = data_dir / ckpt_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=args.devices,
        strategy="ddp",
        max_epochs=args.max_epochs,
        precision=args.precision,
        accumulate_grad_batches=args.accumulate_grad_batches,
        default_root_dir=ckpt_dir.as_posix(),
        log_every_n_steps=20,
        gradient_clip_val=1.0,
    )

    trainer.fit(lit, train_loader, eval_loader)
    trainer.test(lit, dataloaders=test_loader)


if __name__ == "__main__":
    main()
