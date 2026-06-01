# -*- coding: utf-8 -*-
"""
Reference example: LoRA fine-tuning Llama-3.2-1B + TRSP / SBA.Tri.

This script is provided as a *reference implementation* of how to plug
SBA.Tri into a HuggingFace Llama model with PEFT-LoRA.  The TRSP method
itself lives in the ``trsp`` package; this file only contains the data
pipeline and the Lightning training loop used in the paper.

- Data: Alpaca JSON array format (instruction/input/output)
- Method: PEFT LoRA + SBA.Tri in parallel for each layer
- Training: cross-entropy (training set only)
"""

import os, json, argparse, pathlib, re, math
import time
import random
from typing import List, Dict, Tuple, Optional

import torch
import torch.distributed
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    BitsAndBytesConfig
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)

import numpy as np

# ---- TRSP / SBA.Tri ----
# Make the repo root importable so ``import trsp`` works when this script
# is executed directly from the ``examples/`` directory.
import sys
_THIS_DIR = pathlib.Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from trsp import attach_sba_tri_to_llama  # noqa: E402


# ======== Random Seed Setting ========

def set_seed(seed: int = 42):
    """
    Set random seed to ensure experiment reproducibility
    
    Args:
        seed: Random seed value, default is 42
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Set PyTorch deterministic mode (may reduce performance but improve reproducibility)
    # Note: do not enable torch.use_deterministic_algorithms because:
    # 1. Need to set CUBLAS_WORKSPACE_CONFIG environment variable (must be before PyTorch initialization)
    # 2. Will significantly reduce performance (10-30%)
    # 3. For most experiments, only setting seed is sufficient for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # If full determinism is needed (e.g. paper reproduction), uncomment below and set environment variable:
    # import os
    # os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'  # Must be set before PyTorch initialization
    # torch.use_deterministic_algorithms(True, warn_only=True)
    print(f"[Info] Random seed set to: {seed}")

# ======== Data: Alpaca JSON Array ========

class AlpacaJsonDataset(Dataset):
    """
    Read Alpaca format JSON array (each element contains instruction/input/output)
    Training target: only supervise output; set prompt = (instruction? + input + "Answer:"),
    labels only cover answer tokens, other positions set to -100 (not counted in loss).
    """
    def __init__(self, path, tokenizer, max_seq_len=2048, answer_prefix="Answer:"):
        self.path = pathlib.Path(path)
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.answer_prefix = answer_prefix

        # Read all samples (JSON array format)
        self.samples = []
        with self.path.open("r", encoding="utf-8") as f:
            data = json.load(f)
            assert isinstance(data, list), "Data file should be JSON array format"
            for obj in data:
                instr = obj.get("instruction", "") or ""
                inp   = obj.get("input", "") or ""
                out   = obj.get("output", "") or ""
                # Construct prompt
                prefix = (instr.strip() + "\n").strip() if instr else ""
                src = (prefix + inp.rstrip() + f"\n{self.answer_prefix}").strip()
                tgt = " " + out.strip()
                self.samples.append((src, tgt, instr, inp))  # Save original instruction and input

        print(f"[Info] Loaded: {len(self.samples)}  from {self.path}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        src, tgt, instruction, input_text = self.samples[idx]
        tok = self.tokenizer

        # Use chat_template to format prompt (if available)
        if hasattr(tok, "chat_template") and tok.chat_template is not None:
            # Merge instruction and input as user message content
            user_content = ""
            if instruction:
                user_content = instruction.strip()
            if input_text:
                if user_content:
                    user_content = user_content + "\n" + input_text.strip()
                else:
                    user_content = input_text.strip()
            
            # If no content, fallback to original src (remove Answer: part)
            if not user_content:
                parts = src.rsplit(f"\n{self.answer_prefix}", 1)
                user_content = parts[0] if parts else src
            
            # Use chat_template to format
            messages = [{"role": "user", "content": user_content}]
            formatted_prompt = tok.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            src = formatted_prompt.rstrip()

        # Encode: concatenate input and output; labels only supervise output part
        src_ids = tok.encode(src, add_special_tokens=False)
        tgt_ids = tok.encode(tgt + tok.eos_token, add_special_tokens=False)

        # Truncate: limit total length
        keep_src = min(len(src_ids), self.max_seq_len - 1 - min(len(tgt_ids), 64))
        input_ids = src_ids[:keep_src] + tgt_ids
        input_ids = input_ids[: self.max_seq_len - 1] + [tok.eos_token_id]

        # labels: src part all -100, only supervise tgt
        labels = [-100] * keep_src + tgt_ids
        labels = labels[: self.max_seq_len - 1] + [-100]  # EOS not supervised

        attn_mask = [1] * len(input_ids)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attn_mask, dtype=torch.long),
        }

def pad_collate(batch, pad_id):
    # Dynamic padding; right-align input_ids / labels / attention_mask
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

# ======== Lightning Module ========

class LitLlamaLora(pl.LightningModule):
    """
    Lightning wrapper:
    - Training: cross-entropy
    """
    def __init__(self, model, tokenizer, lr=2e-4, weight_decay=0.01,
                 answer_prefix="Answer:",
                 use_sba_tri: bool = True):
        super().__init__()
        self.save_hyperparameters(ignore=["model", "tokenizer"])
        self.model = model
        self.tok = tokenizer
        self.answer_prefix = answer_prefix
        self.use_sba_tri = use_sba_tri
        self.start_time = None

    def on_train_start(self):
        """Record start time when training begins"""
        self.start_time = time.time()
        print(f"[Info] Training start time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self.start_time))}")

    def configure_optimizers(self):
        # AdamW + cosine annealing
        optimizer = torch.optim.AdamW(
            self.parameters(), 
            lr=self.hparams.lr, 
            weight_decay=self.hparams.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=self.trainer.max_epochs
        )
        return {
            "optimizer": optimizer, 
            "lr_scheduler": {
                "scheduler": scheduler, 
                "interval": "epoch"
            }
        }

    def training_step(self, batch, batch_idx):
        outputs = self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"]
        )
        loss = outputs.loss

        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def on_train_end(self):
        """Calculate total time when training ends"""
        if self.start_time is not None:
            end_time = time.time()
            total_time = end_time - self.start_time
            hours = int(total_time // 3600)
            minutes = int((total_time % 3600) // 60)
            seconds = int(total_time % 60)
            print(f"\n[Info] Training completed！")
            print(f"[Info] Training end time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))}")
            print(f"[Info] Total training time: {hours} hours {minutes} minutes {seconds} seconds ({total_time:.2f} seconds)")


# ======== CLI & Entry ========

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True, 
                       help="Alpaca JSON array file path")
    parser.add_argument("--model_name", type=str, required=True,
                       help="HuggingFace model name or local path")
    parser.add_argument("--max_seq_len", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--accumulate_grad_batches", type=int, default=4)
    parser.add_argument("--precision", type=str, default="bf16-mixed", 
                       help="Optional: 16-mixed / bf16-mixed / 32")
    parser.add_argument("--devices", type=int, default=1, help="Number of GPUs")
    parser.add_argument("--answer_prefix", type=str, default="Answer:")
    parser.add_argument("--lora_r", type=int, default=8, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.1, help="LoRA dropout")
    parser.add_argument("--use_4bit", action="store_true", help="Using.*quantization")
    parser.add_argument("--output_dir", type=str, default=None, help="Model output directory")
    args = parser.parse_args()

    # Set random seed to ensure experiment reproducibility
    set_seed(seed=42)

    data_path = pathlib.Path(args.data_path)
    assert data_path.exists(), f"Data file does not exist: {data_path}"

    # Loading tokenizer and model
    print(f"[Info] Loading model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    
    # Set pad_token (if not exists)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Loading model (optional 4-bit quantization)
    if args.use_4bit:
        print("[Info] Using.*quantization")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16
        )
        model = prepare_model_for_kbit_training(model)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )

    # Configuring LoRA
    print(f"[Info] Configuring LoRA: r={args.lora_r}, alpha={args.lora_alpha}, dropout={args.lora_dropout}")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
    )
    model = get_peft_model(model, lora_config)

    # Inject SBA·Tri on LoRA bone, parallel triangular kernel + length-aware gating for each layer
    model = attach_sba_tri_to_llama(
        model,
        max_L_ctx=args.max_seq_len,
        gamma_hidden_dim=16
    )

    model.print_trainable_parameters()

    # Dataset and DataLoader (training set)
    train_set = AlpacaJsonDataset(data_path, tokenizer, args.max_seq_len, args.answer_prefix)

    collate = lambda batch: pad_collate(batch, pad_id=tokenizer.pad_token_id)
    train_loader = DataLoader(
        train_set, 
        batch_size=args.batch_size, 
        shuffle=True,
        num_workers=args.num_workers, 
        collate_fn=collate, 
        pin_memory=True
    )

    # Lightning module (default enable SBA·Tri)
    lit = LitLlamaLora(
        model,
        tokenizer,
        lr=args.lr,
        weight_decay=args.weight_decay,
        answer_prefix=args.answer_prefix,
        use_sba_tri=True,
    )

    # Lightning Trainer
    if args.output_dir is None:
        output_dir = pathlib.Path("./checkpoints_llama32_lora_sba")
    else:
        output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_callback = ModelCheckpoint(
        dirpath=str(output_dir),
        filename="epoch{epoch:02d}-train_loss{train/loss:.3f}",
        monitor="train/loss",
        mode="min",
        save_top_k=3,
        save_last=True,
    )

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=args.devices,
        strategy="ddp" if args.devices > 1 else "auto",
        max_epochs=args.max_epochs,
        precision=args.precision,
        accumulate_grad_batches=args.accumulate_grad_batches,
        default_root_dir=str(output_dir),
        callbacks=[checkpoint_callback],
        log_every_n_steps=20,
        gradient_clip_val=1.0,
    )

    # Training (training set only)
    print("[Info] Starting training (LoRA + SBA·Tri)...")
    trainer.fit(lit, train_loader)
    
    # Save final model
    final_model_path = output_dir / "final_model"
    print(f"[Info] Saving final model to: {final_model_path}")
    model.save_pretrained(str(final_model_path))
    tokenizer.save_pretrained(str(final_model_path))

if __name__ == "__main__":
    main()
