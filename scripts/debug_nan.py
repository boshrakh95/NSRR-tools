#!/usr/bin/env python3
"""
Diagnose NaN source in Phase 0 training on GPU.

Phase 1: spot-check 500 random embedding files for extreme values.
Phase 2: run a full training epoch with ALL subjects; stop at the first NaN
         batch and report which sample caused it.
"""
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

cfg_path = _ROOT / "configs" / "phase0_config.yaml"
with open(cfg_path) as f:
    cfg = yaml.safe_load(f)

from nsrr_tools.datasets.context_window_dataset import ContextWindowDataset
from nsrr_tools.models.sequence_head import LSTMHead

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}\n", flush=True)

# ── Phase 1: spot-check 500 random embedding files ────────────────────────────
print("── Phase 1: spot-checking 500 random embedding files ────────────────", flush=True)
emb_dir = Path(cfg["dataset"]["embedding_dir"])

# Use shape cache to get file list (avoids slow rglob on network FS)
with open(emb_dir / "shape_cache.json") as f:
    cache = json.load(f)
all_keys = list(cache.keys())   # "dataset/subject_id"
print(f"  Total files: {len(all_keys)}", flush=True)

random.seed(0)
sample_keys = random.sample(all_keys, min(500, len(all_keys)))

THRESHOLD = 20.0
extreme = []
global_max = 0.0

for key in sample_keys:
    p = emb_dir / f"{key}.npy"
    x = np.load(p, mmap_mode="r")           # memmap, no full load
    T = x.shape[0]
    idx = np.random.randint(0, T, size=min(20, T))
    amax = float(np.abs(x[idx].astype(np.float32)).max())
    if amax > global_max:
        global_max = amax
    if amax > THRESHOLD:
        extreme.append((key, amax))

print(f"  Spot-check max |value|: {global_max:.4f}", flush=True)
print(f"  Files with sampled |value| > {THRESHOLD}: {len(extreme)}", flush=True)
for name, val in sorted(extreme, key=lambda t: -t[1])[:10]:
    print(f"    {name}  max={val:.2f}", flush=True)

# ── Phase 2: full training epoch, stop at first NaN ───────────────────────────
print("\n── Phase 2: full training epoch with all subjects ───────────────────", flush=True)

ds = ContextWindowDataset(
    cfg=cfg, split="train",
    context_length="30s",
    task="apnea_binary", task_type="seq2label",
)
loader = torch.utils.data.DataLoader(
    ds, batch_size=32, shuffle=True, num_workers=0,
)
print(f"  Dataset: {len(ds)} items, {len(loader)} batches", flush=True)

model = LSTMHead(input_dim=512, hidden_dim=256, num_layers=2,
                 num_classes=2, dropout=0.3).to(device)
model.train()

# Auto class weights (mirrors training script)
train_labels = np.array([entry[2] for entry in ds._index])
counts = np.bincount(train_labels, minlength=2).astype(float)
counts = np.where(counts == 0, 1.0, counts)
w = len(train_labels) / (2 * counts)
w = w / w.sum() * 2
print(f"  Auto class weights: {np.round(w, 4).tolist()}", flush=True)
criterion = nn.CrossEntropyLoss(
    weight=torch.tensor(w, dtype=torch.float32, device=device)
)
optim = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

for bi, (x, mask, y) in enumerate(loader):
    x_raw = x.clone()
    x, mask, y = x.to(device), mask.to(device), y.to(device)

    optim.zero_grad()
    logits = model(x, mask)
    loss = criterion(logits, y)

    nan_loss   = torch.isnan(loss).item()
    nan_logits = torch.isnan(logits).any().item()
    nan_x      = torch.isnan(x).any().item()

    if bi % 100 == 0 or nan_loss or nan_logits:
        print(f"  Batch {bi:4d}: x_max={x_raw.abs().max():.3f}  "
              f"logits=[{logits.min().item():.3f}, {logits.max().item():.3f}]  "
              f"loss={loss.item():.4f}  "
              f"NaN(x={nan_x} logits={nan_logits} loss={nan_loss})",
              flush=True)

    if nan_loss or nan_logits:
        print(f"\n  *** NaN at batch {bi} ***", flush=True)
        for si in range(x_raw.shape[0]):
            smax = x_raw[si].abs().max().item()
            print(f"    sample {si}: x_max={smax:.3f}", flush=True)
        loss.backward()
        bad_params = [n for n, p in model.named_parameters()
                      if p.grad is not None and torch.isnan(p.grad).any()]
        print(f"  Params with NaN grad: {bad_params}", flush=True)
        break

    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optim.step()
else:
    print(f"\n  Full epoch done ({len(loader)} batches) — NO NaN detected!", flush=True)

print("\nDone.", flush=True)
