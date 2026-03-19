#!/usr/bin/env python3
"""
sequence_head.py — Phase 0, Step 3

Lightweight sequence heads that sit on top of frozen SleepFM embeddings.

All heads share the same interface and always output (B, num_classes):

    Input:  x    (B, N, 512)   — context window of flattened embeddings
            mask (B, N)  bool  — True = padded patch (no real signal)
    Output: logits (B, num_classes)

Both task types (seq2label and anchor-based seq2seq) now produce a scalar
label, so a single output shape covers all cases.

Available heads
───────────────
  MeanPool       : masked mean over time → linear.
                   No temporal order used. Useful baseline.

  LSTMHead       : BiLSTM, last valid hidden state → linear.
                   Processes the full context window left-to-right and
                   right-to-left; final state concatenates both directions.
                   For the anchor-based staging task the window is already
                   past-only (causal by construction), so BiLSTM is valid —
                   there is no future leakage.  A unidirectional LSTM would
                   be more appropriate for the final causal model (Phase 1+).

  TransformerHead: CLS token + sinusoidal PE → transformer encoder → linear.
                   Full self-attention within the window.  Capped at 80m
                   (960 patches) due to O(N²) memory; skip for full_night.

Factory
───────
  build_head(cfg) → nn.Module
"""

import math

import torch
import torch.nn as nn


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Masked mean over time (non-padded positions only).

    Args:
        x    : (B, N, D)
        mask : (B, N) bool — True = padded (excluded)

    Returns:
        (B, D)
    """
    valid = (~mask).float().unsqueeze(-1)   # (B, N, 1)
    denom = valid.sum(dim=1).clamp(min=1)   # (B, 1)
    return (x * valid).sum(dim=1) / denom   # (B, D)


# ─────────────────────────────────────────────────────────────────────────────
# Heads
# ─────────────────────────────────────────────────────────────────────────────

class MeanPoolHead(nn.Module):
    """Masked temporal mean-pool → linear classifier.

    No temporal order is used.  Treats the context window as a bag of patches.
    Fast, parameter-light baseline.
    """

    def __init__(self, input_dim: int, num_classes: int, dropout: float = 0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.fc      = nn.Linear(input_dim, num_classes)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x    : (B, N, input_dim)
            mask : (B, N) bool

        Returns:
            logits (B, num_classes)
        """
        pooled = _masked_mean(x, mask)            # (B, input_dim)
        return self.fc(self.dropout(pooled))


class LSTMHead(nn.Module):
    """BiLSTM → last valid hidden state → linear classifier.

    Uses pack_padded_sequence to skip padded patches during the RNN pass.
    The last valid hidden state is the concatenation of the final forward
    and backward states, giving the head access to the whole window.
    """

    def __init__(
        self,
        input_dim:   int,
        hidden_dim:  int,
        num_layers:  int,
        num_classes: int,
        dropout:     float = 0.0,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc      = nn.Linear(hidden_dim * 2, num_classes)  # ×2 for BiLSTM

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x    : (B, N, input_dim)
            mask : (B, N) bool  — True = padded

        Returns:
            logits (B, num_classes)
        """
        lengths = (~mask).long().sum(dim=1).cpu().clamp(min=1)  # (B,)

        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths, batch_first=True, enforce_sorted=False
        )
        _, (h_n, _) = self.lstm(packed)
        # h_n : (num_layers * 2, B, hidden_dim)
        # Last layer: forward = h_n[-2], backward = h_n[-1]
        h = torch.cat([h_n[-2], h_n[-1]], dim=-1)   # (B, hidden_dim * 2)
        return self.fc(self.dropout(h))


class TransformerHead(nn.Module):
    """Transformer encoder with a prepended CLS token → linear classifier.

    Sinusoidal positional encoding (non-trainable).  Pre-LN for stability.
    CLS token output → classifier.

    Memory note: attention is O(N²) in sequence length.
    Practical limit ≈ 80m context (960 patches) on a 16 GB GPU at batch 32.
    Do not use with full_night context.
    """

    def __init__(
        self,
        input_dim:   int,
        hidden_dim:  int,
        num_layers:  int,
        num_heads:   int,
        num_classes: int,
        dropout:     float = 0.0,
        max_seq_len: int   = 2048,
    ):
        super().__init__()

        self.input_proj = (
            nn.Identity() if input_dim == hidden_dim
            else nn.Linear(input_dim, hidden_dim)
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))

        self.register_buffer(
            "pos_enc", self._make_pos_enc(max_seq_len + 1, hidden_dim)
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,    # Pre-LN — more stable than Post-LN
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.dropout     = nn.Dropout(dropout)
        self.fc          = nn.Linear(hidden_dim, num_classes)

    @staticmethod
    def _make_pos_enc(max_len: int, d_model: int) -> torch.Tensor:
        pe  = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div[:d_model // 2])
        return pe.unsqueeze(0)   # (1, max_len, d_model)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x    : (B, N, input_dim)
            mask : (B, N) bool  — True = padded

        Returns:
            logits (B, num_classes)
        """
        B, N, _ = x.shape
        x = self.input_proj(x)                           # (B, N, hidden_dim)

        # Prepend CLS token and add positional encoding
        cls = self.cls_token.expand(B, -1, -1)           # (B, 1, hidden_dim)
        x   = torch.cat([cls, x], dim=1)                 # (B, N+1, hidden_dim)
        x   = x + self.pos_enc[:, : N + 1, :]

        # Key-padding mask: True = ignore position
        # CLS (position 0) is never masked
        cls_mask = torch.zeros(B, 1, dtype=torch.bool, device=mask.device)
        key_mask = torch.cat([cls_mask, mask], dim=1)    # (B, N+1)

        # Convert bool mask → float additive mask to avoid sympy import inside
        # PyTorch's bool-mask validation path
        key_mask_f = key_mask.float().masked_fill(key_mask, float("-inf"))

        out     = self.transformer(x, src_key_padding_mask=key_mask_f)
        cls_out = out[:, 0, :]                           # (B, hidden_dim)
        return self.fc(self.dropout(cls_out))


# ─────────────────────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────────────────────

def build_head(cfg: dict) -> nn.Module:
    """Instantiate the configured head from phase0_config.

    Reads cfg["model"].  num_classes must be patched in before calling
    (train_context_sweep.py does this from the dataset).

    Args:
        cfg : Full phase0_config dict.

    Returns:
        nn.Module with forward(x, mask) → logits (B, num_classes).
    """
    m         = cfg["model"]
    head_type = m["head_type"]

    input_dim   = m["input_dim"]
    num_classes = m["num_classes"]
    hidden_dim  = m["hidden_dim"]
    num_layers  = m["num_layers"]
    dropout     = m["dropout"]

    if head_type == "mean_pool":
        return MeanPoolHead(input_dim, num_classes, dropout)

    elif head_type == "lstm":
        return LSTMHead(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_classes=num_classes,
            dropout=dropout,
        )

    elif head_type == "transformer":
        num_heads = m.get("num_heads", 8)
        return TransformerHead(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            num_classes=num_classes,
            dropout=dropout,
        )

    else:
        raise ValueError(
            f"Unknown head_type: {head_type!r}. "
            "Choose 'mean_pool', 'lstm', or 'transformer'."
        )
