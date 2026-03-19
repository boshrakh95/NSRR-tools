#!/usr/bin/env python3
"""
sequence_head.py — Phase 0, Step 3

Lightweight sequence heads that sit on top of frozen SleepFM embeddings.

All heads share the same interface:

    Input:  x    (B, N, input_dim)   — context window of flattened embeddings
            mask (B, N)  bool        — True = padded patch (ignored)
    Output: logits  (B, num_classes)             for seq2label
                    (B, N, num_classes)           for seq2seq

Available heads
───────────────
  MeanPool       : masked mean → linear. Baseline. Seq2label only.
  LSTMHead       : BiLSTM → last state (seq2label) or all states (seq2seq).
  TransformerHead: CLS token or all tokens → linear.

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
    """Compute mean over non-padded positions.

    Args:
        x    : (B, N, D)
        mask : (B, N) bool — True = padded (excluded from mean)

    Returns:
        (B, D)
    """
    # Invert: valid = ~mask
    valid = (~mask).float().unsqueeze(-1)   # (B, N, 1)
    denom = valid.sum(dim=1).clamp(min=1)   # (B, 1)
    return (x * valid).sum(dim=1) / denom   # (B, D)


# ─────────────────────────────────────────────────────────────────────────────
# Heads
# ─────────────────────────────────────────────────────────────────────────────

class MeanPoolHead(nn.Module):
    """Masked mean-pool over time → linear classifier.

    Suitable for seq2label only (returns a single logit vector per sample).
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
        pooled = _masked_mean(x, mask)           # (B, input_dim)
        return self.fc(self.dropout(pooled))


class LSTMHead(nn.Module):
    """Bidirectional LSTM sequence head.

    seq2label: uses last valid hidden state (via sequence length) → linear.
    seq2seq:   uses all timestep outputs → linear (one per patch).

    Note: padded patches are handled via pack_padded_sequence for efficiency.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        num_classes: int,
        dropout: float = 0.0,
        task_type: str = "seq2label",
    ):
        super().__init__()
        self.task_type  = task_type
        self.hidden_dim = hidden_dim

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        # BiLSTM doubles hidden dim
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x    : (B, N, input_dim)
            mask : (B, N) bool  True=padded

        Returns:
            seq2label : (B, num_classes)
            seq2seq   : (B, N, num_classes)
        """
        B, N, _ = x.shape
        # Sequence lengths (number of valid patches per sample)
        lengths = (~mask).long().sum(dim=1).cpu()       # (B,)
        lengths = lengths.clamp(min=1)                   # avoid 0-length

        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths, batch_first=True, enforce_sorted=False
        )
        out_packed, (h_n, _) = self.lstm(packed)

        if self.task_type == "seq2label":
            # Concatenate final forward and backward hidden states
            # h_n shape: (num_layers * 2, B, hidden_dim)
            fwd = h_n[-2]   # last layer, forward
            bwd = h_n[-1]   # last layer, backward
            h = torch.cat([fwd, bwd], dim=-1)             # (B, hidden_dim*2)
            return self.fc(self.dropout(h))

        else:  # seq2seq
            out, _ = nn.utils.rnn.pad_packed_sequence(
                out_packed, batch_first=True, total_length=N
            )  # (B, N, hidden_dim*2)
            return self.fc(self.dropout(out))             # (B, N, num_classes)


class TransformerHead(nn.Module):
    """Transformer encoder head with a prepended CLS token.

    seq2label: CLS token output → linear.
    seq2seq:   all patch token outputs → linear.

    Uses sinusoidal positional encoding.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        num_heads: int,
        num_classes: int,
        dropout: float = 0.0,
        task_type: str = "seq2label",
        max_seq_len: int = 2048,
    ):
        super().__init__()
        self.task_type = task_type
        self.hidden_dim = hidden_dim

        # Project from embedding dim to transformer hidden dim
        self.input_proj = (
            nn.Identity() if input_dim == hidden_dim
            else nn.Linear(input_dim, hidden_dim)
        )

        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))

        # Sinusoidal positional encoding (non-trainable)
        self.register_buffer(
            "pos_enc", self._make_pos_enc(max_seq_len + 1, hidden_dim)
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,   # Pre-LN for training stability
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, num_classes)

    @staticmethod
    def _make_pos_enc(max_len: int, d_model: int) -> torch.Tensor:
        pe = torch.zeros(max_len, d_model)
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
            mask : (B, N) bool  True=padded

        Returns:
            seq2label : (B, num_classes)
            seq2seq   : (B, N, num_classes)
        """
        B, N, _ = x.shape
        x = self.input_proj(x)                          # (B, N, hidden_dim)

        # Prepend CLS token
        cls = self.cls_token.expand(B, -1, -1)          # (B, 1, hidden_dim)
        x = torch.cat([cls, x], dim=1)                  # (B, N+1, hidden_dim)

        # Positional encoding
        x = x + self.pos_enc[:, : N + 1, :]

        # Key padding mask: True=ignore
        # CLS token (position 0) is never masked
        cls_mask = torch.zeros(B, 1, dtype=torch.bool, device=mask.device)
        key_mask = torch.cat([cls_mask, mask], dim=1)   # (B, N+1)

        # PyTorch's bool key_padding_mask validation imports sympy (not always installed).
        # Convert to float additive mask (0.0 = attend, -inf = ignore) to avoid this.
        key_mask_f = key_mask.float().masked_fill(key_mask, float("-inf"))
        out = self.transformer(x, src_key_padding_mask=key_mask_f)  # (B, N+1, hidden_dim)

        if self.task_type == "seq2label":
            cls_out = out[:, 0, :]                      # (B, hidden_dim)
            return self.fc(self.dropout(cls_out))

        else:  # seq2seq — skip CLS, use patch tokens
            patch_out = out[:, 1:, :]                   # (B, N, hidden_dim)
            return self.fc(self.dropout(patch_out))      # (B, N, num_classes)


# ─────────────────────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────────────────────

def build_head(cfg: dict) -> nn.Module:
    """Instantiate the appropriate head from the phase0_config model section.

    Args:
        cfg : Full phase0_config dict (reads cfg["model"] and cfg["dataset"]).

    Returns:
        nn.Module implementing the chosen head.
    """
    m_cfg     = cfg["model"]
    head_type = m_cfg["head_type"]
    task_type = m_cfg.get("task_type") or cfg["dataset"]["task_type"]

    input_dim  = m_cfg["input_dim"]
    num_classes= m_cfg["num_classes"]
    hidden_dim = m_cfg["hidden_dim"]
    num_layers = m_cfg["num_layers"]
    dropout    = m_cfg["dropout"]

    if head_type == "mean_pool":
        if task_type != "seq2label":
            raise ValueError("MeanPoolHead only supports seq2label.")
        return MeanPoolHead(input_dim, num_classes, dropout)

    elif head_type == "lstm":
        return LSTMHead(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_classes=num_classes,
            dropout=dropout,
            task_type=task_type,
        )

    elif head_type == "transformer":
        # num_heads must divide hidden_dim
        num_heads = m_cfg.get("num_heads", 8)
        return TransformerHead(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            num_classes=num_classes,
            dropout=dropout,
            task_type=task_type,
        )

    else:
        raise ValueError(
            f"Unknown head_type: {head_type!r}. "
            "Choose 'mean_pool', 'lstm', or 'transformer'."
        )
