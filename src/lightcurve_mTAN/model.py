"""
model.py
--------
mTAN (Multi-Time Attention Network) Autoencoder for irregular time series.

Reference: "Multi-Time Attention Networks for Irregularly Sampled Time Series"
           Shukla & Marlin, ICLR 2021.  https://arxiv.org/abs/2101.10318

Architecture
------------
Encoder:
    - Time embedding φ(t) via learned sinusoidal basis
    - K learned reference time points {s_k}
    - mTAN attention: for each s_k, attend over all input times t_i
    - Output: fixed (K, d_model) context → flatten → Linear → z ∈ R^{d_z}

Decoder (symmetric):
    - z → Linear → (K, d_model) context
    - mTAN attention: for each query time t_i, attend over K reference points
    - Output: (T, 2) reconstructed [magpsf, sigmapsf] at each input time
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Time embedding ────────────────────────────────────────────────────────────

class TimeEmbedding(nn.Module):
    """
    Learned sinusoidal time embedding (time2vec style).
    Maps a scalar time t to a d-dimensional vector:
        φ(t) = [w_0·t + b_0, sin(w_1·t + b_1), ..., sin(w_{d-1}·t + b_{d-1})]
    The first component is linear; the rest are periodic.
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        self.W = nn.Parameter(torch.randn(d_model) * 0.01)
        self.b = nn.Parameter(torch.randn(d_model) * 0.01)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: (...,) → (..., d_model)
        t = t.unsqueeze(-1)                                      # (..., 1)
        z = t * self.W + self.b                                  # (..., d_model)
        # Concatenate linear first dim with periodic remaining dims.
        # Avoid in-place ops so autograd can differentiate cleanly.
        return torch.cat([z[..., :1], torch.sin(z[..., 1:])], dim=-1)


# ── mTAN attention block ──────────────────────────────────────────────────────

class mTANAttention(nn.Module):
    """
    One mTAN attention layer.

    Given:
        query_times:  (B, Q)      — times at which we want representations
        key_times:    (B, K)      — times at which we have values
        values:       (B, K, V)   — value vectors at key_times

    Returns:
        context:      (B, Q, d_model)
    """

    def __init__(self, d_model: int, d_time: int, n_heads: int = 4):
        super().__init__()
        assert d_model % n_heads == 0

        self.n_heads  = n_heads
        self.d_head   = d_model // n_heads
        self.d_model  = d_model

        self.time_emb = TimeEmbedding(d_time)

        # Project time embeddings to query / key spaces
        self.q_proj   = nn.Linear(d_time, d_model)
        self.k_proj   = nn.Linear(d_time, d_model)
        self.v_proj   = nn.Linear(values_dim := d_model, d_model)  # values already d_model
        self.out_proj = nn.Linear(d_model, d_model)

        self._values_dim = values_dim

    # allow the value dimension to be set after init
    def set_value_dim(self, v_dim: int):
        self.v_proj = nn.Linear(v_dim, self.d_model)

    def forward(
        self,
        query_times: torch.Tensor,   # (B, Q)
        key_times:   torch.Tensor,   # (B, K)
        values:      torch.Tensor,   # (B, K, V)
        key_mask:    torch.Tensor | None = None,  # (B, K) bool, True = valid
    ) -> torch.Tensor:
        B, Q = query_times.shape
        _, K = key_times.shape

        # Time embeddings
        q_emb = self.time_emb(query_times)           # (B, Q, d_time)
        k_emb = self.time_emb(key_times)             # (B, K, d_time)

        # Project to d_model
        Q_ = self.q_proj(q_emb)                      # (B, Q, d_model)
        K_ = self.k_proj(k_emb)                      # (B, K, d_model)
        V_ = self.v_proj(values)                     # (B, K, d_model)

        # Multi-head split: (B, heads, seq, d_head)
        def split_heads(x):
            return x.view(x.shape[0], x.shape[1], self.n_heads, self.d_head).transpose(1, 2)

        Q_ = split_heads(Q_)                         # (B, H, Q, d_head)
        K_ = split_heads(K_)                         # (B, H, K, d_head)
        V_ = split_heads(V_)                         # (B, H, K, d_head)

        # Scaled dot-product attention
        scale  = math.sqrt(self.d_head)
        scores = torch.matmul(Q_, K_.transpose(-2, -1)) / scale  # (B, H, Q, K)

        if key_mask is not None:
            # mask out padding positions (False = padded)
            inf_mask = (~key_mask).unsqueeze(1).unsqueeze(2)     # (B, 1, 1, K)
            scores   = scores.masked_fill(inf_mask, -1e9)

        attn    = F.softmax(scores, dim=-1)           # (B, H, Q, K)
        context = torch.matmul(attn, V_)             # (B, H, Q, d_head)

        # Merge heads
        context = context.transpose(1, 2).contiguous().view(B, Q, self.d_model)
        return self.out_proj(context)                # (B, Q, d_model)


# ── Encoder ───────────────────────────────────────────────────────────────────

class mTANEncoder(nn.Module):
    """
    Encodes a variable-length irregular sequence to a fixed latent vector z.

    Input:
        x:    (B, T, 5)  — [t, mag, sig, g, r]
        mask: (B, T)     — True where real

    Output:
        z: (B, d_latent)
    """

    def __init__(
        self,
        d_input:   int = 5,
        d_model:   int = 64,
        d_time:    int = 16,
        d_latent:  int = 64,
        n_ref:     int = 64,    # number of learned reference time points
        n_heads:   int = 4,
        n_layers:  int = 2,
        dropout:   float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_ref   = n_ref

        # Learned reference time points in [0, 1]
        self.ref_times = nn.Parameter(torch.linspace(0, 1, n_ref))

        # Project input features (excluding time) to d_model
        self.input_proj = nn.Linear(d_input - 1, d_model)  # -1: time handled separately via embedding

        # Stack of mTAN layers
        self.layers = nn.ModuleList([
            mTANAttention(d_model=d_model, d_time=d_time, n_heads=n_heads)
            for _ in range(n_layers)
        ])
        for layer in self.layers:
            layer.set_value_dim(d_model)

        self.norm    = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.to_z    = nn.Sequential(
            nn.Linear(n_ref * d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_latent),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape

        times    = x[:, :, 0]                        # (B, T)
        features = x[:, :, 1:]                       # (B, T, 4)
        values   = self.dropout(self.input_proj(features))  # (B, T, d_model)

        # Broadcast reference times to batch
        ref = self.ref_times.unsqueeze(0).expand(B, -1)     # (B, n_ref)

        # Attend: reference times query over input times
        for layer in self.layers:
            context = layer(ref, times, values, key_mask=mask)  # (B, n_ref, d_model)
            values_ref = self.norm(context)
            # For subsequent layers, keys/values become the reference representations
            # (we re-attend from reference to input each layer for simplicity)

        # Flatten and project to latent
        flat = values_ref.reshape(B, -1)             # (B, n_ref * d_model)
        z    = self.to_z(flat)                       # (B, d_latent)
        return z


# ── Decoder ───────────────────────────────────────────────────────────────────

class mTANDecoder(nn.Module):
    """
    Decodes z back to observations at arbitrary query times.

    Input:
        z:           (B, d_latent)
        query_times: (B, T)        — the original observation times

    Output:
        out: (B, T, 2)  — reconstructed [magpsf_norm, sigmapsf_norm]
    """

    def __init__(
        self,
        d_model:  int = 64,
        d_time:   int = 16,
        d_latent: int = 64,
        n_ref:    int = 64,
        n_heads:  int = 4,
        n_layers: int = 2,
        dropout:  float = 0.1,
    ):
        super().__init__()
        self.n_ref   = n_ref
        self.d_model = d_model

        # Expand z to a set of reference-point representations
        self.from_z = nn.Sequential(
            nn.Linear(d_latent, d_model),
            nn.GELU(),
            nn.Linear(d_model, n_ref * d_model),
        )

        # Learned reference time points (separate from encoder's)
        self.ref_times = nn.Parameter(torch.linspace(0, 1, n_ref))

        self.layers = nn.ModuleList([
            mTANAttention(d_model=d_model, d_time=d_time, n_heads=n_heads)
            for _ in range(n_layers)
        ])
        for layer in self.layers:
            layer.set_value_dim(d_model)

        self.norm    = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        # Projects filter one-hot (dim=2) into d_model so the decoder
        # can produce different outputs for g vs r at the same time
        self.filter_inject = nn.Linear(2, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, 2)        # → magpsf, sigmapsf

    def forward(
        self,
        z:            torch.Tensor,   # (B, d_latent)
        query_times:  torch.Tensor,   # (B, T)
        filter_onehot: torch.Tensor,  # (B, T, 2)  — which filter each obs belongs to
    ) -> torch.Tensor:
        B, T = query_times.shape

        # Expand latent to reference representations
        ref_vals = self.from_z(z).view(B, self.n_ref, self.d_model)  # (B, n_ref, d_model)
        ref_vals = self.dropout(ref_vals)

        ref = self.ref_times.unsqueeze(0).expand(B, -1)               # (B, n_ref)

        # Attend: query times attend over reference points
        for layer in self.layers:
            out = layer(query_times, ref, ref_vals)                    # (B, T, d_model)
            ref_vals_q = self.norm(out)

        # Inject filter identity into the per-observation representation
        # before the final projection, so the output can differ by band
        filter_proj = self.filter_inject(filter_onehot)                # (B, T, d_model)
        combined    = ref_vals_q + filter_proj                         # (B, T, d_model)

        return self.out_proj(combined)                                 # (B, T, 2)


# ── Full Autoencoder ──────────────────────────────────────────────────────────

class mTANAutoencoder(nn.Module):

    def __init__(
        self,
        d_model:  int   = 64,
        d_time:   int   = 16,
        d_latent: int   = 64,
        n_ref:    int   = 64,
        n_heads:  int   = 4,
        n_layers: int   = 2,
        dropout:  float = 0.1,
    ):
        super().__init__()
        shared = dict(d_model=d_model, d_time=d_time, d_latent=d_latent,
                      n_ref=n_ref, n_heads=n_heads, n_layers=n_layers, dropout=dropout)
        self.encoder = mTANEncoder(d_input=5, **shared)
        self.decoder = mTANDecoder(**shared)

    def forward(
        self,
        x:    torch.Tensor,   # (B, T, 5)
        mask: torch.Tensor,   # (B, T)  bool
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            recon: (B, T, 2)   reconstructed [magpsf, sigmapsf]
            z:     (B, d_latent)
        """
        z            = self.encoder(x, mask)
        times        = x[:, :, 0]                    # (B, T)
        filter_onehot = x[:, :, 3:5]                # (B, T, 2)  — is_g, is_r
        recon        = self.decoder(z, times, filter_onehot)
        return recon, z

    def encode(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Convenience method — encode only, returns z."""
        return self.encoder(x, mask)


# ── Loss ──────────────────────────────────────────────────────────────────────

def masked_chi2_loss(
    recon:       torch.Tensor,   # (B, T, 2)  predicted [mag, sig]
    target:      torch.Tensor,   # (B, T, 2)  true      [mag, sig]  (normalised)
    mask:        torch.Tensor,   # (B, T)     bool
    alpha:       float = 0.5,    # weight of plain MSE term (0=pure chi2, 1=pure MSE)
    beta:        float = 0.1,    # weight of derivative loss term
    eps:         float = 1e-4,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Mixed loss: alpha*MSE + (1-alpha)*chi2 + beta*derivative_MSE.

    The derivative term penalises differences in local slope between adjacent
    observations, directly incentivising the model to learn variability patterns
    rather than collapsing to a smooth mean trend.

    Returns:
        loss:              scalar — mean over all valid observations
        per_object_loss:   (B,)   — mean loss per object (for anomaly scoring)
    """
    mag_pred = recon[..., 0]                                   # (B, T)
    mag_true = target[..., 0]
    # sig_true is now a genuine positive uncertainty (normalised by mag_std
    # in the dataset), so we only need a floor to avoid div-by-zero — not the
    # old abs() hack that masked the broken z-scored sigma.
    sig_true = target[..., 1].clamp(min=eps)

    mse_res  = (mag_pred - mag_true) ** 2                      # (B, T)
    chi2_res = mse_res / (sig_true ** 2)

    point_loss = (alpha * mse_res + (1.0 - alpha) * chi2_res) * mask

    # Derivative loss: compare consecutive differences
    # Only valid where both adjacent observations are real
    adj_mask   = mask[:, :-1] & mask[:, 1:]                   # (B, T-1)
    d_pred     = (mag_pred[:, 1:] - mag_pred[:, :-1]) * adj_mask
    d_true     = (mag_true[:, 1:] - mag_true[:, :-1]) * adj_mask
    deriv_loss = (d_pred - d_true) ** 2 * adj_mask            # (B, T-1)

    n_valid         = mask.sum(dim=1).clamp(min=1)             # (B,)
    n_adj           = adj_mask.sum(dim=1).clamp(min=1)         # (B,)

    per_object_loss = (point_loss.sum(dim=1) / n_valid
                       + beta * deriv_loss.sum(dim=1) / n_adj)
    loss            = per_object_loss.mean()

    return loss, per_object_loss
