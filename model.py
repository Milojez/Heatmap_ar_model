"""
Heatmap Autoregressive Model.

Architecture:
  1. Conditioning encoder
     - 60 tokens: 6 dials × 10 frames
     - Each token = space_proj(geom[dial]) + signal_proj(signal[dial, frame]) + time_emb[frame]
     - Transformer encoder (NUM_ENCODER_BLOCKS layers, bidirectional)

  2. GRU decoder (teacher-forced during training, autoregressive at inference)
     - Input at step i: fixation_emb(fix_{i-1}) concatenated with fixation-index embedding
     - GRU(hidden=HIDDEN_DIM, num_layers=NUM_GRU_LAYERS)

  3. Cross-attention: GRU output attends to encoder memory

  4. Heatmap head → logits [HM_H * HM_W] → softmax → 2D probability map
     Temporal head  → [dt_norm, T_norm]
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import config
from utils.heatmap import sample_from_heatmap


class HeatmapARModel(nn.Module):
    def __init__(self):
        super().__init__()
        H         = config.HIDDEN_DIM
        nf        = config.NUM_SIGNAL_FRAMES
        nd        = config.NUM_DIALS
        sig_dim   = config.SIGNAL_DIM
        n_heads   = config.NUM_HEADS
        hm_cells  = config.HM_W * config.HM_H

        # ── Conditioning encoder ──────────────────────────────────────────────
        self.space_proj  = nn.Linear(4, H)          # dial geom [cx, cy, tx, ty]
        self.signal_proj = nn.Linear(sig_dim, H)    # per-frame signal
        self.time_emb    = nn.Embedding(nf, H)      # frame index 0..nf-1

        enc_layer = nn.TransformerEncoderLayer(
            d_model=H, nhead=n_heads,
            dim_feedforward=H * 4,
            dropout=config.ATTN_DROPOUT,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer,
                                             num_layers=config.NUM_ENCODER_BLOCKS)

        # ── Decoder inputs ────────────────────────────────────────────────────
        # Embed previous fixation (x_norm, y_norm) → H
        self.fix_proj = nn.Linear(2, H)
        # Fixation step index embedding (0 = first output, predict fix_0)
        self.step_emb = nn.Embedding(config.MAX_FIXATIONS, H)
        # Learnable START fixation position (x_norm, y_norm)
        self.start_fix = nn.Parameter(torch.zeros(1, 2))

        # GRU: input = fix_proj + step_emb (both H, summed)
        self.gru = nn.GRU(
            input_size=H,
            hidden_size=H,
            num_layers=config.NUM_GRU_LAYERS,
            batch_first=True,
            dropout=config.ATTN_DROPOUT if config.NUM_GRU_LAYERS > 1 else 0.0,
        )

        # ── Cross-attention: GRU output → encoder memory ──────────────────────
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=H, num_heads=n_heads,
            dropout=config.ATTN_DROPOUT,
            batch_first=True,
        )
        self.cross_norm = nn.LayerNorm(H)

        # ── Output heads ──────────────────────────────────────────────────────
        self.hm_head = nn.Sequential(
            nn.LayerNorm(H),
            nn.Linear(H, H),
            nn.GELU(),
            nn.Linear(H, hm_cells),          # logits over 96×50 cells
        )
        self.temporal_head = nn.Sequential(
            nn.LayerNorm(H),
            nn.Linear(H, H // 2),
            nn.GELU(),
            nn.Linear(H // 2, 2),            # [dt_norm, T_norm]
        )

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _encode(self, cond_geom, cond_signal):
        """
        cond_geom   : [B, 6, 4]
        cond_signal : [B, 6, nf, SIGNAL_DIM]
        Returns memory: [B, 60, H]
        """
        B  = cond_geom.shape[0]
        nf = config.NUM_SIGNAL_FRAMES
        nd = config.NUM_DIALS

        # Expand geom to match all frames: [B, 6, nf, H]
        geom_tok = self.space_proj(cond_geom).unsqueeze(2).expand(-1, -1, nf, -1)

        # Signal per frame: [B, 6, nf, H]
        sig_tok = self.signal_proj(cond_signal)

        # Time embedding: [nf, H] → [1, 1, nf, H]
        t_idx   = torch.arange(nf, device=cond_geom.device)
        t_tok   = self.time_emb(t_idx).unsqueeze(0).unsqueeze(0)

        # Combine: [B, 6, nf, H]
        tokens = geom_tok + sig_tok + t_tok

        # Flatten dial × frame: [B, 60, H]
        tokens = tokens.reshape(B, nd * nf, -1)

        return self.encoder(tokens)   # [B, 60, H]

    def _decode_step(self, fix_xy, step_idx, memory, h_state):
        """
        Single GRU step.
        fix_xy   : [B, 2]   normalised (x, y) of the *previous* fixation
        step_idx : int       current fixation index (0 = predicting fix_0)
        memory   : [B, 60, H]
        h_state  : GRU hidden state

        Returns:
          hm_logits : [B, HM_H*HM_W]
          temporal  : [B, 2]
          h_state   : updated hidden
        """
        B = fix_xy.shape[0]
        device = fix_xy.device

        step_t = torch.full((B,), step_idx, dtype=torch.long, device=device)
        gru_in = self.fix_proj(fix_xy) + self.step_emb(step_t)   # [B, H]

        gru_out, h_new = self.gru(gru_in.unsqueeze(1), h_state)   # [B, 1, H]

        # Cross-attention
        ctx, _ = self.cross_attn(gru_out, memory, memory)         # [B, 1, H]
        ctx = self.cross_norm(gru_out + ctx)                       # residual

        ctx_sq = ctx.squeeze(1)                                    # [B, H]
        return self.hm_head(ctx_sq), self.temporal_head(ctx_sq), h_new

    # ── Forward (teacher-forced) ──────────────────────────────────────────────

    def forward(self, seq, cond_geom, cond_signal):
        """
        seq         : [B, N, 4]  target sequence (teacher-forced input)
        cond_geom   : [B, 6, 4]
        cond_signal : [B, 6, nf, SIGNAL_DIM]

        Returns:
          hm_logits : [B, N, HM_H*HM_W]
          temporal  : [B, N, 2]      predicted [dt_norm, T_norm]
        """
        B, N, _ = seq.shape
        device  = seq.device
        memory  = self._encode(cond_geom, cond_signal)   # [B, 60, H]

        # Teacher-forced inputs: [START, fix_0, fix_1, ..., fix_{N-2}]
        start = self.start_fix.expand(B, 1, 2)           # [B, 1, 2]
        prev_xy = torch.cat([start, seq[:, :-1, :2]], dim=1)   # [B, N, 2]

        # Run all GRU steps in batch (teacher forcing)
        H = config.HIDDEN_DIM
        step_indices = torch.arange(N, device=device)              # [N]
        step_embs    = self.step_emb(step_indices)                 # [N, H]
        fix_embs     = self.fix_proj(prev_xy)                      # [B, N, H]

        gru_in = fix_embs + step_embs.unsqueeze(0)                 # [B, N, H]

        k = config.GRU_HISTORY_STEPS
        if k == 0:
            # Full accumulated history (original behaviour)
            gru_out, _ = self.gru(gru_in)                         # [B, N, H]
        else:
            # Reset h_state every k steps so the GRU sees at most k prev fixations
            gru_outs = []
            h = None
            for t in range(N):
                if t % k == 0:
                    h = None
                out, h = self.gru(gru_in[:, t:t+1, :], h)        # [B, 1, H]
                gru_outs.append(out)
            gru_out = torch.cat(gru_outs, dim=1)                  # [B, N, H]

        # Cross-attention (all steps at once)
        ctx, _ = self.cross_attn(gru_out, memory, memory)         # [B, N, H]
        ctx = self.cross_norm(gru_out + ctx)                       # [B, N, H]

        hm_logits = self.hm_head(ctx)                              # [B, N, HM_H*HM_W]
        temporal  = self.temporal_head(ctx)                        # [B, N, 2]

        return hm_logits, temporal

    # ── Autoregressive generation ─────────────────────────────────────────────

    @torch.no_grad()
    def generate(self, cond_geom, cond_signal, num_fixations,
                 temperature=1.0):
        """
        Autoregressively generate fixations by sampling from heatmaps.

        Returns:
          xy_norm : [B, num_fixations, 4]
                    columns: [x_norm, y_norm, dt_norm, T_norm]
        """
        B      = cond_geom.shape[0]
        device = cond_geom.device
        memory = self._encode(cond_geom, cond_signal)   # [B, 60, H]

        prev_xy = self.start_fix.expand(B, 2)           # [B, 2]
        h_state = None                                   # GRU initialised to zeros

        results = []

        k = config.GRU_HISTORY_STEPS
        for step in range(num_fixations):
            if k > 0 and step % k == 0:
                h_state = None   # reset every k steps
            hm_logits, temporal, h_state = self._decode_step(
                prev_xy, step, memory, h_state
            )

            # Apply temperature and convert to probabilities
            if temperature != 1.0:
                hm_logits = hm_logits / temperature
            probs = F.softmax(hm_logits, dim=-1)         # [B, HM_H*HM_W]

            # Sample from heatmap → normalised coords
            xy_norm, _ = sample_from_heatmap(
                probs,
                hm_w=config.HM_W, hm_h=config.HM_H,
                img_w=config.IMAGE_WIDTH, img_h=config.IMAGE_HEIGHT,
            )   # [B, 2]

            # Concatenate with temporal predictions
            step_out = torch.cat([xy_norm, temporal], dim=-1)   # [B, 4]
            results.append(step_out)

            prev_xy = xy_norm   # feed sampled position back

        return torch.stack(results, dim=1)   # [B, num_fixations, 4]
