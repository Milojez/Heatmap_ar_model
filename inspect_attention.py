"""
inspect_attention.py
====================
Captures and visualises the cross-attention weights from the GRU decoder
attending to the 60 encoder tokens (6 dials × 10 frames).

For a chosen test sample this shows, for each fixation step:
  - Which dials the model attends to most
  - Which temporal frames within each dial it focuses on
  - How attention shifts over the scanpath

The 60-token attention map is reshaped to [N_fixations, 6_dials, 10_frames]
and displayed as a per-step heatmap grid.

Parameters (edit below):
"""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.cm as cm
import torch.nn.functional as F

# ── Parameters ────────────────────────────────────────────────────────────────
SAMPLE_INDEX = 0
TEMPERATURE  = 1.0
SAVE_OUTPUT  = True
SHOW         = False
# ─────────────────────────────────────────────────────────────────────────────

sys.path.insert(0, os.path.dirname(__file__))
import config
from dataset import ScanpathDataset
from model import HeatmapARModel

W, H = config.IMAGE_WIDTH, config.IMAGE_HEIGHT


def load_model(device):
    model = HeatmapARModel().to(device)
    ckpt_path = config.BEST_CHECKPOINT_PATH
    if not os.path.exists(ckpt_path):
        ckpt_path = config.CHECKPOINT_PATH
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"No checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    print(f"Loaded: {ckpt_path}  epoch={ckpt.get('epoch','?')}")
    return model


def generate_with_attention(model, cond_geom, cond_signal, num_fixations,
                             temperature=1.0):
    """
    Autoregressive generation that also returns cross-attention weights.
    Returns:
      pred_seq   : [num_fixations, 4]
      attn_maps  : [num_fixations, 60]  one row per fixation step
    """
    B      = 1
    device = cond_geom.device

    # Encode conditioning
    memory = model._encode(cond_geom, cond_signal)   # [1, 60, H]

    prev_xy = model.start_fix.expand(B, 2)
    h_state = None

    pred_steps = []
    attn_steps = []

    for step in range(num_fixations):
        # GRU step
        step_t  = torch.full((B,), step, dtype=torch.long, device=device)
        gru_in  = model.fix_proj(prev_xy) + model.step_emb(step_t)
        gru_out, h_state = model.gru(gru_in.unsqueeze(1), h_state)

        # Cross-attention with weights
        ctx, attn_w = model.cross_attn(
            gru_out, memory, memory,
            need_weights=True, average_attn_weights=True,
        )   # attn_w: [1, 1, 60]
        ctx = model.cross_norm(gru_out + ctx)

        ctx_sq    = ctx.squeeze(1)
        hm_logits = model.hm_head(ctx_sq)
        temporal  = model.temporal_head(ctx_sq)

        # Sample position
        if temperature != 1.0:
            hm_logits = hm_logits / temperature
        probs = F.softmax(hm_logits, dim=-1)

        idx = torch.multinomial(probs, num_samples=1).squeeze(-1)
        row = idx // config.HM_W
        col = idx %  config.HM_W
        cell_w = W / config.HM_W
        cell_h = H / config.HM_H
        x_px = (col.float() + torch.rand(1, device=device)) * cell_w
        y_px = (row.float() + torch.rand(1, device=device)) * cell_h
        x_norm = x_px / W * 2 - 1
        y_norm = y_px / H * 2 - 1

        xy_norm = torch.stack([x_norm, y_norm], dim=-1)
        step_out = torch.cat([xy_norm, temporal], dim=-1)
        pred_steps.append(step_out.squeeze(0))
        attn_steps.append(attn_w.squeeze(0).squeeze(0))   # [60]

        prev_xy = xy_norm

    pred_seq  = torch.stack(pred_steps, dim=0)    # [N, 4]
    attn_maps = torch.stack(attn_steps, dim=0)    # [N, 60]
    return pred_seq, attn_maps


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model  = load_model(device)

    ds = ScanpathDataset(
        config.TEST_JSON,
        width=W, height=H,
        max_fixations=config.MAX_FIXATIONS,
    )

    sample  = ds[SAMPLE_INDEX]
    name    = sample['name']
    length  = sample['length']
    print(f"Sample {SAMPLE_INDEX}: {name}  GT length={length}")

    cond_geom   = sample['cond_geom'].unsqueeze(0).to(device)
    cond_signal = sample['cond_signal'].unsqueeze(0).to(device)

    with torch.no_grad():
        pred_seq, attn_maps = generate_with_attention(
            model, cond_geom, cond_signal,
            num_fixations=length, temperature=TEMPERATURE,
        )

    # attn_maps: [N, 60] → reshape to [N, 6_dials, 10_frames]
    attn_3d = attn_maps.cpu().numpy().reshape(length,
                                               config.NUM_DIALS,
                                               config.NUM_SIGNAL_FRAMES)

    pred_x = (pred_seq[:, 0].cpu() + 1) / 2 * W
    pred_y = (pred_seq[:, 1].cpu() + 1) / 2 * H
    pred_px = np.stack([pred_x.numpy(), pred_y.numpy()], axis=1)

    gt_seq = sample['seq']
    gt_x   = (gt_seq[:length, 0] + 1) / 2 * W
    gt_y   = (gt_seq[:length, 1] + 1) / 2 * H
    gt_px  = np.stack([gt_x.numpy(), gt_y.numpy()], axis=1)

    # ── Figure 1: Attention per fixation step ─────────────────────────────────
    # Grid: rows = fixation steps, cols = dials;  cells show per-frame attention
    nrows = length
    ncols = config.NUM_DIALS

    fig1, axes = plt.subplots(nrows, ncols,
                               figsize=(ncols * 1.8, nrows * 1.5 + 0.5))
    fig1.suptitle(
        f"Cross-attention weights  —  Sample {SAMPLE_INDEX}: {name}\n"
        f"Row = fixation step, Col = dial, colour = attention to frame",
        color='white', fontsize=10,
    )
    fig1.patch.set_facecolor('#0d0d1a')

    vmax = attn_3d.max()

    for fi in range(nrows):
        for di in range(ncols):
            ax = axes[fi, di] if nrows > 1 else axes[di]
            vals = attn_3d[fi, di]   # [10_frames]
            ax.bar(range(config.NUM_SIGNAL_FRAMES), vals,
                   color=cm.plasma(vals / (vmax + 1e-8)))
            ax.set_ylim(0, vmax * 1.1)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_facecolor('#111')
            if fi == 0:
                ax.set_title(f'D{di+1}', color='white', fontsize=8)
            if di == 0:
                ax.set_ylabel(f'Fix {fi+1}', color='gray', fontsize=7,
                              rotation=0, labelpad=28, va='center')

    os.makedirs(config.VIS_DIR, exist_ok=True)
    if SAVE_OUTPUT:
        p = os.path.join(config.VIS_DIR,
                         f"attention_{SAMPLE_INDEX:04d}_per_step.png")
        fig1.tight_layout()
        fig1.savefig(p, dpi=100, bbox_inches='tight')
        print(f"Saved: {p}")

    # ── Figure 2: Aggregated dial attention over all fixation steps ───────────
    # Sum attention over frames → [N, 6]; then show as heatmap over fixation steps
    dial_attn = attn_3d.sum(axis=2)   # [N, 6]  sum over frames

    fig2, axes2 = plt.subplots(1, 3, figsize=(18, max(4, length * 0.4 + 1)))
    fig2.suptitle(
        f"Dial attention summary  —  Sample {SAMPLE_INDEX}: {name}",
        color='white', fontsize=10,
    )
    fig2.patch.set_facecolor('#0d0d1a')

    # Heatmap: P(attend to dial d | fixation step i)
    ax = axes2[0]
    im = ax.imshow(dial_attn, aspect='auto', cmap='YlOrRd',
                   vmin=0, interpolation='nearest')
    ax.set_xticks(range(6))
    ax.set_xticklabels([f'D{i+1}' for i in range(6)], color='gray', fontsize=9)
    ax.set_yticks(range(length))
    ax.set_yticklabels([f'Fix {i+1}' for i in range(length)],
                        color='gray', fontsize=8)
    ax.set_title('Attention weight per dial\n(summed over frames)',
                 color='white', fontsize=9)
    ax.set_facecolor('#111')
    plt.colorbar(im, ax=ax)

    # Mean attention per dial (across all fixation steps)
    ax2 = axes2[1]
    mean_dial = dial_attn.mean(axis=0)
    ax2.set_facecolor('#1a1a2e')
    ax2.bar([f'D{i+1}' for i in range(6)], mean_dial,
             color=plt.cm.tab10(np.linspace(0, 1, 6)))
    ax2.set_title('Mean dial attention\n(over fixation steps)',
                  color='white', fontsize=9)
    ax2.tick_params(colors='gray')
    for spine in ax2.spines.values():
        spine.set_edgecolor('#333')

    # Scanpath overlay coloured by top attended dial
    ax3 = axes2[2]
    ax3.set_facecolor('#1a1a2e')
    ax3.set_xlim(0, W); ax3.set_ylim(H, 0)
    ax3.set_aspect('equal')
    ax3.set_title('Predicted scanpath\n(colour = top-attended dial)',
                  color='white', fontsize=9)
    ax3.tick_params(colors='gray')

    dial_colors = plt.cm.tab10(np.linspace(0, 1, 6))
    for fi in range(length):
        top_dial = int(np.argmax(dial_attn[fi]))
        c = dial_colors[top_dial]
        ax3.scatter(pred_px[fi, 0], pred_px[fi, 1], color=c, s=60, zorder=5)
        ax3.text(pred_px[fi, 0], pred_px[fi, 1], str(fi + 1),
                 fontsize=7, ha='center', va='center', color=c)
        if fi > 0:
            ax3.plot([pred_px[fi-1, 0], pred_px[fi, 0]],
                     [pred_px[fi-1, 1], pred_px[fi, 1]],
                     '-', color=c, lw=0.8, alpha=0.6)
    # GT in grey for reference
    for fi in range(length):
        ax3.scatter(gt_px[fi, 0], gt_px[fi, 1], color='white',
                    s=20, marker='x', zorder=4, linewidths=0.8)

    legend_patches = [mpatches.Patch(color=dial_colors[i], label=f'Dial {i+1}')
                      for i in range(6)]
    legend_patches.append(mpatches.Patch(color='white', label='GT (×)'))
    ax3.legend(handles=legend_patches, fontsize=7, loc='upper right',
               facecolor='#111', labelcolor='white', framealpha=0.7)

    if SAVE_OUTPUT:
        p = os.path.join(config.VIS_DIR,
                         f"attention_{SAMPLE_INDEX:04d}_summary.png")
        fig2.tight_layout()
        fig2.savefig(p, dpi=110, bbox_inches='tight')
        print(f"Saved: {p}")

    if SHOW:
        plt.show()
    plt.close('all')


if __name__ == '__main__':
    main()
