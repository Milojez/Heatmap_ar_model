"""
Visualize one test sample: GT scanpath vs predicted scanpath + heatmap overlays.

Parameters (edit below):
  SAMPLE_INDEX  — index into the test dataset
  N_RUNS        — number of stochastic generation runs to show (each in a column)
  TEMPERATURE   — sampling temperature (1.0 = raw probs)
  SHOW_ALL_HM   — if True, show per-step heatmaps; if False, show only the last one
"""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import torch.nn.functional as F

# ── Parameters ────────────────────────────────────────────────────────────────
SAMPLE_INDEX = 2
N_RUNS       = 3
TEMPERATURE  = 1.0
SHOW_ALL_HM  = False     # per-step heatmaps vs. just the last one
# ─────────────────────────────────────────────────────────────────────────────

sys.path.insert(0, os.path.dirname(__file__))
import config
from dataset import ScanpathDataset
from model import HeatmapARModel
from utils.heatmap import make_batch_heatmaps
from utils.plots import overlay_heatmap_on_image, draw_scanpath


def load_model(device):
    model = HeatmapARModel().to(device)
    ckpt_path = config.BEST_CHECKPOINT_PATH
    if not os.path.exists(ckpt_path):
        ckpt_path = config.CHECKPOINT_PATH
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"No checkpoint found at {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    print(f"Loaded checkpoint: {ckpt_path}  (epoch {ckpt.get('epoch', '?')})")
    return model


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model  = load_model(device)

    ds = ScanpathDataset(
        config.TEST_JSON,
        width=config.IMAGE_WIDTH, height=config.IMAGE_HEIGHT,
        max_fixations=config.MAX_FIXATIONS,
    )
    sample = ds[SAMPLE_INDEX]
    name   = sample['name']
    length = sample['length']
    print(f"Sample {SAMPLE_INDEX}: {name}  (GT length={length})")

    cond_geom   = sample['cond_geom'].unsqueeze(0).to(device)    # [1, 6, 4]
    cond_signal = sample['cond_signal'].unsqueeze(0).to(device)  # [1, 6, nf, D]
    gt_seq      = sample['seq'].to(device)                       # [N, 4]

    # GT pixel coords
    W, H = config.IMAGE_WIDTH, config.IMAGE_HEIGHT
    gt_x  = (gt_seq[:length, 0] + 1) / 2 * W
    gt_y  = (gt_seq[:length, 1] + 1) / 2 * H
    gt_px = torch.stack([gt_x, gt_y], dim=-1).cpu().numpy()

    # ── Generate N_RUNS stochastic predictions ────────────────────────────────
    all_pred_px = []
    all_logits  = []    # [N_RUNS, N, HM_H*HM_W] — teacher-forced logits for GT display

    with torch.no_grad():
        # Teacher-forced heatmaps (deterministic, based on GT input)
        hm_logits_tf, _ = model(gt_seq.unsqueeze(0), cond_geom, cond_signal)
        # hm_logits_tf: [1, N, cells]

        for run in range(N_RUNS):
            pred_seq = model.generate(
                cond_geom, cond_signal,
                num_fixations=length,
                temperature=TEMPERATURE,
            )   # [1, N, 4]
            pred_seq = pred_seq.squeeze(0)   # [N, 4]
            px = (pred_seq[:, 0] + 1) / 2 * W
            py = (pred_seq[:, 1] + 1) / 2 * H
            all_pred_px.append(torch.stack([px, py], dim=-1).cpu().numpy())

    # ── Figure layout ─────────────────────────────────────────────────────────
    # Columns: GT | Run1 | Run2 | ...
    ncols = 1 + N_RUNS
    if SHOW_ALL_HM:
        nrows = length
        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows))
        if nrows == 1:
            axes = np.array([axes])
        axes_list = list(axes.flat)
    else:
        nrows = 1
        fig, axes = plt.subplots(1, ncols, figsize=(6 * ncols, 6))
        axes_list = list(np.atleast_1d(axes))

    fig.suptitle(f"Sample {SAMPLE_INDEX}: {name}  (GT N={length})", fontsize=11)

    for ax in axes_list:
        ax.set_xlim(0, W); ax.set_ylim(H, 0)
        ax.set_aspect('equal')

    def _col(col_idx, row_idx=0):
        if SHOW_ALL_HM:
            return axes_list[row_idx * ncols + col_idx]
        return axes_list[col_idx]

    if SHOW_ALL_HM:
        # Per-step heatmap + GT dot in each row
        for s in range(length):
            ax = _col(0, s)
            probs = F.softmax(hm_logits_tf[0, s], dim=-1)
            probs_2d = probs.reshape(config.HM_H, config.HM_W).cpu().numpy()
            overlay_heatmap_on_image(ax, probs_2d, alpha=0.6)
            ax.scatter([gt_px[s, 0]], [gt_px[s, 1]], c='red', s=60, zorder=5)
            ax.set_title(f'GT hm step {s+1}', color='black', fontsize=8)

            for run, pred_px in enumerate(all_pred_px):
                ax2 = _col(1 + run, s)
                ax2.scatter([pred_px[s, 0]], [pred_px[s, 1]], c='cyan', s=60, zorder=5)
                ax2.set_title(f'Run {run+1} step {s+1}', color='black', fontsize=8)
    else:
        # Single row: GT full scanpath | each run's full scanpath + last heatmap
        ax_gt = _col(0)
        draw_scanpath(ax_gt, gt_px, length)
        ax_gt.set_title('Ground truth', color='black', fontsize=9)

        for run, pred_px in enumerate(all_pred_px):
            ax_r = _col(1 + run)
            # Show teacher-forced heatmap for last step as background
            last_s = length - 1
            probs = F.softmax(hm_logits_tf[0, last_s], dim=-1)
            probs_2d = probs.reshape(config.HM_H, config.HM_W).cpu().numpy()
            overlay_heatmap_on_image(ax_r, probs_2d, alpha=0.40)
            draw_scanpath(ax_r, pred_px, length, color='tab:blue')
            ax_r.set_title(f'Run {run+1}  T={TEMPERATURE}', color='black', fontsize=9)

    os.makedirs(config.VIS_DIR, exist_ok=True)
    out_path = os.path.join(config.VIS_DIR,
                            f"sample_{SAMPLE_INDEX:04d}_hm_vis.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=110, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == '__main__':
    main()
