"""
inspect_heatmaps.py
===================
Visualises the probability map the model produces at each fixation step
during autoregressive inference — i.e. the heatmap BEFORE the multinomial
sample that places the actual fixation.

Layout:
  One row per sample (N_SAMPLES rows).
  One column per fixation step (up to MAX_FIXATIONS columns).
  Each cell shows the 2D softmax probability map over the image plane,
  with the sampled fixation marked as a dot and the GT fixation as a cross.

Parameters (edit below):
"""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import torch.nn.functional as F

# ── Parameters ────────────────────────────────────────────────────────────────
SAMPLE_INDICES = [0, 1, 2]   # which test samples to show (one row each)
TEMPERATURE    = 1.0
SAVE_OUTPUT    = True
SHOW           = False
CMAP           = 'hot'       # colormap for heatmap cells
# ─────────────────────────────────────────────────────────────────────────────

sys.path.insert(0, os.path.dirname(__file__))
import config
from dataset import ScanpathDataset
from model import HeatmapARModel
from utils.heatmap import sample_from_heatmap


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


@torch.no_grad()
def generate_with_heatmaps(model, cond_geom, cond_signal, num_fixations,
                            temperature=1.0):
    """
    Run one autoregressive pass and collect, for each fixation step:
      - the full probability map [HM_H, HM_W]
      - the sampled pixel coordinate [2]

    Returns:
      prob_maps  : list of num_fixations arrays, each [HM_H, HM_W]
      sampled_px : list of num_fixations arrays, each [2]  (x_px, y_px)
    """
    B      = 1
    device = cond_geom.device

    memory  = model._encode(cond_geom, cond_signal)   # [1, 60, H]
    prev_xy = model.start_fix.expand(B, 2)
    h_state = None

    prob_maps  = []
    sampled_px = []

    k = config.GRU_HISTORY_STEPS
    for step in range(num_fixations):
        if k > 0 and step % k == 0:
            h_state = None   # reset every k steps
        hm_logits, temporal, h_state = model._decode_step(
            prev_xy, step, memory, h_state
        )   # hm_logits: [1, HM_H*HM_W]

        if temperature != 1.0:
            hm_logits = hm_logits / temperature
        probs = F.softmax(hm_logits, dim=-1)   # [1, HM_H*HM_W]

        # Reshape to 2D for visualisation
        prob_2d = probs[0].reshape(config.HM_H, config.HM_W).cpu().numpy()
        prob_maps.append(prob_2d)

        # Sample the fixation position
        xy_norm, xy_px = sample_from_heatmap(
            probs,
            hm_w=config.HM_W, hm_h=config.HM_H,
            img_w=config.IMAGE_WIDTH, img_h=config.IMAGE_HEIGHT,
        )   # [1, 2]

        sampled_px.append(xy_px[0].cpu().numpy())   # [x_px, y_px]
        prev_xy = xy_norm   # feed back

    return prob_maps, sampled_px


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model  = load_model(device)

    ds = ScanpathDataset(
        config.TEST_JSON,
        width=config.IMAGE_WIDTH, height=config.IMAGE_HEIGHT,
        max_fixations=config.MAX_FIXATIONS,
    )

    W, H  = config.IMAGE_WIDTH, config.IMAGE_HEIGHT
    HM_W  = config.HM_W
    HM_H  = config.HM_H

    n_samples = len(SAMPLE_INDICES)
    n_cols    = config.MAX_FIXATIONS   # one column per fixation step

    # Cell size in pixels for the final figure
    cell_w_in = 1.8   # inches per heatmap cell
    cell_h_in = 1.0   # inches per heatmap cell

    fig, axes = plt.subplots(
        n_samples, n_cols,
        figsize=(n_cols * cell_w_in, n_samples * cell_h_in + 0.6),
        squeeze=False,
    )
    fig.suptitle(
        f"Per-fixation probability maps during autoregressive inference\n"
        f"(T={TEMPERATURE})  —  dot=sampled  ×=GT",
        fontsize=10,
    )

    for row_idx, sample_idx in enumerate(SAMPLE_INDICES):
        sample = ds[sample_idx]
        name   = sample['name']
        length = sample['length']

        cond_geom   = sample['cond_geom'].unsqueeze(0).to(device)
        cond_signal = sample['cond_signal'].unsqueeze(0).to(device)
        gt_seq      = sample['seq']   # [N, 4]

        # GT pixel coords
        gt_x  = (gt_seq[:length, 0] + 1) / 2 * W
        gt_y  = (gt_seq[:length, 1] + 1) / 2 * H

        prob_maps, sampled_px = generate_with_heatmaps(
            model, cond_geom, cond_signal,
            num_fixations=length, temperature=TEMPERATURE,
        )

        # Scale factor: heatmap → image coords for scatter overlay
        scale_x = W / HM_W
        scale_y = H / HM_H

        for col_idx in range(n_cols):
            ax = axes[row_idx, col_idx]
            ax.axis('off')

            if col_idx < length:
                hm = prob_maps[col_idx]   # [HM_H, HM_W]

                # Show heatmap stretched to image aspect ratio
                ax.imshow(hm, origin='upper', cmap=CMAP,
                          aspect='auto', interpolation='nearest')

                # Sampled fixation (in heatmap cell coords)
                sx_cell = sampled_px[col_idx][0] / scale_x
                sy_cell = sampled_px[col_idx][1] / scale_y
                ax.scatter([sx_cell], [sy_cell], c='cyan', s=18,
                           zorder=5, linewidths=0)

                # GT fixation (in heatmap cell coords)
                if col_idx < length:
                    gx_cell = gt_x[col_idx].item() / scale_x
                    gy_cell = gt_y[col_idx].item() / scale_y
                    ax.scatter([gx_cell], [gy_cell], c='lime',
                               s=18, marker='x', zorder=6, linewidths=1.0)

                # Step label on the top row only; sample label on the left column only
                if row_idx == 0:
                    ax.set_title(f"Fix {col_idx + 1}", fontsize=7, pad=2)
            else:
                # Blank cell for steps beyond this sample's length
                ax.set_facecolor('#eee')

        # Sample label on the left
        axes[row_idx, 0].set_ylabel(
            f"S{sample_idx}\nN={length}", fontsize=7, rotation=0,
            labelpad=32, va='center',
        )
        axes[row_idx, 0].axis('on')
        axes[row_idx, 0].set_xticks([])
        axes[row_idx, 0].set_yticks([])

    plt.subplots_adjust(wspace=0.04, hspace=0.15, top=0.88, left=0.07)

    os.makedirs(config.VIS_DIR, exist_ok=True)
    tag = "_".join(str(i) for i in SAMPLE_INDICES)
    out_path = os.path.join(config.VIS_DIR, f"heatmaps_per_step_{tag}.png")

    if SAVE_OUTPUT:
        plt.savefig(out_path, dpi=130, bbox_inches='tight')
        print(f"Saved: {out_path}")

    if SHOW:
        plt.show()
    plt.close(fig)


if __name__ == '__main__':
    main()
