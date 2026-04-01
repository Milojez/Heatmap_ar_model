"""
Plotting helpers for the Heatmap AR model.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
import torch
import torch.nn.functional as F

import config


def overlay_heatmap_on_image(ax, probs_2d, alpha=0.5, cmap='hot'):
    """
    Overlay a [HM_H, HM_W] probability map on an axes as a semi-transparent image.
    probs_2d : numpy array [HM_H, HM_W]
    """
    hm = np.clip(probs_2d, 0, None)
    if hm.max() > 0:
        hm = hm / hm.max()
    # Stretch to image size
    hm_img = np.kron(hm, np.ones((
        config.IMAGE_HEIGHT // config.HM_H,
        config.IMAGE_WIDTH  // config.HM_W,
    )))
    ax.imshow(hm_img, cmap=cmap, alpha=alpha,
              extent=[0, config.IMAGE_WIDTH, config.IMAGE_HEIGHT, 0],
              vmin=0, vmax=1)


def draw_scanpath(ax, xy_px, length, color='cyan', radius_scale=0.25, radius_min=5.0):
    """
    Draw fixation circles + saccade lines on ax.
    xy_px : [N, 2] pixel coords
    """
    cmap  = cm.get_cmap('cool', length)
    for i in range(length):
        x, y = xy_px[i]
        r = max(radius_min, radius_scale * 20)
        circle = plt.Circle((x, y), r, color=cmap(i / max(length - 1, 1)),
                             fill=False, linewidth=1.5)
        ax.add_patch(circle)
        ax.text(x, y, str(i + 1), fontsize=6, ha='center', va='center',
                color=cmap(i / max(length - 1, 1)))
        if i > 0:
            x0, y0 = xy_px[i - 1]
            ax.plot([x0, x], [y0, y], '-', color=cmap(i / max(length - 1, 1)),
                    linewidth=0.8, alpha=0.7)


def save_comparison_figure(pred_px, gt_px, pred_hm_logits,
                            pred_length, gt_length,
                            sample_idx, out_path):
    """
    Side-by-side: left = GT scanpath, right = pred scanpath + heatmap overlay.

    pred_px         : [N, 2] pixel coords
    gt_px           : [N, 2] pixel coords
    pred_hm_logits  : [N, HM_H*HM_W] or None — last-step heatmap shown
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    W, H = config.IMAGE_WIDTH, config.IMAGE_HEIGHT

    for ax in axes:
        ax.set_xlim(0, W)
        ax.set_ylim(H, 0)
        ax.set_aspect('equal')
        ax.set_facecolor('#1a1a2e')

    # Left: GT
    axes[0].set_title(f'Ground truth  (N={gt_length})', color='white')
    axes[0].tick_params(colors='white')
    axes[0].set_facecolor('#1a1a2e')
    draw_scanpath(axes[0], gt_px[:gt_length], gt_length)

    # Right: Prediction + last heatmap
    axes[1].set_title(f'Prediction  (N={pred_length})', color='white')
    axes[1].tick_params(colors='white')
    axes[1].set_facecolor('#1a1a2e')
    if pred_hm_logits is not None:
        # Show heatmap for the last valid fixation step
        last_step = min(pred_length - 1, pred_hm_logits.shape[0] - 1)
        probs = F.softmax(pred_hm_logits[last_step], dim=-1)
        probs_2d = probs.reshape(config.HM_H, config.HM_W).cpu().numpy()
        overlay_heatmap_on_image(axes[1], probs_2d, alpha=0.45)
    draw_scanpath(axes[1], pred_px[:pred_length], pred_length, color='lime')

    fig.patch.set_facecolor('#0d0d1a')
    plt.tight_layout()
    plt.savefig(out_path, dpi=100, bbox_inches='tight')
    plt.close(fig)
