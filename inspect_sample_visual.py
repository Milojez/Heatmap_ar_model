"""
inspect_sample_visual.py
========================
Treats each model generation run as a "synthetic participant" and visualises
the resulting scanpath distribution, exactly like visualize_all_participants.py
does for real eye-tracking data.

For a chosen stimulus (SAMPLE_INDEX) the script:
  1. Runs model.generate() N_RUNS times  →  N_RUNS synthetic scanpaths
  2. Overlays all synthetic scanpaths on a blank canvas
  3. Shows a 2D fixation density heatmap (predicted vs GT)
  4. Shows per-dial statistics:
       - % of total fixation time per dial (bar chart)
       - P(dial | fixation index) heatmap

Parameters (edit below):
"""

import os
import re
import sys
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import torch.nn.functional as F

# ── Parameters ────────────────────────────────────────────────────────────────
SAMPLE_INDEX = 2          # index into the test dataset
N_RUNS       = 30         # synthetic participants (generation runs)
TEMPERATURE  = 1.0        # sampling temperature
SAVE_OUTPUT  = True       # save figures to VIS_DIR
SHOW         = False      # plt.show() at end
# ─────────────────────────────────────────────────────────────────────────────

sys.path.insert(0, os.path.dirname(__file__))
import config
from dataset import ScanpathDataset, denorm_duration
from model import HeatmapARModel

W, H = config.IMAGE_WIDTH, config.IMAGE_HEIGHT


# ── Helpers ───────────────────────────────────────────────────────────────────

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


def assign_to_dial(x_px, y_px, dial_centers):
    """Return index of nearest dial centre."""
    dists = [((x_px - cx) ** 2 + (y_px - cy) ** 2) ** 0.5
             for cx, cy in dial_centers]
    return int(np.argmin(dists))


def get_dial_centers(sample):
    """Extract dial centres (pixel) from a dataset sample, sorted by position."""
    dials = sorted(sample['dials_raw'], key=lambda d: d['dial_position'])
    return [(d['center_x_px'], d['center_y_px']) for d in dials]


def frame_key(name):
    first = name[0] if isinstance(name, list) else name
    m = re.match(r'.+?_frame_(\d+)', first)
    return int(m.group(1)) if m else 0


def dial_stats(all_xy, all_durations, dial_centers):
    """
    all_xy        : list of [N, 2] arrays (pixel coords per scanpath)
    all_durations : list of [N] arrays (duration ms per scanpath)
    Returns:
      time_pct  : [6]   % of total fixation time per dial
      count_pct : [6]   % of total fixation count per dial
    """
    dial_time  = np.zeros(6)
    dial_count = np.zeros(6)
    for xy, durs in zip(all_xy, all_durations):
        for i in range(len(xy)):
            d = assign_to_dial(xy[i, 0], xy[i, 1], dial_centers)
            dial_time[d]  += durs[i]
            dial_count[d] += 1
    time_total  = dial_time.sum()
    count_total = dial_count.sum()
    time_pct  = dial_time  / time_total  * 100 if time_total  > 0 else dial_time
    count_pct = dial_count / count_total * 100 if count_total > 0 else dial_count
    return time_pct, count_pct


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model  = load_model(device)

    ds = ScanpathDataset(
        config.TEST_JSON,
        width=W, height=H,
        max_fixations=config.MAX_FIXATIONS,
    )

    with open(config.TEST_JSON, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)

    sample     = ds[SAMPLE_INDEX]
    raw_sample = raw_data[SAMPLE_INDEX]
    name       = sample['name']
    gt_length  = sample['length']
    stim_key   = frame_key(raw_sample['name'])
    print(f"Sample {SAMPLE_INDEX}: {name}  GT length={gt_length}  stimulus={stim_key}")

    dial_centers = [(d['center_x_px'], d['center_y_px'])
                    for d in sorted(raw_sample['dials'],
                                    key=lambda d: d['dial_position'])]

    cond_geom   = sample['cond_geom'].unsqueeze(0).to(device)
    cond_signal = sample['cond_signal'].unsqueeze(0).to(device)
    gt_seq      = sample['seq']   # [N, 4]

    # GT pixel coords (selected sample)
    gt_x  = (gt_seq[:gt_length, 0] + 1) / 2 * W
    gt_y  = (gt_seq[:gt_length, 1] + 1) / 2 * H
    gt_px = np.stack([gt_x.numpy(), gt_y.numpy()], axis=1)   # [gt_length, 2]

    # ── Collect ALL participants for this stimulus ─────────────────────────────
    same_stim = [s for s in raw_data if frame_key(s['name']) == stim_key]
    print(f"Participants with same stimulus: {len(same_stim)}")

    all_part_xy  = []
    all_part_dur = []
    for s in same_stim:
        n = s['length']
        if n == 0:
            continue
        xy  = np.stack([s['X'][:n], s['Y'][:n]], axis=1).astype(float)
        dur = np.array(s['T'][:n], dtype=float)
        all_part_xy.append(xy)
        all_part_dur.append(dur)

    # GT durations for selected sample (for density/overlay figure)
    gt_T_norm = gt_seq[:gt_length, 3]
    gt_dur    = denorm_duration(gt_T_norm, ds.norm_stats).numpy()

    # ── Generate N_RUNS synthetic scanpaths ───────────────────────────────────
    all_pred_xy  = []
    all_pred_dur = []

    with torch.no_grad():
        for run in range(N_RUNS):
            pred = model.generate(
                cond_geom, cond_signal,
                num_fixations=gt_length,
                temperature=TEMPERATURE,
            ).squeeze(0)   # [N, 4]

            px = (pred[:, 0] + 1) / 2 * W
            py = (pred[:, 1] + 1) / 2 * H
            all_pred_xy.append(np.stack([px.cpu().numpy(),
                                         py.cpu().numpy()], axis=1))
            all_pred_dur.append(
                denorm_duration(pred[:, 3].cpu(), ds.norm_stats).numpy()
            )

    print(f"Generated {N_RUNS} synthetic scanpaths.")

    # ── Statistics ────────────────────────────────────────────────────────────
    part_time_pct, part_count_pct = dial_stats(all_part_xy, all_part_dur, dial_centers)
    pred_time_pct, pred_count_pct = dial_stats(all_pred_xy, all_pred_dur, dial_centers)

    # ── Figure 1: Overlay of all synthetic scanpaths ─────────────────────────
    fig1, axes1 = plt.subplots(1, 2, figsize=(18, 6))
    fig1.suptitle(f"Sample {SAMPLE_INDEX}: {name}\n"
                  f"GT N={gt_length} | Synthetic participants N_RUNS={N_RUNS}",
                  fontsize=11)
    for ax in axes1:
        ax.set_xlim(0, W); ax.set_ylim(H, 0)
        ax.set_aspect('equal')
        # Draw dial centres
        for cx, cy in dial_centers:
            ax.plot(cx, cy, '+', color='black', markersize=12, markeredgewidth=1.5)

    # Left: GT scanpath
    cmap_gt = cm.get_cmap('cool', gt_length)
    for i in range(gt_length):
        c = cmap_gt(i / max(gt_length - 1, 1))
        ax = axes1[0]
        ax.scatter(gt_px[i, 0], gt_px[i, 1], color=c, s=60, zorder=5)
        ax.text(gt_px[i, 0], gt_px[i, 1], str(i + 1), fontsize=7,
                ha='center', va='center', color=c)
        if i > 0:
            ax.plot([gt_px[i-1, 0], gt_px[i, 0]],
                    [gt_px[i-1, 1], gt_px[i, 1]], '-', color=c, lw=0.8, alpha=0.6)
    axes1[0].set_title('Ground truth')

    # Right: all synthetic scanpaths overlaid
    ax_pred = axes1[1]
    alpha_per_run = max(0.08, min(0.4, 1.0 / N_RUNS ** 0.5))
    for xy in all_pred_xy:
        n = len(xy)
        clr = cm.tab10(np.random.randint(0, 10) / 10)
        for i in range(n):
            ax_pred.scatter(xy[i, 0], xy[i, 1], color=clr,
                            s=15, alpha=alpha_per_run, zorder=3)
            if i > 0:
                ax_pred.plot([xy[i-1, 0], xy[i, 0]],
                             [xy[i-1, 1], xy[i, 1]], '-',
                             color=clr, lw=0.5, alpha=alpha_per_run * 0.7)
    axes1[1].set_title(f'{N_RUNS} synthetic participants  T={TEMPERATURE}')

    os.makedirs(config.VIS_DIR, exist_ok=True)
    if SAVE_OUTPUT:
        p = os.path.join(config.VIS_DIR,
                         f"inspect_{SAMPLE_INDEX:04d}_overlay.png")
        fig1.tight_layout()
        fig1.savefig(p, dpi=110, bbox_inches='tight')
        print(f"Saved: {p}")

    # ── Figure 2: Density heatmaps ────────────────────────────────────────────
    def make_density(xy_list, nbins_x=96, nbins_y=50):
        xs = np.concatenate([xy[:, 0] for xy in xy_list])
        ys = np.concatenate([xy[:, 1] for xy in xy_list])
        hm, _, _ = np.histogram2d(xs, ys,
                                   bins=[nbins_x, nbins_y],
                                   range=[[0, W], [0, H]])
        return hm.T   # [nbins_y, nbins_x]

    pred_density = make_density(all_pred_xy)
    gt_density   = make_density(all_part_xy)   # all participants of this stimulus

    fig2, axes2 = plt.subplots(1, 2, figsize=(16, 5))
    fig2.suptitle(f"Fixation Density  —  Sample {SAMPLE_INDEX}: {name}", fontsize=11)

    for ax, hm, title in zip(axes2,
                              [gt_density, pred_density],
                              [f'All participants (N={len(all_part_xy)})',
                               f'Model density ({N_RUNS} runs)']):
        ax.imshow(hm, origin='upper', cmap='hot',
                  extent=[0, W, H, 0], aspect='auto')
        for cx, cy in dial_centers:
            ax.plot(cx, cy, '+', color='white', markersize=10, markeredgewidth=1.5)
        ax.set_title(title)

    if SAVE_OUTPUT:
        p = os.path.join(config.VIS_DIR,
                         f"inspect_{SAMPLE_INDEX:04d}_density.png")
        fig2.tight_layout()
        fig2.savefig(p, dpi=110, bbox_inches='tight')
        print(f"Saved: {p}")

    # ── Figure 3: Dial statistics ─────────────────────────────────────────────
    dial_labels = [f"Dial {i+1}" for i in range(6)]
    colors      = plt.cm.tab10(np.linspace(0, 1, 6))
    x           = np.arange(6)
    w           = 0.35

    fig3, axes3 = plt.subplots(1, 2, figsize=(14, 5))
    fig3.suptitle(
        f"Dial Statistics  —  Sample {SAMPLE_INDEX}: {name}\n"
        f"All participants (N={len(all_part_xy)})  vs  Model ({N_RUNS} runs)",
        fontsize=11,
    )

    # Panel 1: % fixation TIME per dial — all participants
    axes3[0].bar(x, part_time_pct, color=colors)
    axes3[0].set_xticks(x)
    axes3[0].set_xticklabels(dial_labels, rotation=30, ha='right')
    axes3[0].set_ylabel('% fixation time')
    axes3[0].set_title(f'% Fixation time per dial\n(all {len(all_part_xy)} participants)')
    axes3[0].set_ylim(0, 100)
    axes3[0].grid(axis='y', alpha=0.3)

    # Panel 2: % fixation COUNT per dial — all participants vs model (grouped bars)
    bars_p = axes3[1].bar(x - w/2, part_count_pct, w,
                           label=f'All participants (N={len(all_part_xy)})',
                           color='steelblue', alpha=0.85)
    bars_m = axes3[1].bar(x + w/2, pred_count_pct, w,
                           label=f'Model ({N_RUNS} runs)',
                           color='tomato', alpha=0.85)
    axes3[1].set_xticks(x)
    axes3[1].set_xticklabels(dial_labels, rotation=30, ha='right')
    axes3[1].set_ylabel('% fixation count')
    axes3[1].set_title('% Fixation count per dial')
    axes3[1].set_ylim(0, 100)
    axes3[1].legend()
    axes3[1].grid(axis='y', alpha=0.3)

    if SAVE_OUTPUT:
        p = os.path.join(config.VIS_DIR,
                         f"inspect_{SAMPLE_INDEX:04d}_stats.png")
        fig3.tight_layout()
        fig3.savefig(p, dpi=110, bbox_inches='tight')
        print(f"Saved: {p}")

    if SHOW:
        plt.show()
    plt.close('all')


if __name__ == '__main__':
    main()
