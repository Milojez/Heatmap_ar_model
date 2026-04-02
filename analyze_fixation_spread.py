"""
analyze_fixation_spread.py
==========================
Produces two fixation density figures for the test set:

  1. test_heatmap_with_clusters.png
       Human fixation density + 6 dial centroids + 1/2-sigma rings

  2. human_vs_model_heatmap.png
       Side-by-side: human vs model fixation density,
       both with 6 dial centroids + 1/2-sigma rings
"""

import json
import re
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from scipy.ndimage import gaussian_filter
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.dirname(__file__))
import config

W         = config.IMAGE_WIDTH
H         = config.IMAGE_HEIGHT
CELL_SIZE = ((W / config.HM_W) + (H / config.HM_H)) / 2   # ~19.8 px
OUT_DIR   = os.path.join(config.VIS_DIR, "analysis")
os.makedirs(OUT_DIR, exist_ok=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

def frame_key(name):
    first = name[0] if isinstance(name, list) else name
    m = re.match(r'.+?_frame_(\d+)', first)
    return int(m.group(1)) if m else 0


def load_all():
    records = []
    for path, split in [(config.TRAIN_JSON, "train"),
                        (config.VAL_JSON,   "val"),
                        (config.TEST_JSON,  "test")]:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        for s in data:
            s["_split"] = split
        records.extend(data)
    print(f"Loaded {len(records)} samples (train+val+test).")
    return records


def within_cluster_spread(records):
    """
    Groups fixations by (stimulus x dial), computes per-group centroids,
    returns array of within-cluster distances in px.
    Used only to derive the sigma ring radii.
    """
    groups = defaultdict(list)
    for s in records:
        n = s["length"]
        if n == 0 or not s["dials"]:
            continue
        fk     = frame_key(s["name"])
        cx_arr = np.array([d["center_x_px"]  for d in s["dials"]], dtype=float)
        cy_arr = np.array([d["center_y_px"]  for d in s["dials"]], dtype=float)
        pos_arr= np.array([d["dial_position"] for d in s["dials"]], dtype=int)
        for x, y in zip(s["X"][:n], s["Y"][:n]):
            best = int(np.argmin((cx_arr - x)**2 + (cy_arr - y)**2))
            groups[(fk, int(pos_arr[best]))].append((x, y))

    wc = []
    for pts_list in groups.values():
        if len(pts_list) < 2:
            continue
        pts      = np.array(pts_list, dtype=float)
        centroid = pts.mean(axis=0)
        wc.extend(np.linalg.norm(pts - centroid, axis=1).tolist())
    return np.array(wc, dtype=float)


def assign_to_dials(xs, ys, dials_meta):
    """Assign (x,y) list to nearest dial; returns dict dial_pos->[points]."""
    cx_arr  = np.array([d["center_x_px"]  for d in dials_meta], dtype=float)
    cy_arr  = np.array([d["center_y_px"]  for d in dials_meta], dtype=float)
    pos_arr = np.array([d["dial_position"] for d in dials_meta], dtype=int)
    result  = defaultdict(list)
    for x, y in zip(xs, ys):
        best = int(np.argmin((cx_arr - x)**2 + (cy_arr - y)**2))
        result[int(pos_arr[best])].append((x, y))
    return result


def build_density(xs, ys, bins_x=192, bins_y=100, sigma=1.5):
    h, _, _ = np.histogram2d(xs, ys, bins=[bins_x, bins_y],
                             range=[[0, W], [0, H]])
    return gaussian_filter(h.T, sigma=sigma)


def dial_centroids(dial_fix):
    """One centroid per dial, with fixation count."""
    return [(np.array(pts, dtype=float)[:, 0].mean(),
             np.array(pts, dtype=float)[:, 1].mean(),
             d,
             len(pts))
            for d, pts in sorted(dial_fix.items())]


def overlay(ax, density, cents, r1, r2, title):
    dial_colors = plt.cm.tab10(np.linspace(0, 0.6, 6))
    ax.imshow(density, origin="upper", extent=[0, W, H, 0],
              cmap="hot", aspect="auto")
    total = sum(c[3] for c in cents)
    for cx, cy, d, n in cents:
        col = dial_colors[d - 1]
        ax.add_patch(plt.Circle((cx, cy), r2, fill=False, color=col,
                                linewidth=1.0, linestyle="--", alpha=0.7))
        ax.add_patch(plt.Circle((cx, cy), r1, fill=False, color=col,
                                linewidth=1.6, linestyle="-",  alpha=0.95))
        ax.plot(cx, cy, "o", color=col, markersize=7,
                markeredgecolor="white", markeredgewidth=1.0, zorder=6)
        ax.text(cx, cy - r1 - 10, f"D{d}\n{n/total*100:.0f}%",
                color=col, fontsize=7, ha="center", va="bottom",
                fontweight="bold", zorder=7)
    ax.set_xlim(0, W); ax.set_ylim(H, 0)
    ax.set_xlabel("x (px)"); ax.set_ylabel("y (px)")
    ax.set_title(title, fontsize=10)


# ── Figure 1: human test density ─────────────────────────────────────────────

def plot_human_density(test_records, r1, r2):
    dial_fix  = defaultdict(list)
    all_xs, all_ys = [], []
    for s in test_records:
        n = s["length"]
        if n == 0 or not s["dials"]:
            continue
        fix = assign_to_dials(s["X"][:n], s["Y"][:n], s["dials"])
        for d, pts in fix.items():
            dial_fix[d].extend(pts)
        all_xs.extend(s["X"][:n]); all_ys.extend(s["Y"][:n])

    density = build_density(all_xs, all_ys)
    cents   = dial_centroids(dial_fix)

    fig, ax = plt.subplots(figsize=(16, 8.4))
    overlay(ax, density, cents, r1, r2,
            f"Human  ({len(all_xs):,} fixations, {len(test_records)} test samples)")

    dial_colors = plt.cm.tab10(np.linspace(0, 0.6, 6))
    handles = ([Line2D([0],[0], marker="o", color="w",
                       markerfacecolor=dial_colors[i], markersize=9,
                       label=f"Dial {i+1}") for i in range(6)] +
               [Line2D([0],[0], color="white", lw=1.6, linestyle="-",
                       label=f"1-sigma p68={r1:.0f} px ({r1/CELL_SIZE:.1f} cells)"),
                Line2D([0],[0], color="white", lw=1.0, linestyle="--",
                       label=f"2-sigma p90={r2:.0f} px ({r2/CELL_SIZE:.1f} cells)")])
    ax.legend(handles=handles, loc="upper right", fontsize=8,
              facecolor="#333333", labelcolor="white", framealpha=0.85)

    plt.tight_layout()
    p = os.path.join(OUT_DIR, "test_heatmap_with_clusters.png")
    plt.savefig(p, dpi=130, bbox_inches="tight"); plt.close()
    print(f"Saved: {p}")


# ── Figure 2: human vs model ──────────────────────────────────────────────────

def run_model_on_test(test_records):
    import torch
    from dataset import ScanpathDataset
    from model import HeatmapARModel

    device = torch.device("cpu")

    print("  Loading datasets...")
    train_ds = ScanpathDataset(config.TRAIN_JSON,
                               max_fixations=config.MAX_FIXATIONS,
                               max_samples=config.MAX_TRAIN_SAMPLES)
    test_ds  = ScanpathDataset(config.TEST_JSON,
                               max_fixations=config.MAX_FIXATIONS,
                               norm_stats=train_ds.norm_stats,
                               cond_norm_stats=train_ds.cond_norm_stats)

    ckpt_path = (config.BEST_CHECKPOINT_PATH
                 if os.path.exists(config.BEST_CHECKPOINT_PATH)
                 else config.CHECKPOINT_PATH)
    print(f"  Checkpoint: {os.path.basename(ckpt_path)}")
    ckpt  = torch.load(ckpt_path, map_location=device)
    model = HeatmapARModel().to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"  epoch={ckpt.get('epoch','?')}  "
          f"best_dist={ckpt.get('best_dist', float('nan')):.1f} px")

    ds_index = {(s.get("subject"), frame_key(s["name"])): i
                for i, s in enumerate(test_ds.data)}

    dial_fix = defaultdict(list)
    all_xs, all_ys = [], []
    n_done = 0

    with torch.no_grad():
        for raw_s in test_records:
            length = raw_s["length"]
            if length == 0:
                continue
            idx = ds_index.get((raw_s.get("subject"), frame_key(raw_s["name"])))
            if idx is None:
                continue

            item = test_ds[idx]
            pred = model.generate(
                item["cond_geom"].unsqueeze(0),
                item["cond_signal"].unsqueeze(0),
                num_fixations=length,
                temperature=config.INFERENCE_TEMPERATURE,
            ).squeeze(0)   # [length, 4]

            px = ((pred[:, 0] + 1) / 2 * W).numpy()
            py = ((pred[:, 1] + 1) / 2 * H).numpy()

            # Dial centres from cond_geom (sorted by position 1..6)
            cx_arr = ((item["cond_geom"][:, 0] + 1) / 2 * W).numpy()
            cy_arr = ((item["cond_geom"][:, 1] + 1) / 2 * H).numpy()

            for x, y in zip(px, py):
                best = int(np.argmin((cx_arr - x)**2 + (cy_arr - y)**2))
                dial_fix[best + 1].append((x, y))
                all_xs.append(x); all_ys.append(y)

            n_done += 1
            if n_done % 500 == 0:
                print(f"    {n_done}/{len(test_records)} done...")

    print(f"  Done: {len(all_xs):,} predicted fixations.")
    return dial_fix, all_xs, all_ys


def plot_human_vs_model(test_records, r1, r2):
    # Human
    h_fix = defaultdict(list)
    h_xs, h_ys = [], []
    for s in test_records:
        n = s["length"]
        if n == 0 or not s["dials"]:
            continue
        fix = assign_to_dials(s["X"][:n], s["Y"][:n], s["dials"])
        for d, pts in fix.items():
            h_fix[d].extend(pts)
        h_xs.extend(s["X"][:n]); h_ys.extend(s["Y"][:n])

    # Model
    print("Running model inference...")
    m_fix, m_xs, m_ys = run_model_on_test(test_records)

    dial_colors = plt.cm.tab10(np.linspace(0, 0.6, 6))
    fig, axes = plt.subplots(1, 2, figsize=(28, 8.4))
    fig.suptitle(
        f"Fixation density: human (test) vs model  |  "
        f"1-sigma={r1:.0f} px   2-sigma={r2:.0f} px\n"
        f"% = fraction of total fixations on that dial",
        fontsize=11,
    )

    overlay(axes[0], build_density(h_xs, h_ys), dial_centroids(h_fix), r1, r2,
            f"Human  ({len(h_xs):,} fixations, {len(test_records)} test samples)")
    overlay(axes[1], build_density(m_xs, m_ys), dial_centroids(m_fix), r1, r2,
            f"Model  ({len(m_xs):,} predicted fixations, "
            f"T={config.INFERENCE_TEMPERATURE})")

    handles = ([Line2D([0],[0], marker="o", color="w",
                       markerfacecolor=dial_colors[i], markersize=9,
                       label=f"Dial {i+1}") for i in range(6)] +
               [Line2D([0],[0], color="white", lw=1.6, linestyle="-",
                       label=f"1-sigma p68={r1:.0f} px"),
                Line2D([0],[0], color="white", lw=1.0, linestyle="--",
                       label=f"2-sigma p90={r2:.0f} px")])
    fig.legend(handles=handles, loc="lower center", ncol=8, fontsize=8,
               facecolor="#333333", labelcolor="white", framealpha=0.85,
               bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout()
    p = os.path.join(OUT_DIR, "human_vs_model_heatmap.png")
    plt.savefig(p, dpi=130, bbox_inches="tight"); plt.close()
    print(f"Saved: {p}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    records      = load_all()
    test_records = [r for r in records if r["_split"] == "test"]
    print(f"Test samples: {len(test_records)}")

    print("Computing within-cluster spread for sigma rings...")
    wc_px = within_cluster_spread(records)
    r1    = np.percentile(wc_px, 68)   # ~100 px
    r2    = np.percentile(wc_px, 90)   # ~140 px
    print(f"  1-sigma (p68) = {r1:.0f} px = {r1/CELL_SIZE:.1f} cells")
    print(f"  2-sigma (p90) = {r2:.0f} px = {r2/CELL_SIZE:.1f} cells")

    plot_human_density(test_records, r1, r2)
    plot_human_vs_model(test_records, r1, r2)

    print(f"\nAll figures saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
