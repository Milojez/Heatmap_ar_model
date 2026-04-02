"""
Per-participant test analysis for the Heatmap AR model.

Produces:
  1. Three fixation density heatmaps:  all participants | target participant | model
  2. Per-sample Dist(H) / center / spread:
       - Target participant vs group
       - Model (conditioned on target) vs group
       - Average across ALL participants vs group  (RUN_ALL_SUBJECTS=True)
  3. AOI dwell fraction bar chart
  4. Proximity-to-AOI correlation per dial

Inference is cached to data/predictions/ — rerun skips inference.
"""

import re
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
import scipy.ndimage as ndimage
from pathlib import Path
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm
import sys

sys.path.insert(0, str(Path(__file__).parent))

import config
from model import HeatmapARModel
from dataset import ScanpathDataset, denorm_duration

# ── User settings ──────────────────────────────────────────────────────────────
PARTICIPANT_ID   = 1
CKPT_PATH        = Path(config.BEST_CHECKPOINT_PATH)
N_AR_RUNS        = 1     # generation runs per test sample (1 = comparable to participant)
RUN_ALL_SUBJECTS = False
OUT_DIR          = Path(config.CHECKPOINT_DIR) / "test_analysis"
PRED_CACHE_DIR   = Path(config.PRED_DIR)
# ──────────────────────────────────────────────────────────────────────────────

AOI_BY_POSITION = {
    1: (116,  536,    1,  421),
    2: (750,  1170,   1,  421),
    3: (1385, 1805,   1,  421),
    4: (116,  536,  659, 1079),
    5: (750,  1170, 659, 1079),
    6: (1385, 1805, 659, 1079),
}
W, H = config.IMAGE_WIDTH, config.IMAGE_HEIGHT


# ── Small helpers ──────────────────────────────────────────────────────────────

def frame_key(name):
    first = name[0] if isinstance(name, list) else name
    m = re.match(r'.+?_frame_(\d+)', first)
    return int(m.group(1)) if m else 0


def hungarian_dist(xy1, xy2):
    n = min(len(xy1), len(xy2))
    if n == 0: return float('nan')
    cost = np.linalg.norm(xy1[:n, None, :] - xy2[None, :n, :], axis=-1)
    r, c = linear_sum_assignment(cost)
    return float(cost[r, c].mean())


def center_loss(xy1, xy2):
    n = min(len(xy1), len(xy2))
    if n == 0: return float('nan')
    cost = np.linalg.norm(xy1[:n, None, :] - xy2[None, :n, :], axis=-1)
    r, c = linear_sum_assignment(cost)
    return float(np.linalg.norm(xy1[r].mean(axis=0) - xy2[c].mean(axis=0)))


def spread_loss(xy1, xy2):
    n = min(len(xy1), len(xy2))
    if n == 0: return float('nan')
    cost = np.linalg.norm(xy1[:n, None, :] - xy2[None, :n, :], axis=-1)
    r, c = linear_sum_assignment(cost)
    s1 = np.linalg.norm(xy1[r] - xy1[r].mean(axis=0), axis=-1).mean()
    s2 = np.linalg.norm(xy2[c] - xy2[c].mean(axis=0), axis=-1).mean()
    return float(abs(s1 - s2))


def make_density(xs, ys, sigma_px=25, bins=(95, 49)):
    hist, _, _ = np.histogram2d(xs, ys, bins=bins, range=[[0, W], [0, H]])
    hist = ndimage.gaussian_filter(hist.T, sigma=sigma_px * bins[0] / W)
    return hist


def aoi_fracs(xs, ys, Ts=None):
    xs, ys  = np.array(xs), np.array(ys)
    weights = np.array(Ts) if Ts is not None else np.ones(len(xs))
    total   = weights.sum() + 1e-12
    return {pos: float(weights[(xs > x0) & (xs <= x1) & (ys >= y0) & (ys < y1)].sum() / total)
            for pos, (x0, x1, y0, y1) in AOI_BY_POSITION.items()}


def fixations_xy(item_raw):
    n = item_raw['length']
    return np.stack([item_raw['X'][:n], item_raw['Y'][:n]], axis=1).astype(float)


# ── Data loading ───────────────────────────────────────────────────────────────

def load_test_data():
    with open(config.TEST_JSON, encoding='utf-8') as f:
        raw = json.load(f)
    by_frame = {}
    for s in raw:
        by_frame.setdefault(frame_key(s['name']), []).append(s)
    sorted_keys = sorted(by_frame)
    return raw, by_frame, sorted_keys


def train_aoi_fracs():
    """Return (dwell_fracs, count_fracs) pooled over the entire training set."""
    with open(config.TRAIN_JSON, encoding='utf-8') as f:
        train_raw = json.load(f)
    xs, ys, Ts = [], [], []
    for s in train_raw:
        n = s['length']
        xs.extend(s['X'][:n])
        ys.extend(s['Y'][:n])
        Ts.extend(s['T'][:n])
    print(f"Training data: {len(train_raw)} samples, {len(xs)} fixations total.")
    return aoi_fracs(xs, ys, Ts), aoi_fracs(xs, ys)


def target_for_participant(pid, by_frame, sorted_keys):
    target = [next((s for s in by_frame[k] if s['subject'] == pid), None)
              for k in sorted_keys]
    return [(k, s) for k, s in zip(sorted_keys, target) if s is not None]


# ── Prediction cache ───────────────────────────────────────────────────────────

def _cache_path(pid):
    return PRED_CACHE_DIR / f"{CKPT_PATH.stem}_p{pid}_runs{N_AR_RUNS}.json"


def _save_predictions(predictions, target, pid):
    PRED_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache = [{"frame_key": k, "xy": p.tolist()} if p is not None else None
             for (k, _), p in zip(target, predictions)]
    with open(_cache_path(pid), 'w') as f:
        json.dump(cache, f)


def _load_predictions(pid):
    path = _cache_path(pid)
    if not path.exists():
        return None
    with open(path) as f:
        cache = json.load(f)
    return [np.array(e["xy"]) if e is not None else None for e in cache]


# ── Model ──────────────────────────────────────────────────────────────────────

def load_model(device):
    if not CKPT_PATH.exists():
        raise FileNotFoundError(f"Checkpoint not found: {CKPT_PATH}")
    ckpt  = torch.load(CKPT_PATH, map_location=device)
    model = HeatmapARModel().to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    print(f"Loaded: {CKPT_PATH.name}  epoch={ckpt.get('epoch','?')}")
    return model


# ── Inference ──────────────────────────────────────────────────────────────────

def run_inference(model, test_ds, target, device, pid):
    cached = _load_predictions(pid)
    if cached is not None:
        print(f"  P{pid}: loaded {len(cached)} cached predictions.")
        return cached

    ds_index = {(s['subject'], frame_key(s['name'])): i
                for i, s in enumerate(test_ds.data)}

    predictions = []
    with torch.no_grad():
        for k, raw_s in tqdm(target, desc=f"Inference P{pid}", unit="sample"):
            ds_idx = ds_index.get((pid, k))
            if ds_idx is None:
                predictions.append(None)
                continue

            item   = test_ds[ds_idx]
            length = raw_s['length']

            cond_geom   = item['cond_geom'].unsqueeze(0).to(device)
            cond_signal = item['cond_signal'].unsqueeze(0).to(device)

            # Concatenate N_AR_RUNS independent runs — each is a valid scanpath.
            # Do NOT average coordinates: runs sampling different dials would create
            # phantom fixations in empty space between dials.
            run_xys = []
            for _ in range(N_AR_RUNS):
                pred = model.generate(
                    cond_geom, cond_signal,
                    num_fixations=length,
                    temperature=config.INFERENCE_TEMPERATURE,
                ).squeeze(0).cpu()   # [length, 4]
                xs = (pred[:, 0] + 1) / 2 * W
                ys = (pred[:, 1] + 1) / 2 * H
                run_xys.append(np.stack([xs.numpy(), ys.numpy()], axis=1))

            predictions.append(np.concatenate(run_xys, axis=0))  # [N_AR_RUNS*length, 2]

    _save_predictions(predictions, target, pid)
    return predictions


# ── Per-sample metrics ─────────────────────────────────────────────────────────

def compute_sample_metrics(target, predictions, by_frame, pid):
    p_dist, p_center, p_spread = [], [], []
    m_dist, m_center, m_spread = [], [], []

    for (k, raw_s), pred_xy in zip(target, predictions):
        group = [fixations_xy(s) for s in by_frame[k]
                 if s['subject'] != pid and len(s['X']) > 0]
        t_xy  = fixations_xy(raw_s)

        p_dist.append(float(np.nanmean([hungarian_dist(t_xy, g) for g in group])) if group else float('nan'))
        p_center.append(float(np.nanmean([center_loss(t_xy, g)  for g in group])) if group else float('nan'))
        p_spread.append(float(np.nanmean([spread_loss(t_xy, g)  for g in group])) if group else float('nan'))

        if pred_xy is not None and group:
            m_dist.append(float(np.nanmean([hungarian_dist(pred_xy, g) for g in group])))
            m_center.append(float(np.nanmean([center_loss(pred_xy, g)  for g in group])))
            m_spread.append(float(np.nanmean([spread_loss(pred_xy, g)  for g in group])))
        else:
            m_dist.append(float('nan')); m_center.append(float('nan')); m_spread.append(float('nan'))

    return (p_dist, p_center, p_spread), (m_dist, m_center, m_spread)


# ── Plots ──────────────────────────────────────────────────────────────────────

def plot_heatmaps(all_xs, all_ys, target_xs, target_ys, pred_xs, pred_ys, tag):
    fig, axes = plt.subplots(1, 3, figsize=(21, 5))
    for ax, (xs, ys, cmap), title in zip(
            axes,
            [(all_xs, all_ys, 'Greens'), (target_xs, target_ys, 'Blues'), (pred_xs, pred_ys, 'Reds')],
            [f"All participants ({len(all_xs)} fix.)",
             f"P{PARTICIPANT_ID} ({len(target_xs)} fix.)",
             f"Model ({len(pred_xs)} fix.)"]):
        if not xs:
            ax.set_title(title + "\n(no data)"); continue
        ax.imshow(make_density(np.array(xs), np.array(ys)),
                  origin='upper', aspect='auto', extent=[0, W, H, 0], cmap=cmap)
        for pos, (x0, x1, y0, y1) in AOI_BY_POSITION.items():
            ax.add_patch(plt.Rectangle((x0, y0), x1-x0, y1-y0,
                                       fill=False, edgecolor='white',
                                       linewidth=0.8, linestyle='--'))
            ax.text((x0+x1)/2, (y0+y1)/2, str(pos), color='white',
                    ha='center', va='center', fontsize=8, fontweight='bold')
        ax.set_title(title, fontsize=10); ax.axis('off')
    fig.suptitle(f"Fixation heatmaps — {tag}", fontsize=12)
    plt.tight_layout()
    p = OUT_DIR / f"{tag}_heatmaps.png"
    plt.savefig(p, dpi=120, bbox_inches='tight'); plt.close()
    print(f"Saved -> {p}")


def plot_metrics(p_metrics, m_metrics, avg_p_metrics, tag):
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    colors = {'target': 'steelblue', 'model': 'tomato', 'avg': 'gray'}
    avg_cols = list(zip(*avg_p_metrics)) if avg_p_metrics else [None, None, None]

    for ax, name, p_vals, m_vals, avg_vals in zip(
            axes,
            ['Dist(H) [px]', 'Center loss [px]', 'Spread loss [px]'],
            zip(*p_metrics), zip(*m_metrics), avg_cols):

        def _plot(vals, color, label, ls='-'):
            vals = list(vals)
            valid = [(i, v) for i, v in enumerate(vals) if not np.isnan(v)]
            if not valid: return
            idx, ys = zip(*valid)
            ax.plot(idx, ys, color=color, linewidth=1.2, linestyle=ls,
                    label=f"{label} mean={np.nanmean(vals):.0f}px")
            ax.scatter(idx, ys, color=color, s=12, zorder=3)
            ax.axhline(np.nanmean(vals), color=color, linestyle=':', alpha=0.5)

        _plot(p_vals, colors['target'], f'P{PARTICIPANT_ID} vs group')
        _plot(m_vals, colors['model'],  'Model vs group', ls='--')
        if avg_vals is not None:
            _plot(avg_vals, colors['avg'], 'Avg all P vs group', ls='-.')
        ax.set_ylabel(name); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Sample index (chronological)")
    fig.suptitle(f"Per-sample metrics vs group  |  {tag}", fontsize=12)
    plt.tight_layout()
    p = OUT_DIR / f"{tag}_metrics.png"
    plt.savefig(p, dpi=120, bbox_inches='tight'); plt.close()
    print(f"Saved -> {p}")


def _four_bar_plot(ax, dials, vals_list, labels, colors, ylabel, title):
    x = np.arange(len(dials))
    n = len(vals_list)
    w = 0.8 / n
    offsets = np.linspace(-(n - 1) / 2, (n - 1) / 2, n) * w
    for vals, label, color, offset in zip(vals_list, labels, colors, offsets):
        ax.bar(x + offset, [vals[d] for d in dials], w,
               label=label, color=color, alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels([f"Dial {d}" for d in dials])
    ax.set_ylabel(ylabel); ax.set_title(title)
    ax.legend(fontsize=8); ax.grid(True, axis='y', alpha=0.3)


def plot_routing(target_fracs, all_test_fracs, model_fracs, train_fracs, tag):
    dials = sorted(AOI_BY_POSITION)
    fig, ax = plt.subplots(figsize=(11, 4))
    _four_bar_plot(
        ax, dials,
        [target_fracs, all_test_fracs, model_fracs, train_fracs],
        [f'P{PARTICIPANT_ID}', 'All test participants', 'Model', 'Training data'],
        ['steelblue', 'gold', 'tomato', 'seagreen'],
        "Fraction of dwell time",
        f"AOI dwell-time fraction  |  {tag}",
    )
    plt.tight_layout()
    p = OUT_DIR / f"{tag}_aoi_fractions.png"
    plt.savefig(p, dpi=120, bbox_inches='tight'); plt.close()
    print(f"Saved -> {p}")


def plot_routing_count(target_count, all_test_count, model_count, train_count, tag):
    dials = sorted(AOI_BY_POSITION)
    fig, ax = plt.subplots(figsize=(11, 4))
    _four_bar_plot(
        ax, dials,
        [target_count, all_test_count, model_count, train_count],
        [f'P{PARTICIPANT_ID}', 'All test participants', 'Model', 'Training data'],
        ['steelblue', 'gold', 'tomato', 'seagreen'],
        "Fraction of fixation count",
        f"AOI fixation-count fraction  |  {tag}",
    )
    plt.tight_layout()
    p = OUT_DIR / f"{tag}_aoi_count_fractions.png"
    plt.savefig(p, dpi=120, bbox_inches='tight'); plt.close()
    print(f"Saved -> {p}")


def dial_by_fixnum(samples_xy_dur, max_fixations):
    """
    Compute dwell-time (or count if dur=None) fraction per (fixation_index, dial).

    samples_xy_dur : list of (xy [N,2], dur [N] or None)
    Returns        : [max_fixations, 6]  — each row sums to ≤1 (0 where no data)
    """
    sorted_dials = sorted(AOI_BY_POSITION.items())   # [(pos, (x0,x1,y0,y1)), ...]
    dial_time  = np.zeros((max_fixations, 6))
    total_time = np.zeros(max_fixations)

    for xy, dur in samples_xy_dur:
        n = min(len(xy), max_fixations)
        for fi in range(n):
            x, y = xy[fi, 0], xy[fi, 1]
            w = float(dur[fi]) if dur is not None else 1.0
            for di, (_, (x0, x1, y0, y1)) in enumerate(sorted_dials):
                if x0 < x <= x1 and y0 <= y < y1:
                    dial_time[fi, di] += w
                    break
            total_time[fi] += w

    fracs = np.zeros_like(dial_time)
    mask  = total_time > 0
    fracs[mask] = dial_time[mask] / total_time[mask, None]
    return fracs   # [max_fixations, 6]


def plot_routing_by_fixnum(target, predictions, by_frame, tag):
    """
    Four heatmaps [dials × fixation_index]:
      • P{PARTICIPANT_ID}      — actual dwell time per step
      • All test participants  — dwell time per step (pooled)
      • Model                  — fixation count per step (N_AR_RUNS runs)
      • Training data          — dwell time per step (all training samples)
    """
    MF = config.MAX_FIXATIONS

    # ── P{PARTICIPANT_ID} ─────────────────────────────────────────────────────
    part_samples = []
    for _, raw_s in target:
        n = raw_s['length']
        if n == 0:
            continue
        xy  = np.stack([raw_s['X'][:n], raw_s['Y'][:n]], axis=1).astype(float)
        dur = np.array(raw_s['T'][:n], dtype=float)
        part_samples.append((xy, dur))
    fracs_part = dial_by_fixnum(part_samples, MF)

    # ── All test participants (pooled) ────────────────────────────────────────
    all_test_samples = []
    for samples in by_frame.values():
        for s in samples:
            n = s['length']
            if n == 0:
                continue
            xy  = np.stack([s['X'][:n], s['Y'][:n]], axis=1).astype(float)
            dur = np.array(s['T'][:n], dtype=float)
            all_test_samples.append((xy, dur))
    fracs_all = dial_by_fixnum(all_test_samples, MF)

    # ── Model (split concatenated runs back into individual scanpaths) ────────
    model_samples = []
    for (_, raw_s), pred_xy in zip(target, predictions):
        if pred_xy is None:
            continue
        length = raw_s['length']
        if length == 0:
            continue
        n_runs = len(pred_xy) // length
        for r in range(n_runs):
            run_xy = pred_xy[r * length:(r + 1) * length]
            model_samples.append((run_xy, None))   # no duration from model
    fracs_model = dial_by_fixnum(model_samples, MF)

    # ── Training data ─────────────────────────────────────────────────────────
    with open(config.TRAIN_JSON, encoding='utf-8') as f:
        train_raw = json.load(f)
    train_samples = []
    for s in train_raw:
        n = s['length']
        if n == 0:
            continue
        xy  = np.stack([s['X'][:n], s['Y'][:n]], axis=1).astype(float)
        dur = np.array(s['T'][:n], dtype=float)
        train_samples.append((xy, dur))
    fracs_train = dial_by_fixnum(train_samples, MF)

    # ── Plot ──────────────────────────────────────────────────────────────────
    dial_labels = [f"Dial {d}" for d in sorted(AOI_BY_POSITION)]
    fix_labels  = [str(i + 1) for i in range(MF)]

    titles = [
        f"P{PARTICIPANT_ID} — dwell time\n(N={len(part_samples)} scanpaths)",
        f"All test participants — dwell time\n(N={len(all_test_samples)} scanpaths)",
        f"Model — fixation count\n(N={len(model_samples)} runs)",
        f"Training data — dwell time\n(N={len(train_samples)} scanpaths)",
    ]
    matrices = [fracs_part.T, fracs_all.T, fracs_model.T, fracs_train.T]   # each [6, MF]

    # shared colour scale across all panels so comparisons are honest
    vmax = max(m.max() for m in matrices) or 1.0

    fig, axes = plt.subplots(1, 4, figsize=(22, 4), sharey=True)
    for ax, mat, title in zip(axes, matrices, titles):
        im = ax.imshow(mat, aspect='auto', origin='upper',
                       cmap='YlOrRd', vmin=0, vmax=vmax)
        ax.set_xticks(range(MF));   ax.set_xticklabels(fix_labels, fontsize=8)
        ax.set_yticks(range(6));    ax.set_yticklabels(dial_labels, fontsize=8)
        ax.set_xlabel("Fixation index")
        ax.set_title(title, fontsize=9)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Fraction")

    fig.suptitle(f"Dwell-time fraction per fixation step  |  {tag}\n"
                 f"Does the model track the participant / test population, "
                 f"or just the training prior?",
                 fontsize=11)
    plt.tight_layout()
    p = OUT_DIR / f"{tag}_aoi_by_fixnum.png"
    plt.savefig(p, dpi=120, bbox_inches='tight'); plt.close()
    print(f"Saved -> {p}")


def plot_proximity_correlation(target, by_frame, predictions, tag):
    dials = sorted(AOI_BY_POSITION)
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))

    for ax, dial_pos in tqdm(zip(axes.flat, dials), total=len(dials),
                              desc="Proximity corr", unit="dial"):
        x0, x1, y0, y1 = AOI_BY_POSITION[dial_pos]
        prox_vals, target_in, model_in = [], [], []

        for (k, raw_s), pred_xy in zip(target, predictions):
            dial_data = next((d for d in raw_s['dials'] if d['dial_position'] == dial_pos), None)
            if dial_data is None or pred_xy is None or 'needle_to_threshold_norm' not in dial_data:
                continue
            prox_vals.append(float(np.mean(dial_data['needle_to_threshold_norm'])))
            txy = fixations_xy(raw_s)
            target_in.append(float(np.any(
                (txy[:,0]>x0) & (txy[:,0]<=x1) & (txy[:,1]>=y0) & (txy[:,1]<y1)
            )))
            model_in.append(float(np.any(
                (pred_xy[:,0]>x0) & (pred_xy[:,0]<=x1) &
                (pred_xy[:,1]>=y0) & (pred_xy[:,1]<y1)
            )))

        prox_arr = np.array(prox_vals)
        t_arr, m_arr = np.array(target_in), np.array(model_in)

        if len(prox_arr) > 5:
            bins    = np.percentile(prox_arr, [0, 20, 40, 60, 80, 100])
            bin_ids = np.digitize(prox_arr, bins[1:-1])
            centers = [(bins[i]+bins[i+1])/2 for i in range(len(bins)-1)]
            t_rates = [t_arr[bin_ids==b].mean() if (bin_ids==b).any() else np.nan for b in range(len(centers))]
            m_rates = [m_arr[bin_ids==b].mean() if (bin_ids==b).any() else np.nan for b in range(len(centers))]
            ax.plot(centers, t_rates, 'o-',  color='steelblue', label=f'P{PARTICIPANT_ID}', linewidth=1.5)
            ax.plot(centers, m_rates, 's--', color='tomato',   label='Model',               linewidth=1.5)
        else:
            ax.scatter(prox_arr, t_arr, color='steelblue', alpha=0.4, s=12)
            ax.scatter(prox_arr, m_arr, color='tomato',    alpha=0.4, s=12)

        title = f"Dial {dial_pos}"
        if len(prox_arr) > 2:
            title += (f"  r_P={np.corrcoef(prox_arr,t_arr)[0,1]:.2f}"
                      f"  r_M={np.corrcoef(prox_arr,m_arr)[0,1]:.2f}")
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("Needle proximity (lower = closer to threshold)")
        ax.set_ylabel("P(fixate on AOI)")
        ax.set_ylim(-0.05, 1.05); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    fig.suptitle(f"Proximity-to-AOI fixation rate  |  {tag}", fontsize=11)
    plt.tight_layout()
    p = OUT_DIR / f"{tag}_proximity_correlation.png"
    plt.savefig(p, dpi=120, bbox_inches='tight'); plt.close()
    print(f"Saved -> {p}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tag    = f"{CKPT_PATH.stem}_p{PARTICIPANT_ID}"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PRED_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    raw, by_frame, sorted_keys = load_test_data()
    target = target_for_participant(PARTICIPANT_ID, by_frame, sorted_keys)
    if not target:
        raise ValueError(f"Participant {PARTICIPANT_ID} not found in test data.")
    print(f"P{PARTICIPANT_ID}: {len(target)} samples in test set.")

    print("Loading datasets...")
    train_ds = ScanpathDataset(config.TRAIN_JSON, max_fixations=config.MAX_FIXATIONS,
                               max_samples=config.MAX_TRAIN_SAMPLES)
    test_ds  = ScanpathDataset(config.TEST_JSON,  max_fixations=config.MAX_FIXATIONS,
                               norm_stats=train_ds.norm_stats,
                               cond_norm_stats=train_ds.cond_norm_stats)

    print("Loading model...")
    model = load_model(device)

    predictions = run_inference(model, test_ds, target, device, PARTICIPANT_ID)

    (p_dist, p_center, p_spread), (m_dist, m_center, m_spread) = \
        compute_sample_metrics(target, predictions, by_frame, PARTICIPANT_ID)

    avg_p_dist_aligned = avg_p_center_aligned = avg_p_spread_aligned = None
    all_p_mean_dist = all_p_mean_center = all_p_mean_spread = []

    if RUN_ALL_SUBJECTS:
        all_subjects = sorted({s['subject'] for samples in by_frame.values() for s in samples})
        print(f"\nRunning/loading inference for all {len(all_subjects)} participants...")
        per_sample_d = [[] for _ in sorted_keys]
        per_sample_c = [[] for _ in sorted_keys]
        per_sample_s = [[] for _ in sorted_keys]
        frame_to_idx = {k: i for i, k in enumerate(sorted_keys)}

        all_p_mean_dist, all_p_mean_center, all_p_mean_spread = [], [], []
        for pid in tqdm(all_subjects, desc="All participants", unit="participant"):
            t_p = target_for_participant(pid, by_frame, sorted_keys)
            if not t_p: continue
            preds_p = run_inference(model, test_ds, t_p, device, pid)
            (pd, pc, ps), _ = compute_sample_metrics(t_p, preds_p, by_frame, pid)
            all_p_mean_dist.append(np.nanmean(pd))
            all_p_mean_center.append(np.nanmean(pc))
            all_p_mean_spread.append(np.nanmean(ps))
            for (k, _), dv, cv, sv in zip(t_p, pd, pc, ps):
                si = frame_to_idx.get(k)
                if si is not None:
                    per_sample_d[si].append(dv)
                    per_sample_c[si].append(cv)
                    per_sample_s[si].append(sv)

        avg_d = [float(np.nanmean(v)) if v else float('nan') for v in per_sample_d]
        avg_c = [float(np.nanmean(v)) if v else float('nan') for v in per_sample_c]
        avg_s = [float(np.nanmean(v)) if v else float('nan') for v in per_sample_s]
        fti   = {k: i for i, k in enumerate(sorted_keys)}
        avg_p_dist_aligned   = [avg_d[fti[k]] for k, _ in target]
        avg_p_center_aligned = [avg_c[fti[k]] for k, _ in target]
        avg_p_spread_aligned = [avg_s[fti[k]] for k, _ in target]

    # Collect fixations for heatmap + AOI fractions
    all_xs, all_ys, all_Ts = [], [], []
    for samples in tqdm(by_frame.values(), desc="Collecting fixations", unit="frame"):
        for s in samples:
            n = s['length']
            all_xs.extend(s['X'][:n]); all_ys.extend(s['Y'][:n]); all_Ts.extend(s['T'][:n])

    target_xs, target_ys, target_Ts = [], [], []
    for _, s in target:
        n = s['length']
        target_xs.extend(s['X'][:n]); target_ys.extend(s['Y'][:n]); target_Ts.extend(s['T'][:n])

    pred_xs, pred_ys = [], []
    for p in predictions:
        if p is not None:
            pred_xs.extend(p[:, 0]); pred_ys.extend(p[:, 1])

    plot_heatmaps(all_xs, all_ys, target_xs, target_ys, pred_xs, pred_ys, tag)

    avg_series = (list(zip(avg_p_dist_aligned, avg_p_center_aligned, avg_p_spread_aligned))
                  if avg_p_dist_aligned is not None else None)
    plot_metrics(list(zip(p_dist, p_center, p_spread)),
                 list(zip(m_dist, m_center, m_spread)),
                 avg_series, tag)

    train_dwell_fracs, train_count_fracs = train_aoi_fracs()

    plot_routing(
        aoi_fracs(target_xs, target_ys, target_Ts),   # P1 dwell
        aoi_fracs(all_xs, all_ys, all_Ts),             # all test dwell
        aoi_fracs(pred_xs, pred_ys),                   # model (count — no durations)
        train_dwell_fracs,                              # training dwell
        tag,
    )
    plot_routing_count(
        aoi_fracs(target_xs, target_ys),               # P1 count
        aoi_fracs(all_xs, all_ys),                     # all test count
        aoi_fracs(pred_xs, pred_ys),                   # model count
        train_count_fracs,                              # training count
        tag,
    )
    plot_routing_by_fixnum(target, predictions, by_frame, tag)
    plot_proximity_correlation(target, by_frame, predictions, tag)

    header = f"{'Metric':<22} {'P'+str(PARTICIPANT_ID)+' vs group':>18} {'Model vs group':>18}"
    if RUN_ALL_SUBJECTS:
        header += f" {'Avg all P vs group':>20}"
    print(f"\n{header}\n{'-'*len(header)}")
    for name, pv, mv, av in zip(
            ['Dist(H) [px]', 'Center loss [px]', 'Spread loss [px]'],
            [p_dist, p_center, p_spread],
            [m_dist, m_center, m_spread],
            [all_p_mean_dist, all_p_mean_center, all_p_mean_spread]):
        row = f"{name:<22} {np.nanmean(pv):>18.1f} {np.nanmean(mv):>18.1f}"
        if RUN_ALL_SUBJECTS:
            row += f" {np.nanmean(av):>20.1f}"
        print(row)

    target_fracs   = aoi_fracs(target_xs, target_ys, target_Ts)
    all_test_fracs = aoi_fracs(all_xs, all_ys, all_Ts)
    model_fracs    = aoi_fracs(pred_xs, pred_ys)
    print(f"\nAOI dwell fractions — P{PARTICIPANT_ID} | All test | Model | Training:")
    print(f"  {'Dial':<6} {'P'+str(PARTICIPANT_ID):>8} {'AllTest':>8} {'Model':>8} {'Train':>8} {'M-P':>8} {'M-T':>8}")
    for d in sorted(AOI_BY_POSITION):
        print(f"  {d:<6} {target_fracs[d]:>8.3f} {all_test_fracs[d]:>8.3f} "
              f"{model_fracs[d]:>8.3f} {train_dwell_fracs[d]:>8.3f} "
              f"{model_fracs[d]-target_fracs[d]:>+8.3f} "
              f"{model_fracs[d]-train_dwell_fracs[d]:>+8.3f}")

    print(f"\nAll outputs saved to {OUT_DIR}/")


if __name__ == '__main__':
    main()
