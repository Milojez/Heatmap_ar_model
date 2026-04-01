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
N_AR_RUNS        = 3     # generation runs averaged per prediction
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


def target_for_participant(pid, by_frame, sorted_keys):
    target = [next((s for s in by_frame[k] if s['subject'] == pid), None)
              for k in sorted_keys]
    return [(k, s) for k, s in zip(sorted_keys, target) if s is not None]


# ── Prediction cache ───────────────────────────────────────────────────────────

def _cache_path(pid):
    return PRED_CACHE_DIR / f"{CKPT_PATH.stem}_p{pid}.json"


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

            # Average N_AR_RUNS independent samples for a stable prediction
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

            predictions.append(np.mean(run_xys, axis=0))

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
                  origin='lower', aspect='auto', extent=[0, W, 0, H], cmap=cmap)
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


def plot_routing(target_fracs, model_fracs, tag):
    dials = sorted(AOI_BY_POSITION)
    x, w  = np.arange(len(dials)), 0.35
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(x - w/2, [target_fracs[d] for d in dials], w, label=f'P{PARTICIPANT_ID}',
           color='steelblue', alpha=0.85)
    ax.bar(x + w/2, [model_fracs[d]  for d in dials], w, label='Model',
           color='tomato', alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels([f"Dial {d}" for d in dials])
    ax.set_ylabel("Fraction of dwell time")
    ax.set_title(f"AOI dwell fraction  |  {tag}")
    ax.legend(); ax.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    p = OUT_DIR / f"{tag}_aoi_fractions.png"
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

    # Collect fixations for heatmap
    all_xs, all_ys = [], []
    for samples in tqdm(by_frame.values(), desc="Collecting fixations", unit="frame"):
        for s in samples:
            all_xs.extend(s['X'][:s['length']]); all_ys.extend(s['Y'][:s['length']])

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

    plot_routing(aoi_fracs(target_xs, target_ys, target_Ts),
                 aoi_fracs(pred_xs, pred_ys), tag)
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

    target_fracs = aoi_fracs(target_xs, target_ys, target_Ts)
    model_fracs  = aoi_fracs(pred_xs, pred_ys)
    print(f"\nAOI fractions — P{PARTICIPANT_ID} vs Model:")
    print(f"  {'Dial':<6} {'P'+str(PARTICIPANT_ID):>8} {'Model':>8} {'Diff':>8}")
    for d in sorted(AOI_BY_POSITION):
        print(f"  {d:<6} {target_fracs[d]:>8.3f} {model_fracs[d]:>8.3f} "
              f"{model_fracs[d]-target_fracs[d]:>+8.3f}")

    print(f"\nAll outputs saved to {OUT_DIR}/")


if __name__ == '__main__':
    main()
