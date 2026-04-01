"""
Evaluation metrics for the Heatmap AR model.
Reuses the same metric suite as the autoregressive_model:
  Hungarian-matched pixel distance, KLD, MultiMatch.
"""

import torch
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.stats import entropy
from multimatch_gaze import docomparison

import config
from dataset import denorm_delta_t, denorm_duration


def seq_to_pixels(xy_norm, width, height):
    """xy_norm: [B, N, 2] → [B, N, 2] pixel coords."""
    xs = (xy_norm[:, :, 0] + 1) / 2 * width
    ys = (xy_norm[:, :, 1] + 1) / 2 * height
    return torch.stack([xs, ys], dim=-1)


def compute_kld(pred_px, gt_px, lengths, nbins=64):
    pred_np = pred_px.detach().cpu().numpy()
    gt_np   = gt_px.detach().cpu().numpy()
    pred_x, pred_y, gt_x, gt_y = [], [], [], []
    for b in range(pred_np.shape[0]):
        n = int(lengths[b])
        pred_x.extend(pred_np[b, :n, 0].tolist())
        pred_y.extend(pred_np[b, :n, 1].tolist())
        gt_x.extend(gt_np[b,   :n, 0].tolist())
        gt_y.extend(gt_np[b,   :n, 1].tolist())

    def kld_1d(real, gen):
        lo, hi = min(min(real), min(gen)), max(max(real), max(gen))
        bins   = np.linspace(lo, hi, nbins + 1)
        hr     = np.histogram(real, bins=bins)[0].astype(float)
        hg     = np.histogram(gen,  bins=bins)[0].astype(float)
        hr /= hr.sum() + 1e-10
        hg /= hg.sum() + 1e-10
        return float(entropy(hr, hg + 1e-10))

    kld_x = kld_1d(gt_x, pred_x)
    kld_y = kld_1d(gt_y, pred_y)
    return {'kld_x': kld_x, 'kld_y': kld_y, 'kld_mean': (kld_x + kld_y) / 2}


def compute_multimatch(pred_px, gt_px, pred_seq, gt_seq,
                       lengths, norm_stats, width, height):
    pred_np  = pred_px.detach().cpu().numpy()
    gt_np    = gt_px.detach().cpu().numpy()
    pred_cpu = pred_seq.detach().cpu()
    gt_cpu   = gt_seq.detach().cpu()

    scores = []
    for b in range(pred_np.shape[0]):
        n  = max(int(lengths[b]), 3)
        pp = pred_np[b, :n]
        gp = gt_np[b,   :n]

        # pred_seq[b]: [N, 4], T_norm at index 3
        p_T = denorm_duration(pred_cpu[b, :n, 3], norm_stats)
        g_T = denorm_duration(gt_cpu[b,   :n, 3], norm_stats)

        sp1 = pd.DataFrame({'start_x': pp[:, 0], 'start_y': pp[:, 1],
                            'duration': p_T.numpy()}).to_records()
        sp2 = pd.DataFrame({'start_x': gp[:, 0], 'start_y': gp[:, 1],
                            'duration': g_T.numpy()}).to_records()
        try:
            scores.append(docomparison(sp1, sp2, screensize=(width, height)))
        except Exception:
            pass

    if not scores:
        return {k: float('nan') for k in
                ['mm_shape', 'mm_direction', 'mm_length', 'mm_position', 'mm_duration']}
    scores_np = np.array(scores)
    labels    = ['mm_shape', 'mm_direction', 'mm_length', 'mm_position', 'mm_duration']
    return {k: float(np.nanmean(scores_np[:, i])) for i, k in enumerate(labels)}


def compute_hungarian_metrics(pred_px, gt_px, pred_seq, gt_seq,
                               lengths, norm_stats, width=1904, height=988):
    B       = pred_px.shape[0]
    pred_np = pred_px.detach().cpu().numpy()
    gt_np   = gt_px.detach().cpu().numpy()
    pred_cpu = pred_seq.detach().cpu()
    gt_cpu   = gt_seq.detach().cpu()

    all_dists, all_center, all_spread = [], [], []
    all_dt_mae, all_T_mae = [], []

    for b in range(B):
        n  = int(lengths[b])
        pp = pred_np[b, :n]
        gp = gt_np[b,   :n]
        cost = np.linalg.norm(pp[:, None, :] - gp[None, :, :], axis=-1)
        r, c = linear_sum_assignment(cost)
        all_dists.extend(cost[r, c].tolist())

        pp_m, gp_m = pp[r], gp[c]
        c_pred = pp_m.mean(0);  c_gt = gp_m.mean(0)
        all_center.append(float(np.abs(c_pred - c_gt).mean()))
        all_spread.append(float(abs(
            np.linalg.norm(pp_m - c_pred, axis=-1).mean() -
            np.linalg.norm(gp_m - c_gt,   axis=-1).mean()
        )))

        p_dt = denorm_delta_t(pred_cpu[b, :n, 2], norm_stats)
        g_dt = denorm_delta_t(gt_cpu[b,   :n, 2], norm_stats)
        p_T  = denorm_duration(pred_cpu[b, :n, 3], norm_stats)
        g_T  = denorm_duration(gt_cpu[b,   :n, 3], norm_stats)
        all_dt_mae.append(float(np.abs(p_dt.numpy()[r] - g_dt.numpy()[c]).mean()))
        all_T_mae.append( float(np.abs(p_T.numpy()[r]  - g_T.numpy()[c]).mean()))

    kld = compute_kld(pred_px, gt_px, lengths)
    mm  = compute_multimatch(pred_px, gt_px, pred_seq, gt_seq,
                              lengths, norm_stats, width, height)
    return {
        'mean_pixel_dist': float(np.mean(all_dists)),
        'center_loss':     float(np.mean(all_center)),
        'spread_loss':     float(np.mean(all_spread)),
        'delta_t_mae_ms':  float(np.mean(all_dt_mae)),
        'T_mae_ms':        float(np.mean(all_T_mae)),
        **kld, **mm,
    }


def evaluate(model, val_dataloader, device, temperature=None):
    """
    Generate predictions for one validation batch and compute all metrics.
    """
    if temperature is None:
        temperature = config.INFERENCE_TEMPERATURE

    model.eval()
    ds = val_dataloader.dataset

    batch       = next(iter(val_dataloader))
    cond_geom   = batch['cond_geom'].to(device)
    cond_signal = batch['cond_signal'].to(device)
    gt_seq      = batch['seq'].to(device)          # [B, N, 4]
    lengths     = batch['length']

    with torch.no_grad():
        pred_seq = model.generate(
            cond_geom, cond_signal,
            num_fixations=config.MAX_FIXATIONS,
            temperature=temperature,
        )   # [B, N, 4]

    pred_px = seq_to_pixels(pred_seq, ds.width, ds.height)
    gt_px   = seq_to_pixels(gt_seq,   ds.width, ds.height)

    model.train()
    return compute_hungarian_metrics(
        pred_px, gt_px, pred_seq, gt_seq,
        lengths, ds.norm_stats,
        width=ds.width, height=ds.height,
    )
