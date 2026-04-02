"""
Dataset for the Heatmap Autoregressive model.
Identical to the autoregressive_model dataset — same JSON, same normalisation.
The teacher-forcing shift and heatmap target construction are handled in train.py.
"""

import json
import math
import random
import torch
from dataclasses import dataclass
from torch.utils.data import Dataset
import config as _config


@dataclass(frozen=True)
class ScanpathNormStats:
    delta_t_log_mean:  float
    delta_t_log_std:   float
    duration_log_mean: float
    duration_log_std:  float


@dataclass(frozen=True)
class CondNormStats:
    speed_std:   float
    urgency_std: float


def fit_norm_stats(samples) -> ScanpathNormStats:
    all_dt, all_dur = [], []
    for s in samples:
        all_dt.append(torch.as_tensor(s['delta_t_start'], dtype=torch.float32))
        all_dur.append(torch.as_tensor(s['T'],            dtype=torch.float32))
    dt_log  = torch.log1p(torch.cat(all_dt))
    dur_log = torch.log1p(torch.cat(all_dur))
    eps = 1e-6
    return ScanpathNormStats(
        delta_t_log_mean  = dt_log.mean().item(),
        delta_t_log_std   = max(dt_log.std(unbiased=False).item(), eps),
        duration_log_mean = dur_log.mean().item(),
        duration_log_std  = max(dur_log.std(unbiased=False).item(), eps),
    )


def fit_cond_norm_stats(samples, urgency_eps=0.05) -> CondNormStats:
    all_speeds, all_urgency = [], []
    for s in samples:
        for dial in s['dials']:
            all_speeds.extend(dial['speed'])
            norms = dial['needle_to_threshold_norm']
            for t in range(len(norms)):
                prev = norms[t - 1] if t > 0 else (norms[1] if len(norms) > 1 else norms[0])
                rate = max(prev - norms[t], 0.0)
                all_urgency.append(rate / (norms[t] + urgency_eps))
    sp = torch.tensor(all_speeds,  dtype=torch.float32)
    ug = torch.tensor(all_urgency, dtype=torch.float32)
    eps = 1e-6
    return CondNormStats(
        speed_std   = max(sp.std(unbiased=False).item(), eps),
        urgency_std = max(ug.std(unbiased=False).item(), eps),
    )


def norm_delta_t(val, stats: ScanpathNormStats):
    return (torch.log1p(torch.tensor(val)).item() - stats.delta_t_log_mean) / stats.delta_t_log_std

def norm_duration(val, stats: ScanpathNormStats):
    return (torch.log1p(torch.tensor(val)).item() - stats.duration_log_mean) / stats.duration_log_std

def denorm_delta_t(val_norm, stats: ScanpathNormStats):
    return torch.expm1(val_norm * stats.delta_t_log_std + stats.delta_t_log_mean).clamp(min=0.0)

def denorm_duration(val_norm, stats: ScanpathNormStats):
    return torch.expm1(val_norm * stats.duration_log_std + stats.duration_log_mean).clamp(min=0.0)


class ScanpathDataset(Dataset):
    def __init__(self, json_path, width=1904, height=988, max_fixations=12,
                 max_samples=None,
                 norm_stats: ScanpathNormStats = None,
                 cond_norm_stats: CondNormStats = None,
                 cond_noise_std: float = 0.0):
        with open(json_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        if max_samples is not None:
            random.seed(_config.SEED)
            random.shuffle(self.data)
            self.data = self.data[:max_samples]
        self.width         = width
        self.height        = height
        self.max_fixations = max_fixations

        self.norm_stats      = norm_stats      if norm_stats      is not None else fit_norm_stats(self.data)
        self.cond_norm_stats = cond_norm_stats if cond_norm_stats is not None else fit_cond_norm_stats(self.data)
        self.cond_noise_std  = cond_noise_std

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item   = self.data[idx]
        stats  = self.norm_stats
        cstats = self.cond_norm_stats
        length = item['length']

        # ── Target fixation sequence [max_fixations, 4] ──────────────────────
        # Each row: [x_norm, y_norm, dt_norm, T_norm]
        fix_rows = []
        for i in range(length):
            fix_rows.append([
                item['X'][i] / self.width  * 2 - 1,
                item['Y'][i] / self.height * 2 - 1,
                norm_delta_t(item['delta_t_start'][i], stats),
                norm_duration(item['T'][i],             stats),
            ])
        while len(fix_rows) < self.max_fixations:
            fix_rows.append([0.0, 0.0, 0.0, 0.0])

        seq = torch.tensor(fix_rows, dtype=torch.float32)   # [max_fixations, 4]

        # padding_mask: True = padded position (loss not computed)
        padding_mask = torch.zeros(self.max_fixations, dtype=torch.bool)
        if length < self.max_fixations:
            padding_mask[length:] = True

        # ── Conditioning tokens ───────────────────────────────────────────────
        dials_by_pos = sorted(item['dials'], key=lambda d: d['dial_position'])
        nf       = _config.NUM_SIGNAL_FRAMES
        features = _config.SIGNAL_FEATURES
        urgency_eps = 0.05

        geom_rows   = []   # [6, 4]
        signal_rows = []   # [6, nf, SIGNAL_DIM]

        for dial in dials_by_pos:
            geom_rows.append([
                dial['center_x_px']    / self.width  * 2 - 1,
                dial['center_y_px']    / self.height * 2 - 1,
                dial['threshold_x_px'] / self.width  * 2 - 1,
                dial['threshold_y_px'] / self.height * 2 - 1,
            ])

            angles = dial['angle'][:nf]
            speeds = dial['speed'][:nf]
            norms  = dial['needle_to_threshold_norm'][:nf]
            while len(angles) < nf:
                angles = angles + [angles[-1]]
                speeds = speeds + [speeds[-1]]
                norms  = norms  + [norms[-1]]

            frame_tokens = []
            for fi in range(nf):
                row = []
                if 'sin_cos' in features:
                    row += [math.sin(angles[fi]), math.cos(angles[fi])]
                if 'speed' in features:
                    row += [math.tanh(speeds[fi] / cstats.speed_std)]
                if 'urgency' in features:
                    prev_norm = norms[fi - 1] if fi > 0 else (norms[1] if nf > 1 else norms[0])
                    raw_rate  = max(prev_norm - norms[fi], 0.0)
                    row += [math.tanh(raw_rate / (norms[fi] + urgency_eps) / cstats.urgency_std)]
                if 'distance' in features:
                    # needle_to_threshold_norm already in [0, 1]: 0 = at threshold, 1 = far away
                    # invert so that 1 = critical (at threshold), 0 = safe (far away)
                    row += [1.0 - norms[fi]]
                frame_tokens.append(row)
            signal_rows.append(frame_tokens)

        cond_geom   = torch.tensor(geom_rows,   dtype=torch.float32)   # [6, 4]
        cond_signal = torch.tensor(signal_rows, dtype=torch.float32)   # [6, nf, SIGNAL_DIM]
        if self.cond_noise_std > 0.0:
            cond_signal = cond_signal + torch.randn_like(cond_signal) * self.cond_noise_std

        return {
            'seq':          seq,           # [max_fixations, 4]  — TARGET
            'cond_geom':    cond_geom,     # [6, 4]
            'cond_signal':  cond_signal,   # [6, nf, SIGNAL_DIM]
            'padding_mask': padding_mask,  # [max_fixations] bool
            'length':       length,
            'name':         str(item['name']),
        }
