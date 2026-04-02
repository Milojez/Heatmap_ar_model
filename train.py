"""
Training loop for the Heatmap Autoregressive model.

Loss:
  L = soft_cross_entropy(hm_logits, target_heatmap)
    + LAMBDA_TEMPORAL * MSE(temporal_pred, temporal_gt)
  where soft_cross_entropy = -(target_heatmap * log_softmax(hm_logits)).sum(dim=-1)
  averaged over valid (non-padded) fixation steps.
"""

import os
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
import sys
import csv
import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

import config
from dataset import ScanpathDataset
from model import HeatmapARModel
from metrics import evaluate
from utils.heatmap import make_batch_heatmaps

sys.path.insert(0, os.path.dirname(__file__))


# ── Loss ──────────────────────────────────────────────────────────────────────

def soft_ce_loss(hm_logits, target_hm, padding_mask):
    """
    hm_logits  : [B, N, HM_H*HM_W]   raw logits
    target_hm  : [B, N, HM_H*HM_W]   Gaussian-smoothed probability map (sums to 1)
    padding_mask: [B, N] bool         True = padded (excluded from loss)

    Returns scalar mean loss over valid steps.
    """
    log_probs = F.log_softmax(hm_logits, dim=-1)             # [B, N, cells]
    ce        = -(target_hm * log_probs).sum(dim=-1)         # [B, N]
    valid     = ~padding_mask                                 # [B, N]
    return (ce * valid).sum() / valid.sum().clamp(min=1)


def temporal_mse_loss(temporal_pred, temporal_gt, padding_mask):
    """
    temporal_pred / temporal_gt : [B, N, 2]  [dt_norm, T_norm]
    """
    diff  = (temporal_pred - temporal_gt) ** 2               # [B, N, 2]
    valid = (~padding_mask).unsqueeze(-1)                     # [B, N, 1]
    return (diff * valid).sum() / (valid.sum() * 2).clamp(min=1)


# ── Learning-rate schedule ────────────────────────────────────────────────────

def get_lr(epoch, base_lr, lr_min, warmup_epochs, total_epochs):
    if epoch < warmup_epochs:
        return base_lr * (epoch + 1) / warmup_epochs
    t_restart = config.LR_RESTART_EVERY
    if t_restart > 0:
        epoch_in_cycle = (epoch - warmup_epochs) % t_restart
        progress = epoch_in_cycle / t_restart
    else:
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
    return lr_min + (base_lr - lr_min) * 0.5 * (1 + math.cos(math.pi * progress))


# ── Main training ─────────────────────────────────────────────────────────────

def main():
    torch.manual_seed(config.SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Directories
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(config.VIS_DIR,        exist_ok=True)
    os.makedirs(config.PRED_DIR,       exist_ok=True)

    # Datasets
    train_ds = ScanpathDataset(
        config.TRAIN_JSON,
        width=config.IMAGE_WIDTH, height=config.IMAGE_HEIGHT,
        max_fixations=config.MAX_FIXATIONS,
        max_samples=config.MAX_TRAIN_SAMPLES,
        cond_noise_std=config.COND_SIGNAL_NOISE_STD,
    )
    val_ds = ScanpathDataset(
        config.VAL_JSON,
        width=config.IMAGE_WIDTH, height=config.IMAGE_HEIGHT,
        max_fixations=config.MAX_FIXATIONS,
        max_samples=config.MAX_VAL_SAMPLES,
        norm_stats=train_ds.norm_stats,
        cond_norm_stats=train_ds.cond_norm_stats,
    )
    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE,
                              shuffle=True, drop_last=True, num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=config.VAL_BATCH_SIZE,
                              shuffle=False, drop_last=False, num_workers=0)

    print(f"Train: {len(train_ds)} samples | Val: {len(val_ds)} samples")

    # Model
    model = HeatmapARModel().to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=config.LEARNING_RATE,
                                  weight_decay=config.WEIGHT_DECAY)

    # Resume if checkpoint exists
    start_epoch = 0
    best_dist   = float('inf')
    if os.path.exists(config.CHECKPOINT_PATH):
        try:
            ckpt = torch.load(config.CHECKPOINT_PATH, map_location=device)
            model.load_state_dict(ckpt['model'])
            optimizer.load_state_dict(ckpt['optimizer'])
            start_epoch = ckpt.get('epoch', 0) + 1
            best_dist   = ckpt.get('best_dist', float('inf'))
            print(f"Resumed from epoch {start_epoch - 1}")
        except Exception as e:
            print(f"[WARNING] Checkpoint corrupted, starting from scratch: {e}")
            os.remove(config.CHECKPOINT_PATH)

    # Logging
    log_exists = os.path.exists(config.LOG_PATH)
    log_file   = open(config.LOG_PATH, 'a', newline='')
    log_writer = csv.writer(log_file)
    if not log_exists:
        log_writer.writerow([
            'epoch', 'train_loss', 'train_hm_loss', 'train_temp_loss',
            'val_dist', 'val_kld_mean',
            'val_mm_shape', 'val_mm_direction', 'val_mm_length',
            'val_mm_position', 'val_mm_duration',
        ])

    train_losses, val_dists = [], []

    for epoch in range(start_epoch, config.EPOCHS):
        t0 = time.time()

        # Periodically release fragmented CUDA cache
        if device.type == 'cuda' and epoch % 25 == 0:
            torch.cuda.empty_cache()

        # ── LR schedule ────────────────────────────────────────────────────
        lr = get_lr(epoch, config.LEARNING_RATE, config.LR_MIN,
                    config.WARMUP_EPOCHS, config.EPOCHS)
        for pg in optimizer.param_groups:
            pg['lr'] = lr

        # ── Training epoch ─────────────────────────────────────────────────
        model.train()
        epoch_loss = 0.0
        epoch_hm   = 0.0
        epoch_temp = 0.0
        n_batches  = 0

        optimizer.zero_grad()

        for step, batch in enumerate(train_loader):
            seq          = batch['seq'].to(device)           # [B, N, 4]
            cond_geom    = batch['cond_geom'].to(device)
            cond_signal  = batch['cond_signal'].to(device)
            padding_mask = batch['padding_mask'].to(device)  # [B, N]

            B, N, _ = seq.shape

            # Build Gaussian-smoothed target heatmaps from true (x,y)
            xy_norm = seq[:, :, :2]                          # [B, N, 2]
            target_hm = make_batch_heatmaps(
                xy_norm, config.HM_W, config.HM_H,
                sigma_cells=config.HM_SIGMA_CELLS,
            )                                                # [B, N, HM_H, HM_W]
            target_hm = target_hm.reshape(B, N, -1)         # [B, N, HM_H*HM_W]

            # Forward
            hm_logits, temporal_pred = model(seq, cond_geom, cond_signal)

            # Losses
            loss_hm   = soft_ce_loss(hm_logits, target_hm, padding_mask)
            loss_temp = temporal_mse_loss(temporal_pred, seq[:, :, 2:], padding_mask)
            loss      = loss_hm + config.LAMBDA_TEMPORAL * loss_temp

            # Gradient accumulation
            (loss / config.ACCUM_STEPS).backward()
            if (step + 1) % config.ACCUM_STEPS == 0:
                nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP_NORM)
                optimizer.step()
                optimizer.zero_grad()

            epoch_loss += loss.item()
            epoch_hm   += loss_hm.item()
            epoch_temp += loss_temp.item()
            n_batches  += 1

        epoch_loss /= max(n_batches, 1)
        epoch_hm   /= max(n_batches, 1)
        epoch_temp /= max(n_batches, 1)
        train_losses.append(epoch_loss)

        elapsed = time.time() - t0
        print(f"Epoch {epoch:4d}/{config.EPOCHS}  "
              f"loss={epoch_loss:.4f} (hm={epoch_hm:.4f} t={epoch_temp:.4f})  "
              f"lr={lr:.2e}  {elapsed:.1f}s", end='')

        # ── Validation ─────────────────────────────────────────────────────
        val_metrics = {}
        if (epoch + 1) % config.EVAL_EVERY == 0 or epoch == config.EPOCHS - 1:
            val_metrics = evaluate(model, val_loader, device,
                                   temperature=config.INFERENCE_TEMPERATURE)
            dist = val_metrics.get('mean_pixel_dist', float('inf'))
            val_dists.append(dist)
            print(f"  | dist={dist:.1f}  "
                  f"kld={val_metrics.get('kld_mean', float('nan')):.3f}  "
                  f"mm_shape={val_metrics.get('mm_shape', float('nan')):.3f}")

            # Save best
            if dist < best_dist:
                best_dist = dist
                _safe_save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                            'epoch': epoch, 'best_dist': best_dist},
                           config.BEST_CHECKPOINT_PATH)
        else:
            print()

        # Save last
        _safe_save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                    'epoch': epoch, 'best_dist': best_dist},
                   config.CHECKPOINT_PATH)

        if config.SAVE_EVERY > 0 and (epoch + 1) % config.SAVE_EVERY == 0:
            ep_path = config.CHECKPOINT_PATH.replace('_last.pth', f'_ep{epoch:04d}.pth')
            _safe_save({'model': model.state_dict(), 'epoch': epoch}, ep_path)

        log_writer.writerow([
            epoch, epoch_loss, epoch_hm, epoch_temp,
            val_metrics.get('mean_pixel_dist', ''),
            val_metrics.get('kld_mean', ''),
            val_metrics.get('mm_shape', ''),
            val_metrics.get('mm_direction', ''),
            val_metrics.get('mm_length', ''),
            val_metrics.get('mm_position', ''),
            val_metrics.get('mm_duration', ''),
        ])
        log_file.flush()

    log_file.close()
    _plot_metrics(train_losses, val_dists)
    print("Training complete.")


def _safe_save(obj, path):
    """torch.save wrapped in try/except so a disk error doesn't crash training."""
    tmp = path + '.tmp'
    try:
        torch.save(obj, tmp)
        os.replace(tmp, path)
    except Exception as e:
        print(f"\n  [WARNING] Checkpoint save failed ({path}): {e}")
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass


def _plot_metrics(train_losses, val_dists):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(train_losses)
    axes[0].set_title('Train Loss')
    axes[0].set_xlabel('Epoch')

    if val_dists:
        xs = [(i + 1) * config.EVAL_EVERY for i in range(len(val_dists))]
        axes[1].plot(xs, val_dists)
        axes[1].set_title('Val Mean Pixel Dist')
        axes[1].set_xlabel('Epoch')

    plt.tight_layout()
    plt.savefig(config.PLOT_PATH)
    plt.close()


if __name__ == '__main__':
    main()
