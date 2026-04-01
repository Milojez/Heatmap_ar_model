"""
Heatmap utilities: target generation and sampling.
"""

import torch
import torch.nn.functional as F


def make_batch_heatmaps(xy_norm, hm_w, hm_h, sigma_cells=2.5):
    """
    Build Gaussian-smoothed target probability heatmaps for a batch of fixations.

    xy_norm : [B, N, 2]  normalised coords in [-1, 1]  (x_norm, y_norm)
    hm_w    : int        heatmap width  (columns)
    hm_h    : int        heatmap height (rows)
    sigma_cells : float  Gaussian sigma in heatmap cell units

    Returns : [B, N, hm_h, hm_w]  probability maps (sum to 1 over spatial dims)
    """
    B, N, _ = xy_norm.shape
    device  = xy_norm.device

    # Convert [-1,1] to heatmap cell coordinates
    cx = (xy_norm[:, :, 0] + 1) / 2 * hm_w   # [B, N]  in [0, hm_w]
    cy = (xy_norm[:, :, 1] + 1) / 2 * hm_h   # [B, N]  in [0, hm_h]

    xs = torch.arange(hm_w, device=device, dtype=torch.float32)
    ys = torch.arange(hm_h, device=device, dtype=torch.float32)
    yy, xx = torch.meshgrid(ys, xs, indexing='ij')   # [hm_h, hm_w]

    cx = cx.unsqueeze(-1).unsqueeze(-1)   # [B, N, 1, 1]
    cy = cy.unsqueeze(-1).unsqueeze(-1)   # [B, N, 1, 1]

    heatmaps = torch.exp(
        -((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma_cells ** 2)
    )   # [B, N, hm_h, hm_w]

    # Normalise to probability distribution
    heatmaps = heatmaps / heatmaps.sum(dim=(-2, -1), keepdim=True).clamp(min=1e-8)
    return heatmaps


def sample_from_heatmap(probs, hm_w, hm_h, img_w, img_h):
    """
    Categorical sample from a batch of probability heatmaps.
    Adds uniform sub-cell noise so the result is a continuous coordinate.

    probs  : [B, hm_h * hm_w]  flattened probability map (sums to 1)
    Returns: xy_norm [B, 2]   normalised coords in [-1, 1]
             xy_px   [B, 2]   pixel coords
    """
    B      = probs.shape[0]
    device = probs.device

    # Sample one cell per batch element
    idx = torch.multinomial(probs, num_samples=1).squeeze(-1)   # [B]
    row = idx // hm_w                                            # [B]
    col = idx % hm_w                                             # [B]

    # Cell centre + uniform noise within the cell for sub-cell precision
    cell_w = img_w / hm_w
    cell_h = img_h / hm_h

    x_px = (col.float() + torch.rand(B, device=device)) * cell_w   # [B]
    y_px = (row.float() + torch.rand(B, device=device)) * cell_h   # [B]

    x_norm = x_px / img_w * 2 - 1
    y_norm = y_px / img_h * 2 - 1

    return (torch.stack([x_norm, y_norm], dim=-1),
            torch.stack([x_px,   y_px],   dim=-1))
