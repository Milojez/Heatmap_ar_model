"""
Produces a clean architecture diagram for HeatmapARModel.
Run:  python plot_architecture.py
Saves: architecture_hm_ar.png  (next to this script)
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import os

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "architecture_hm_ar.png")

# ── Colour palette ─────────────────────────────────────────────────────────────
C_INPUT  = "#dbeafe"   # light blue  — inputs
C_ENC    = "#dcfce7"   # light green — conditioning encoder
C_DEC    = "#fef9c3"   # light yellow — GRU decoder
C_ATTN   = "#fce7f3"   # light pink  — cross-attention
C_HEAD   = "#ede9fe"   # light purple — output heads
C_OUT    = "#ffedd5"   # light orange — outputs
C_BORDER = "#374151"

def box(ax, x, y, w, h, label, sublabel=None, color="#ffffff", fontsize=9):
    rect = FancyBboxPatch((x, y), w, h,
                          boxstyle="round,pad=0.02",
                          linewidth=1.2, edgecolor=C_BORDER,
                          facecolor=color, zorder=3)
    ax.add_patch(rect)
    cy = y + h / 2
    if sublabel:
        ax.text(x + w/2, cy + 0.08, label,
                ha="center", va="center", fontsize=fontsize,
                fontweight="bold", zorder=4, color="#111827")
        ax.text(x + w/2, cy - 0.12, sublabel,
                ha="center", va="center", fontsize=fontsize - 1.5,
                zorder=4, color="#374151", style="italic")
    else:
        ax.text(x + w/2, cy, label,
                ha="center", va="center", fontsize=fontsize,
                fontweight="bold", zorder=4, color="#111827")

def arrow(ax, x0, y0, x1, y1, label=None, color="#374151"):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", color=color,
                                lw=1.4, mutation_scale=12),
                zorder=2)
    if label:
        mx, my = (x0+x1)/2, (y0+y1)/2
        ax.text(mx + 0.04, my, label, fontsize=7.5, color="#6b7280",
                va="center", zorder=5)

def bracket(ax, x, y_bot, y_top, label, color="#6b7280"):
    ax.annotate("", xy=(x, y_top), xytext=(x, y_bot),
                arrowprops=dict(arrowstyle="-", color=color, lw=1.0, linestyle="dashed"))
    ax.text(x + 0.04, (y_bot + y_top) / 2, label,
            fontsize=7.5, color=color, va="center")


fig, ax = plt.subplots(figsize=(13, 16))
ax.set_xlim(0, 6)
ax.set_ylim(0, 16)
ax.axis("off")
fig.patch.set_facecolor("#f9fafb")

# ── Title ──────────────────────────────────────────────────────────────────────
ax.text(3, 15.6, "HeatmapARModel — Architecture",
        ha="center", va="center", fontsize=13, fontweight="bold", color="#111827")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — INPUTS
# ══════════════════════════════════════════════════════════════════════════════
ax.text(0.15, 15.1, "INPUTS", fontsize=8, color="#6b7280", fontweight="bold")

box(ax, 0.2, 14.4, 2.4, 0.55, "cond_geom  [B, 6, 4]",
    "dial geometry: cx, cy, tx, ty", C_INPUT)
box(ax, 2.9, 14.4, 2.9, 0.55, "cond_signal  [B, 6, 10, 2]",
    "per frame: urgency + distance", C_INPUT)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — CONDITIONING ENCODER
# ══════════════════════════════════════════════════════════════════════════════
ax.text(0.15, 14.0, "CONDITIONING ENCODER  (runs once)", fontsize=8,
        color="#166534", fontweight="bold")

# Three projection boxes side by side
box(ax, 0.2,  13.2, 1.6, 0.6, "space_proj",   "Linear(4→256)", C_ENC)
box(ax, 2.0,  13.2, 1.6, 0.6, "signal_proj",  "Linear(2→256)", C_ENC)
box(ax, 3.8,  13.2, 1.6, 0.6, "time_emb",     "Embed(10,256)", C_ENC)

# Arrows from inputs to projections
arrow(ax, 1.40, 14.40, 1.00, 13.80)   # geom → space_proj
arrow(ax, 3.35, 14.40, 2.80, 13.80)   # signal → signal_proj
arrow(ax, 3.35, 14.40, 4.60, 13.80)   # signal (frame idx) → time_emb

# Sum node
box(ax, 2.2, 12.4, 1.4, 0.55, "element-wise sum", "[B, 60, 256]", "#f0fdf4")
arrow(ax, 1.00, 13.20, 2.60, 12.95)
arrow(ax, 2.80, 13.20, 2.90, 12.95)
arrow(ax, 4.60, 13.20, 3.20, 12.95)

# Transformer encoder
box(ax, 1.5, 11.3, 2.8, 0.85,
    "TransformerEncoder  ×3 blocks",
    "self-attn (4 heads, bidirectional) + FFN(256→1024→256)", C_ENC, fontsize=8.5)
arrow(ax, 2.90, 12.40, 2.90, 12.15)
ax.text(3.35, 11.85, "[B, 60, 256]", fontsize=7.5, color="#6b7280", va="center")

# Memory output
box(ax, 2.0, 10.6, 1.8, 0.55, "memory  [B, 60, 256]", color="#bbf7d0")
arrow(ax, 2.90, 11.30, 2.90, 11.15)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — GRU DECODER
# ══════════════════════════════════════════════════════════════════════════════
ax.text(0.15, 10.2, "GRU DECODER  (autoregressive, step i)", fontsize=8,
        color="#854d0e", fontweight="bold")

box(ax, 0.2,  9.4, 1.7, 0.6, "fix_proj",  "Linear(2→256)", C_DEC)
box(ax, 2.1,  9.4, 1.7, 0.6, "step_emb", "Embed(12,256)", C_DEC)

# Labels for inputs to decoder
ax.text(1.05, 9.15, "prev (x,y)", fontsize=7.5, ha="center", color="#92400e")
ax.text(2.95, 9.15, "step index i", fontsize=7.5, ha="center", color="#92400e")

arrow(ax, 1.05, 9.40, 1.05, 9.20,  color="#92400e")
arrow(ax, 2.95, 9.40, 2.95, 9.20,  color="#92400e")

# START token note
ax.text(0.2, 9.08, "step 0: START token (learnable)", fontsize=7, color="#6b7280", style="italic")

# Sum
box(ax, 1.5, 8.55, 1.8, 0.6, "sum  [B, 256]", color="#fef9c3")
arrow(ax, 1.05, 9.40, 1.90, 9.15)
arrow(ax, 2.95, 9.40, 2.50, 9.15)

# GRU
box(ax, 1.5, 7.65, 1.8, 0.7,
    "GRU  (1 layer, 256)",
    "reset h every 1 step", C_DEC)
arrow(ax, 2.40, 8.55, 2.40, 8.35)
ax.text(3.35, 7.95, "GRU_HISTORY=1", fontsize=7.5, color="#6b7280", va="center", style="italic")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — CROSS-ATTENTION
# ══════════════════════════════════════════════════════════════════════════════
ax.text(0.15, 7.35, "CROSS-ATTENTION", fontsize=8, color="#9d174d", fontweight="bold")

box(ax, 1.5, 6.45, 1.8, 0.7,
    "MultiheadAttention",
    "Q=GRU out  K/V=memory\n4 heads + residual + LN", C_ATTN, fontsize=8)
arrow(ax, 2.40, 7.65, 2.40, 7.15)

# Memory arrow into cross-attn (from side)
ax.annotate("", xy=(1.5, 6.80), xytext=(0.7, 6.80),
            arrowprops=dict(arrowstyle="-|>", color="#9d174d", lw=1.2, mutation_scale=11))
ax.annotate("", xy=(0.7, 6.80), xytext=(0.7, 10.88),
            arrowprops=dict(arrowstyle="-", color="#9d174d", lw=1.2, linestyle="dashed"))
ax.annotate("", xy=(0.7, 10.88), xytext=(2.0, 10.88),
            arrowprops=dict(arrowstyle="-", color="#9d174d", lw=1.2, linestyle="dashed"))
ax.text(0.3, 8.8, "memory\n[B,60,256]", fontsize=7, color="#9d174d", ha="center",
        va="center", style="italic")

# ctx output
box(ax, 1.8, 5.7, 1.2, 0.55, "ctx  [B, 256]", color="#fce7f3")
arrow(ax, 2.40, 6.45, 2.40, 6.25)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — OUTPUT HEADS
# ══════════════════════════════════════════════════════════════════════════════
ax.text(0.15, 5.4, "OUTPUT HEADS", fontsize=8, color="#4c1d95", fontweight="bold")

# Split arrow
arrow(ax, 2.40, 5.70, 1.30, 5.10)
arrow(ax, 2.40, 5.70, 3.70, 5.10)

# Heatmap head
box(ax, 0.2, 3.9, 2.1, 1.05,
    "Heatmap Head",
    "LN → Linear(256→256) → GELU\n→ Linear(256→4800)\n→ softmax → sample", C_HEAD, fontsize=8)
arrow(ax, 1.30, 5.10, 1.30, 4.95)

# Temporal head
box(ax, 2.8, 3.9, 2.1, 1.05,
    "Temporal Head",
    "LN → Linear(256→128) → GELU\n→ Linear(128→2)", C_HEAD, fontsize=8)
arrow(ax, 3.70, 5.10, 3.70, 4.95)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — OUTPUTS
# ══════════════════════════════════════════════════════════════════════════════
box(ax, 0.2,  3.0, 2.1, 0.65, "x_norm, y_norm", "sampled from 96×50 heatmap", C_OUT)
box(ax, 2.8,  3.0, 2.1, 0.65, "dt_norm, T_norm", "fixation onset & duration", C_OUT)
arrow(ax, 1.30, 3.90, 1.30, 3.65)
arrow(ax, 3.70, 3.90, 3.70, 3.65)

# Combine arrow
ax.annotate("", xy=(2.40, 2.40), xytext=(1.30, 3.0),
            arrowprops=dict(arrowstyle="-|>", color=C_BORDER, lw=1.3, mutation_scale=11))
ax.annotate("", xy=(2.40, 2.40), xytext=(3.70, 3.0),
            arrowprops=dict(arrowstyle="-|>", color=C_BORDER, lw=1.3, mutation_scale=11))

box(ax, 1.5, 1.6, 1.8, 0.65,
    "Output step i:  [B, 4]",
    "x, y, dt, T  → feed x,y back", "#f1f5f9")
arrow(ax, 2.40, 2.40, 2.40, 2.25)

# Feedback arrow (x,y fed back as prev_xy)
ax.annotate("", xy=(5.5, 9.50), xytext=(5.5, 2.10),
            arrowprops=dict(arrowstyle="-", color="#2563eb", lw=1.3,
                            linestyle="dashed", mutation_scale=11))
ax.annotate("", xy=(1.90, 9.50), xytext=(5.5, 9.50),
            arrowprops=dict(arrowstyle="-|>", color="#2563eb", lw=1.3, mutation_scale=11))
ax.annotate("", xy=(5.5, 2.10), xytext=(3.30, 2.10),
            arrowprops=dict(arrowstyle="-", color="#2563eb", lw=1.3))
ax.text(5.6, 5.8, "x, y\nfed back\nas prev_xy\n(step i+1)",
        fontsize=7.5, color="#2563eb", va="center", ha="left", style="italic")

# Repeat note
ax.text(2.40, 1.35, "↻  repeat for N = 12 fixations",
        ha="center", fontsize=8, color="#6b7280", style="italic")

# ── Legend ─────────────────────────────────────────────────────────────────────
legend_items = [
    mpatches.Patch(facecolor=C_INPUT, edgecolor=C_BORDER, label="Input"),
    mpatches.Patch(facecolor=C_ENC,   edgecolor=C_BORDER, label="Conditioning encoder"),
    mpatches.Patch(facecolor=C_DEC,   edgecolor=C_BORDER, label="GRU decoder"),
    mpatches.Patch(facecolor=C_ATTN,  edgecolor=C_BORDER, label="Cross-attention"),
    mpatches.Patch(facecolor=C_HEAD,  edgecolor=C_BORDER, label="Output heads"),
    mpatches.Patch(facecolor=C_OUT,   edgecolor=C_BORDER, label="Output"),
]
ax.legend(handles=legend_items, loc="lower left", fontsize=8,
          framealpha=0.9, bbox_to_anchor=(0.0, 0.0))

plt.tight_layout()
plt.savefig(OUT, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
print(f"Saved: {OUT}")
plt.show()
