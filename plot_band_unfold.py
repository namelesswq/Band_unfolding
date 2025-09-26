#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_unfolded_with_xticks.py

Features:
 - HSP labels displayed on bottom x-axis; supports mapping k-index -> k_on_path (idx2k)
 - Automatically replaces "GAMMA" with "Γ"
 - If multiple HSP indices map to almost-identical x positions, they are merged and label shown as "A|L"
 - fatband: uniform color; alpha & size ~ weight
 - density: weighted 2D histogram (viridis default)
 - supports --ylim, --hsp, --ef (Fermi level), and other options
 - NEW: --vmin/--vmax for density colormap limits, --smooth-sigma for optional Gaussian smoothing
"""
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
import re
from typing import Tuple, Dict, List, Optional

# -------------------------
# simple gaussian smoothing (use scipy if available, else numpy separable conv)
# -------------------------
def gaussian_smooth(arr: np.ndarray, sigma: float) -> np.ndarray:
    """Return Gaussian-smoothed copy of 2D array arr with given sigma (in pixels).
    If sigma <= 0, return arr unchanged.
    Uses scipy.ndimage.gaussian_filter if available; otherwise uses separable numpy conv.
    """
    if sigma is None or sigma <= 0.0:
        return arr
    try:
        from scipy.ndimage import gaussian_filter
        return gaussian_filter(arr, sigma=sigma)
    except Exception:
        # Pure numpy separable convolution
        # build 1D kernel truncated at 3 sigma each side (size = 6*sigma rounded up, odd)
        sigma = float(sigma)
        radius = max(1, int(np.ceil(3.0 * sigma)))
        x = np.arange(-radius, radius + 1)
        kernel = np.exp(-0.5 * (x / sigma) ** 2)
        kernel /= kernel.sum()
        # convolve rows then cols
        # pad array using reflect to reduce border effects
        padded = np.pad(arr, ((radius, radius), (radius, radius)), mode='reflect')
        # convolve along axis 1 (cols) for each row
        tmp = np.empty_like(padded, dtype=float)
        for i in range(padded.shape[0]):
            tmp[i, :] = np.convolve(padded[i, :], kernel, mode='same')
        # then convolve along axis 0 (rows)
        out_padded = np.empty_like(tmp, dtype=float)
        for j in range(tmp.shape[1]):
            out_padded[:, j] = np.convolve(tmp[:, j], kernel, mode='same')
        # crop back
        out = out_padded[radius:-radius, radius:-radius]
        return out

# -------------------------
# IO parser: build idx2k mapping if possible
# -------------------------
def read_unfolded_file(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray], Dict[int, float]]:
    xs = []
    ys = []
    ws = []
    kidxs_list = []
    idx2k = {}
    with open(path, 'r', encoding='utf-8') as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith('#'):
                continue
            toks = line.split()
            nums = []
            for t in toks:
                try:
                    nums.append(float(t))
                except:
                    pass
            if len(nums) < 3:
                continue
            # detect possible k-index in first numeric token
            has_kidx = False
            kidx_val = None
            if len(nums) >= 4:
                f0 = nums[0]
                if abs(f0 - round(f0)) < 1e-8:
                    has_kidx = True
                    kidx_val = int(round(f0))
            # interpret k,e,w (common: k = nums[-3], e = nums[-2], w = nums[-1])
            k = nums[-3]; e = nums[-2]; w = nums[-1]
            xs.append(k); ys.append(e); ws.append(w)
            if has_kidx:
                kidxs_list.append(kidx_val)
                if kidx_val not in idx2k:
                    idx2k[kidx_val] = float(k)
            else:
                kidxs_list.append(None)
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    ws = np.asarray(ws, dtype=float)
    if xs.size == 0:
        return xs, ys, ws, None, {}
    if np.allclose(ws, 0.0):
        ws = np.ones_like(ws) * 1e-6
    if any(item is not None for item in kidxs_list):
        kidxs = np.array([int(x) if x is not None else -1 for x in kidxs_list], dtype=int)
    else:
        kidxs = None
    return xs, ys, ws, kidxs, idx2k

# -------------------------
# parse hsp arg "GAMMA:0,X:49"
# -------------------------
def parse_hsp_arg(hsp_arg: Optional[str]) -> List[Tuple[str,int]]:
    if not hsp_arg:
        return []
    pairs = []
    for token in re.split(r'[,\s]+', hsp_arg.strip()):
        if token == '':
            continue
        if ':' not in token:
            continue
        lab, sidx = token.split(':', 1)
        lab = lab.strip()
        try:
            idx = int(sidx)
        except:
            continue
        pairs.append((lab, idx))
    return pairs

# -------------------------
# normalize label text (GAMMA -> Γ)
# -------------------------
def normalize_label(lab: str) -> str:
    if lab.upper() == "GAMMA":
        return "Γ"
    return lab

# -------------------------
# helper to compute HSP xtick positions & labels with merging
# -------------------------
def compute_hsp_xticks(hsp_list: List[Tuple[str,int]], xs_orig: np.ndarray, idx2k: Dict[int,float], merge_tol: Optional[float]=None) -> Tuple[List[float], List[str]]:
    raw = []
    for lab, idx in hsp_list:
        if idx2k and (idx in idx2k):
            xpos = float(idx2k[idx])
        else:
            if idx < 0 or idx >= len(xs_orig):
                print(f"[WARN] HSP index {idx} not found in idx2k and out of row-index range. Skipping {lab}:{idx}")
                continue
            xpos = float(xs_orig[idx])
        raw.append((xpos, normalize_label(lab)))

    if not raw:
        return [], []

    raw_sorted = sorted(raw, key=lambda t: t[0])
    xs_min = float(np.min(xs_orig)); xs_max = float(np.max(xs_orig))
    data_range = max(1e-8, xs_max - xs_min)
    if merge_tol is None:
        merge_tol = data_range * 1e-5

    merged_positions = []
    merged_labels = []

    cur_x, cur_labels = raw_sorted[0][0], [raw_sorted[0][1]]
    for xpos, lab in raw_sorted[1:]:
        if abs(xpos - cur_x) <= merge_tol:
            if lab not in cur_labels:
                cur_labels.append(lab)
        else:
            merged_positions.append(cur_x)
            merged_labels.append("|".join(cur_labels))
            cur_x = xpos
            cur_labels = [lab]
    merged_positions.append(cur_x)
    merged_labels.append("|".join(cur_labels))

    return merged_positions, merged_labels

# -------------------------
# fatband plotting
# -------------------------
def plot_fatband_fixedcolor(xs_sorted, ys_sorted, ws_sorted,
                            xs_orig, ys_orig, ws_orig,
                            idx2k: Dict[int,float],
                            hsp_list: List[Tuple[str,int]],
                            color='C0',
                            size_min=4.0, size_max=120.0,
                            alpha_min=0.06, alpha_max=1.0,
                            y_lim=None,
                            ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(8,6))
    else:
        fig = ax.figure

    w = ws_sorted.astype(float).copy()
    wmin = float(np.nanmin(w)); wmax = float(np.nanmax(w))
    wnorm = (w - wmin) / (wmax - wmin) if wmax > wmin else np.ones_like(w) * 0.5
    alphas = alpha_min + (alpha_max - alpha_min) * wnorm
    sizes = size_min + (size_max - size_min) * wnorm

    base_rgba = mcolors.to_rgba(color)
    colors = np.tile(base_rgba, (len(wnorm), 1))
    colors[:, 3] = alphas

    ax.scatter(xs_sorted, ys_sorted, s=sizes, c=colors, marker='o', linewidths=0, rasterized=True)

    if hsp_list:
        positions, labels = compute_hsp_xticks(hsp_list, xs_orig, idx2k)
        if positions:
            ax.set_xticks(positions)
            ax.set_xticklabels(labels, fontsize=9)

    ax.set_xlabel('k (path coord)')
    ax.set_ylabel('Energy')
    ax.set_title('Fatband (uniform color; alpha & size ~ weight)')
    ax.grid(True, linestyle=':', alpha=0.4)

    if y_lim is not None:
        ax.set_ylim(y_lim)

    xmin = float(np.nanmin(xs_sorted)); xmax = float(np.nanmax(xs_sorted))
    ax.set_xlim(xmin, xmax)
    ax.margins(x=0)

    ax.xaxis.set_ticks_position('bottom')
    ax.xaxis.set_label_position('bottom')

    return fig, ax

# -------------------------
# density plotting
# -------------------------
def plot_density(xs_sorted, ys_sorted, ws_sorted,
                 xs_orig, idx2k: Dict[int,float], hsp_list: List[Tuple[str,int]],
                 bins_k=300, bins_e=400, log_scale=False, cmap='viridis', y_lim=None,
                 vmin: Optional[float]=None, vmax: Optional[float]=None, smooth_sigma: float=0.0, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(8,6))
    else:
        fig = ax.figure

    nk = int(bins_k); ne = int(bins_e)
    kmin, kmax = float(np.nanmin(xs_sorted)), float(np.nanmax(xs_sorted))
    emin, emax = float(np.nanmin(ys_sorted)), float(np.nanmax(ys_sorted))

    if y_lim is not None:
        emin, emax = float(y_lim[0]), float(y_lim[1])

    km = 1e-8 * max(1.0, abs(kmax - kmin))
    em = 1e-8 * max(1.0, abs(emax - emin))
    extent = [kmin - km, kmax + km, emin - em, emax + em]

    H, k_edges, e_edges = np.histogram2d(xs_sorted, ys_sorted, bins=(nk, ne),
                                         range=[[extent[0], extent[1]], [extent[2], extent[3]]],
                                         weights=ws_sorted)
    disp = np.log1p(H) if log_scale else H

    # optional smoothing (on disp)
    if smooth_sigma and smooth_sigma > 0.0:
        disp = gaussian_smooth(disp, smooth_sigma)

    im = ax.imshow(disp.T, extent=extent, aspect='auto', origin='lower', cmap=cmap, vmin=vmin, vmax=vmax)
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Weighted density')

    if hsp_list:
        positions, labels = compute_hsp_xticks(hsp_list, xs_orig, idx2k)
        if positions:
            ax.set_xticks(positions)
            ax.set_xticklabels(labels, fontsize=9)

    ax.set_xlabel('k (path coord)')
    ax.set_ylabel('Energy')
    ax.set_title('Density (weighted)')
    if y_lim is not None:
        ax.set_ylim(y_lim)

    ax.set_xlim(kmin, kmax)
    ax.margins(x=0)

    ax.xaxis.set_ticks_position('bottom')
    ax.xaxis.set_label_position('bottom')

    return fig, ax, (k_edges, e_edges, H, disp)

# -------------------------
# CLI & main
# -------------------------
def main():
    p = argparse.ArgumentParser(description='Plot unfolded band with merged HSP labels as x-axis ticks.')
    p.add_argument('file', help='input data file')
    p.add_argument('--mode', choices=['fatband','density'], default='fatband')
    p.add_argument('--out', '-o', default=None, help='output image file')
    p.add_argument('--color', default='C0', help='base color for fatband')
    p.add_argument('--size-min', type=float, default=4.0)
    p.add_argument('--size-max', type=float, default=120.0)
    p.add_argument('--alpha-min', type=float, default=0.06)
    p.add_argument('--alpha-max', type=float, default=1.0)
    p.add_argument('--ylim', nargs=2, type=float, metavar=('YMIN','YMAX'), help='y-axis limits (energy)')
    p.add_argument('--bins-k', type=int, default=300)
    p.add_argument('--bins-e', type=int, default=400)
    p.add_argument('--log', action='store_true', help='log1p display for density')
    p.add_argument('--cmap', default='viridis')
    p.add_argument('--vmin', type=float, default=None, help='vmin for density colormap')
    p.add_argument('--vmax', type=float, default=None, help='vmax for density colormap')
    p.add_argument('--smooth-sigma', type=float, default=0.0, help='Gaussian smoothing sigma (pixels) for density')
    p.add_argument('--hsp', type=str, default=None, help='HSP list: "GAMMA:0,X:49,..." (indices are k-index if present, else 0-based row-index)')
    p.add_argument('--ef', '--fermi', dest='ef', type=float, default=None, help='Fermi energy (draw horizontal dashed line at this energy)')
    args = p.parse_args()

    infile = Path(args.file)
    xs_orig, ys_orig, ws_orig, kidxs, idx2k = read_unfolded_file(infile)
    if xs_orig.size == 0:
        raise RuntimeError("No numeric data parsed from file.")

    order = np.lexsort((ys_orig, xs_orig))
    xs_sorted = xs_orig[order]; ys_sorted = ys_orig[order]; ws_sorted = ws_orig[order]

    hsp_list = parse_hsp_arg(args.hsp)
    # default placeholder hsp_list can be set by user; comment out the hard-coded list
    hsp_list = [("GAMMA", 0), ("X", 49), ("K", 99), ("GAMMA", 149),
            ("A", 199), ("L", 249), ("H", 299), ("A", 349),
            ("L", 350), ("M", 399), ("H", 400), ("K", 449)]

    y_lim = tuple(args.ylim) if args.ylim else None

    if args.mode == 'fatband':
        fig, ax = plot_fatband_fixedcolor(xs_sorted, ys_sorted, ws_sorted,
                                          xs_orig, ys_orig, ws_orig,
                                          idx2k, hsp_list,
                                          color=args.color,
                                          size_min=args.size_min, size_max=args.size_max,
                                          alpha_min=args.alpha_min, alpha_max=args.alpha_max,
                                          y_lim=y_lim)
    else:
        fig, ax, _ = plot_density(xs_sorted, ys_sorted, ws_sorted,
                                  xs_orig, idx2k, hsp_list,
                                  bins_k=args.bins_k, bins_e=args.bins_e,
                                  log_scale=args.log, cmap=args.cmap,
                                  y_lim=y_lim, vmin=args.vmin, vmax=args.vmax,
                                  smooth_sigma=args.smooth_sigma)

    # draw Fermi level line if requested
    if args.ef is not None:
        ef = float(args.ef)
        ax.axhline(y=ef, color='k', linestyle='--', linewidth=1.0, alpha=0.9)
        trans = ax.get_yaxis_transform()
        ax.text(1.01, ef, f'E_F={ef}', transform=trans, ha='left', va='center', fontsize=9, clip_on=False)

    plt.tight_layout(pad=0.5)
    if args.out:
        fig.savefig(args.out, dpi=300, bbox_inches='tight')
        print(f"[INFO] Saved to {args.out}")
    else:
        plt.show()

if __name__ == '__main__':
    main()
