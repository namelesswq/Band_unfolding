#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_phonon_fat_density.py

读取三列文件 (kpath, freq, weight) 并绘制：
 - fatband: 统一颜色，alpha & size ~ weight
 - density: weighted 2D histogram (+ 可选平滑, vmin/vmax, log)

支持：
 --mode fatband|density
 --ylim ymin ymax
 --hsp "GAMMA:0,X:49,..." (IDX 优先使用 idx2k map，再 fallback 到行索引)
 --overlay-bands bands.out.gnu (可选)
 --ef-overlay (overlay 能量平移)
 --vmin/--vmax/--smooth-sigma/--log 等
"""
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
import re
from typing import List, Tuple, Dict, Optional

# -------------------------
# Utilities
# -------------------------
def read_threecol_file(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[int,float]]:
    """
    读入三列 (k, freq, weight)
    返回 xs, ys, ws, idx2k_map (如果文件包含 k-index -> k 的映射，这里目前只收集行索引->k)
    注意：这里假定每行至少三个数值，额外token会被忽略。
    """
    xs = []
    ys = []
    ws = []
    idx2k = {}  # 如果你在文件中有 k-index 信息，可以扩展解析逻辑写入这里
    with open(path, 'r', encoding='utf-8') as f:
        for i, raw in enumerate(f):
            line = raw.strip()
            if not line or line.startswith('#'):
                continue
            toks = line.split()
            # 尝试把前三个数读成 float
            try:
                k = float(toks[0])
                e = float(toks[1])
                w = float(toks[2]) if len(toks) > 2 else 1.0
            except Exception:
                # 忽略不能解析的行
                continue
            idx = len(xs)  # 行索引（0-based 计数有效数据行）
            xs.append(k); ys.append(e); ws.append(w)
            idx2k[idx] = k
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    ws = np.asarray(ws, dtype=float)
    if xs.size == 0:
        return xs, ys, ws, {}
    # 避免权重完全为 0
    if np.allclose(ws, 0.0):
        ws = np.ones_like(ws) * 1e-8
    return xs, ys, ws, idx2k

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

def normalize_label(lab: str) -> str:
    if lab.upper() == 'GAMMA':
        return 'Γ'
    return lab

def compute_xticks_from_hsp(hsp_list: List[Tuple[str,int]], idx2k: Dict[int,float]) -> Tuple[List[float], List[str]]:
    pos = []
    labs = []
    for lab, idx in hsp_list:
        if idx in idx2k:
            pos.append(float(idx2k[idx]))
            labs.append(normalize_label(lab))
        else:
            print(f"[WARN] HSP idx {idx} not found in idx2k; skipping {lab}:{idx}")
    return pos, labs

# gaussian smoothing (uses scipy if available; fallback to numpy separable conv)
def gaussian_smooth(arr: np.ndarray, sigma: float) -> np.ndarray:
    if sigma is None or sigma <= 0.0:
        return arr
    try:
        from scipy.ndimage import gaussian_filter
        return gaussian_filter(arr, sigma=sigma)
    except Exception:
        sigma = float(sigma)
        radius = max(1, int(np.ceil(3.0 * sigma)))
        x = np.arange(-radius, radius+1)
        kernel = np.exp(-0.5*(x/sigma)**2)
        kernel /= kernel.sum()
        padded = np.pad(arr, ((radius, radius), (radius, radius)), mode='reflect')
        tmp = np.empty_like(padded, dtype=float)
        for i in range(padded.shape[0]):
            tmp[i,:] = np.convolve(padded[i,:], kernel, mode='same')
        out_padded = np.empty_like(tmp, dtype=float)
        for j in range(tmp.shape[1]):
            out_padded[:,j] = np.convolve(tmp[:,j], kernel, mode='same')
        out = out_padded[radius:-radius, radius:-radius]
        return out

# -------------------------
# plotters
# -------------------------
def plot_fatband(xs, ys, ws, hsp_list, idx2k,
                 color='C0', size_min=6, size_max=120, alpha_min=0.08, alpha_max=1.0,
                 y_lim=None, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(8,6))
    else:
        fig = ax.figure

    # normalize weights
    w = ws.copy().astype(float)
    wmin = np.nanmin(w); wmax = np.nanmax(w)
    if wmax > wmin:
        wn = (w - wmin) / (wmax - wmin)
    else:
        wn = np.ones_like(w)*0.5
    alphas = alpha_min + (alpha_max - alpha_min)*wn
    sizes = size_min + (size_max - size_min)*wn

    base = mcolors.to_rgba(color)
    cols = np.tile(base, (len(wn),1))
    cols[:,3] = alphas

    ax.scatter(xs, ys, s=sizes, c=cols, marker='o', linewidths=0, rasterized=True)

    if hsp_list:
        positions, labels = compute_xticks_from_hsp(hsp_list, idx2k)
        if positions:
            ax.set_xticks(positions)
            ax.set_xticklabels(labels, fontsize=10)

    ax.set_xlabel('k (path coord)')
    ax.set_ylabel('Frequency')
    ax.set_title('Fatband (size & alpha ~ weight)')
    ax.grid(True, linestyle=':', alpha=0.4)
    if y_lim is not None:
        ax.set_ylim(y_lim)

    ax.set_xlim(float(np.nanmin(xs)), float(np.nanmax(xs)))
    ax.margins(x=0)
    ax.xaxis.set_ticks_position('bottom')
    return fig, ax

def plot_density(xs, ys, ws, hsp_list, idx2k,
                 bins_k=300, bins_e=400, log_scale=False, cmap='viridis',
                 vmin=None, vmax=None, smooth_sigma=0.0, y_lim=None, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(8,6))
    else:
        fig = ax.figure

    kmin, kmax = float(np.nanmin(xs)), float(np.nanmax(xs))
    emin, emax = float(np.nanmin(ys)), float(np.nanmax(ys))
    if y_lim is not None:
        emin, emax = float(y_lim[0]), float(y_lim[1])

    # small margins to include edge points
    km = 1e-8 * max(1.0, abs(kmax - kmin))
    em = 1e-8 * max(1.0, abs(emax - emin))
    extent = [kmin-km, kmax+km, emin-em, emax+em]

    H, k_edges, e_edges = np.histogram2d(xs, ys, bins=(int(bins_k), int(bins_e)),
                                         range=[[extent[0], extent[1]], [extent[2], extent[3]]],
                                         weights=ws)
    disp = np.log1p(H) if log_scale else H

    if smooth_sigma and smooth_sigma > 0.0:
        disp = gaussian_smooth(disp, smooth_sigma)

    im = ax.imshow(disp.T, extent=extent, aspect='auto', origin='lower', cmap=cmap, vmin=vmin, vmax=vmax)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('Weighted density')

    if hsp_list:
        positions, labels = compute_xticks_from_hsp(hsp_list, idx2k)
        if positions:
            ax.set_xticks(positions)
            ax.set_xticklabels(labels, fontsize=10)

    ax.set_xlabel('k (path coord)')
    ax.set_ylabel('Frequency')
    ax.set_title('Density (weighted)')
    if y_lim is not None:
        ax.set_ylim(y_lim)
    ax.set_xlim(kmin, kmax)
    ax.margins(x=0)
    ax.xaxis.set_ticks_position('bottom')
    return fig, ax, (k_edges, e_edges, H, disp)

# -------------------------
# 读取 overlay bands（每行第1列 k，其余列为多个支的频率）
# 支持用空行分段（每个段对应一个连续的 k 栏）
# 返回: list_of_segments_per_branch
#  - result[b] = [ (k_array_seg1, e_array_seg1), (k_array_seg2, e_array_seg2), ... ]
#    即每个 branch 有若干个 segment（空行分段）
# -------------------------
def read_bands_multi(path: Path):
    """
    Parse a multi-branch bands.out.gnu-like file:
    Each non-empty line: k  e1  e2  e3  ...
    Empty line: break segment (so use to separate different k-path pieces)
    Returns a list `branches`, where branches[b] is a list of segments; each segment is (k_array, e_array)
    """
    # We'll first collect segments as list-of-rows (each row a list: [k, e1, e2, ...])
    all_segments_rows = []   # list of segments; segment is list of rows
    cur_rows = []
    with open(path, 'r', encoding='utf-8') as f:
        for raw in f:
            line = raw.strip()
            if not line:
                # break segment
                if cur_rows:
                    all_segments_rows.append(cur_rows)
                    cur_rows = []
                continue
            if line.startswith('#'):
                continue
            toks = line.split()
            # try parse numbers
            try:
                nums = [float(t) for t in toks]
            except Exception:
                # skip unparsable lines
                continue
            if len(nums) < 2:
                continue
            cur_rows.append(nums)
    if cur_rows:
        all_segments_rows.append(cur_rows)

    if not all_segments_rows:
        return []  # nothing

    # determine number of branches from the first non-empty row of first segment
    ncols = max(len(row) for seg in all_segments_rows for row in seg)
    # first col is k, remaining are branches
    nbranches = max(1, ncols - 1)

    # Build per-branch segments
    branches = []  # will be list of lists of segments, one list per branch
    for b in range(nbranches):
        branches.append([])

    for seg_rows in all_segments_rows:
        # convert to arrays; rows may have missing entries for some columns -> pad with NaN
        rows_arr = np.array([r + [np.nan] * (ncols - len(r)) for r in seg_rows], dtype=float)
        kcol = rows_arr[:, 0]
        for b in range(nbranches):
            ecol = rows_arr[:, 1 + b]  # may contain NaN
            # avoid segments that are all nan
            if np.all(np.isnan(ecol)):
                continue
            # keep only rows where ecol is finite
            mask = np.isfinite(ecol)
            if not np.any(mask):
                continue
            k_valid = kcol[mask]
            e_valid = ecol[mask]
            # store as a segment
            branches[b].append((k_valid.copy(), e_valid.copy()))
    return branches


# -------------------------
# overlay 多支折线，自动把 overlay 的 k 线性缩放到 target range
# branches: output of read_bands_multi (list per branch of segments)
# ef_overlay: 如果不为 None，则对 overlay 的所有能量做 e - ef_overlay
# -------------------------
def overlay_bands_multi(ax, branches, target_kmin, target_kmax,
                        color='k', lw=0.8, alpha=1.0):
    """
    Plot every branch and its segments on `ax`.
    branches: list where branches[b] is list of (k_arr, e_arr) segments.
    """
    if not branches:
        return
    # collect all k from all branches' segments to compute source k range
    all_k = np.hstack([seg_k for b in branches for (seg_k, seg_e) in b if seg_k.size > 0]) if any(len(b)>0 for b in branches) else np.array([])
    if all_k.size == 0:
        return
    src_kmin = float(np.nanmin(all_k)); src_kmax = float(np.nanmax(all_k))
    if abs(src_kmax - src_kmin) < 1e-12:
        src_kmin -= 0.5; src_kmax += 0.5

    def rescale_k(k):
        return (k - src_kmin) / (src_kmax - src_kmin) * (target_kmax - target_kmin) + target_kmin

    # plot each branch (use slightly different zorder so overlays on top)
    nbranches = len(branches)
    for ib, segs in enumerate(branches):
        if not segs:
            continue
        # optional color cycling per branch: keep same base color but vary alpha/linestyle if desired
        for seg_k, seg_e in segs:
            if seg_k.size == 0:
                continue
            k_res = rescale_k(seg_k)
            e_plot = seg_e.copy()
            ax.plot(k_res, e_plot, color=color, linewidth=lw, alpha=alpha)


# -------------------------
# CLI & main
# -------------------------
def main():
    p = argparse.ArgumentParser(description='Plot phonon unfolded data (k, freq, weight).')
    p.add_argument('file', help='input three-column data file (k freq weight)')
    p.add_argument('--mode', choices=['fatband','density'], default='fatband')
    p.add_argument('--out', '-o', default=None)
    p.add_argument('--color', default='C0', help='fatband color')
    p.add_argument('--size-min', type=float, default=6.0)
    p.add_argument('--size-max', type=float, default=120.0)
    p.add_argument('--alpha-min', type=float, default=0.08)
    p.add_argument('--alpha-max', type=float, default=1.0)
    p.add_argument('--ylim', nargs=2, type=float, metavar=('YMIN','YMAX'))
    p.add_argument('--bins-k', type=int, default=300)
    p.add_argument('--bins-e', type=int, default=400)
    p.add_argument('--log', action='store_true', help='apply log1p to density')
    p.add_argument('--cmap', default='viridis')
    p.add_argument('--vmin', type=float, default=None)
    p.add_argument('--vmax', type=float, default=None)
    p.add_argument('--smooth-sigma', type=float, default=0.0, help='gaussian smooth sigma (pixels)')
    p.add_argument('--hsp', type=str, default=None, help='HSP list like "GAMMA:0,X:49,..." (IDX uses row-index)')
    p.add_argument('--overlay-bands', type=str, default=None, help='bands.out.gnu to overlay as lines')
    p.add_argument('--overlay-color', default='r')
    p.add_argument('--overlay-lw', type=float, default=0.8)
    p.add_argument('--overlay-alpha', type=float, default=1.0)
    args = p.parse_args()

    infile = Path(args.file)
    if not infile.exists():
        raise FileNotFoundError(f"{infile} not found")

    xs, ys, ws, idx2k = read_threecol_file(infile)
    if xs.size == 0:
        raise RuntimeError("No numeric data parsed.")

    # sort by k then freq for nicer scatter plotting
    order = np.lexsort((ys, xs))
    xs_s = xs[order]; ys_s = ys[order]; ws_s = ws[order]

    hsp_list = parse_hsp_arg(args.hsp) if args.hsp else []
    hsp_list = [("GAMMA", 0), ("M", 49), ("K", 99), ("GAMMA", 149),
            ("A", 199), ("L", 249), ("H", 299), ("A", 349)]
    y_lim = tuple(args.ylim) if args.ylim else None

    if args.mode == 'fatband':
        fig, ax = plot_fatband(xs_s, ys_s, ws_s, hsp_list, idx2k,
                               color=args.color,
                               size_min=args.size_min, size_max=args.size_max,
                               alpha_min=args.alpha_min, alpha_max=args.alpha_max,
                               y_lim=y_lim)
    else:
        fig, ax, _ = plot_density(xs_s, ys_s, ws_s, hsp_list, idx2k,
                                  bins_k=args.bins_k, bins_e=args.bins_e,
                                  log_scale=args.log, cmap=args.cmap,
                                  vmin=args.vmin, vmax=args.vmax,
                                  smooth_sigma=args.smooth_sigma, y_lim=y_lim)

    # overlay bands if requested
    if args.overlay_bands:
        bands_path = Path(args.overlay_bands)
        if bands_path.exists():
            branches = read_bands_multi(Path(args.overlay_bands))
            overlay_bands_multi(ax, branches, float(np.nanmin(xs_s)), float(np.nanmax(xs_s)),
                            color=args.overlay_color, lw=args.overlay_lw,
                            alpha=args.overlay_alpha)
        else:
            print(f"[WARN] overlay bands file {bands_path} not found; skipping overlay.")


    plt.tight_layout(pad=0.5)
    if args.out:
        fig.savefig(args.out, dpi=300, bbox_inches='tight')
        print(f"[INFO] saved to {args.out}")
    else:
        plt.show()

if __name__ == '__main__':
    main()
