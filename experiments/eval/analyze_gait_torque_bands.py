"""Analyze joint torque error bands over a single gait cycle.

Goal
----
Given recorded joint torque time series, segment the data into gait cycles
*from the signal itself* (no fixed cycle length assumption), then compute
mean/std across cycles and plot an error band over 0–100% gait phase.

This is intentionally lightweight and CSV-first, to complement
`record_joint_torques.py`.

Example
-------
cd /Users/ccg/Study/Mujoco/IL/musculoco_learning/experiments/eval
python analyze_gait_torque_bands.py \
  --csv ./runs/run_001/torques_and_ctrl.csv \
  --fs 50 \
  --ref ankle_angle_r \
  --event peak \
  --out ./runs/run_001/torque_bands.png
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CycleDetectionResult:
    fs: float
    ref_col: str
    event: str
    est_period_samples: int
    peak_indices: np.ndarray
    cycles: List[Tuple[int, int]]
    rejected_cycles: int


def _butter_lowpass(cutoff_hz: float, fs: float, order: int = 4):
    from scipy.signal import butter

    nyq = 0.5 * fs
    if cutoff_hz <= 0 or cutoff_hz >= nyq:
        raise ValueError(f"cutoff_hz must be in (0, {nyq}), got {cutoff_hz}")
    normal_cutoff = cutoff_hz / nyq
    b, a = butter(order, normal_cutoff, btype="low", analog=False)
    return b, a


def lowpass_filter(x: np.ndarray, cutoff_hz: float, fs: float, order: int = 4) -> np.ndarray:
    from scipy.signal import filtfilt

    b, a = _butter_lowpass(cutoff_hz=cutoff_hz, fs=fs, order=order)
    return filtfilt(b, a, x)


def estimate_fs_from_time(time_s: Sequence[float]) -> float:
    t = np.asarray(time_s, dtype=float)
    if t.size < 3:
        raise ValueError("Need at least 3 time samples to estimate fs")
    dt = np.diff(t)
    dt = dt[np.isfinite(dt)]
    dt = dt[dt > 0]
    if dt.size == 0:
        raise ValueError("Time column must be strictly increasing")
    return float(1.0 / np.median(dt))


def estimate_period_samples_autocorr(
    x: np.ndarray,
    fs: float,
    min_period_s: float = 0.4,
    max_period_s: float = 2.0,
) -> int:
    """Estimate dominant period using autocorrelation peak search."""

    x0 = np.asarray(x, dtype=float)
    x0 = x0[np.isfinite(x0)]
    if x0.size < 10:
        raise ValueError("Not enough finite samples to estimate period")

    # Demean to reduce DC component.
    x0 = x0 - np.mean(x0)

    ac = np.correlate(x0, x0, mode="full")
    ac = ac[ac.size // 2 :]

    min_lag = max(1, int(round(min_period_s * fs)))
    max_lag = min(ac.size - 1, int(round(max_period_s * fs)))
    if max_lag <= min_lag:
        raise ValueError(
            f"Invalid lag search range: min_lag={min_lag}, max_lag={max_lag}. "
            "Adjust min_period_s/max_period_s or provide --fs."
        )

    search = ac[min_lag : max_lag + 1]
    best = int(np.argmax(search)) + min_lag
    return int(best)


def detect_gait_cycles(
    df: pd.DataFrame,
    time_col: str,
    ref_col: str,
    fs: float,
    *,
    event: str = "peak",
    ref_lowpass_hz: float = 6.0,
    ref_filter_order: int = 4,
    min_period_s: float = 0.4,
    max_period_s: float = 2.0,
    distance_ratio: float = 0.6,
    tol_ratio: float = 0.35,
) -> CycleDetectionResult:
    """Detect gait cycle boundaries from a reference signal.

    Strategy
    --------
    1) Low-pass filter reference.
    2) Estimate dominant period via autocorrelation.
    3) Use `scipy.signal.find_peaks` on (ref) or (-ref) to detect events.
    4) Cycles are consecutive events; filter by duration near the estimated period.

    Notes
    -----
    Without ground reaction forces / contact sensors, "event" corresponds to a
    repeatable kinematic event (e.g., a consistent ankle angle peak). This is
    still strict (data-driven), but the biomechanical meaning of the event
    depends on the chosen reference.
    """

    if time_col not in df.columns:
        raise KeyError(f"Missing time column: {time_col}")
    if ref_col not in df.columns:
        raise KeyError(f"Missing reference column: {ref_col}")
    if event not in {"peak", "trough"}:
        raise ValueError("event must be 'peak' or 'trough'")

    ref_raw = df[ref_col].to_numpy(dtype=float)
    ref_f = lowpass_filter(ref_raw, cutoff_hz=ref_lowpass_hz, fs=fs, order=int(ref_filter_order))

    est_period = estimate_period_samples_autocorr(
        ref_f,
        fs=fs,
        min_period_s=min_period_s,
        max_period_s=max_period_s,
    )

    from scipy.signal import find_peaks

    target = ref_f if event == "peak" else -ref_f

    # Heuristics: keep the API simple; user can tune via CLI.
    distance = max(1, int(round(distance_ratio * est_period)))
    prominence = float(0.25 * np.nanstd(target))

    peaks, _props = find_peaks(target, distance=distance, prominence=prominence)
    peaks = np.asarray(peaks, dtype=int)

    # Build cycles from consecutive peaks.
    cycles: List[Tuple[int, int]] = []
    rejected = 0
    if peaks.size >= 2:
        min_len = int(round((1.0 - tol_ratio) * est_period))
        max_len = int(round((1.0 + tol_ratio) * est_period))
        for s, e in zip(peaks[:-1], peaks[1:]):
            length = int(e - s)
            if length < min_len or length > max_len:
                rejected += 1
                continue
            cycles.append((int(s), int(e)))

    return CycleDetectionResult(
        fs=float(fs),
        ref_col=ref_col,
        event=event,
        est_period_samples=int(est_period),
        peak_indices=peaks,
        cycles=cycles,
        rejected_cycles=int(rejected),
    )


def resample_cycles_to_phase(
    x: np.ndarray,
    cycles: Sequence[Tuple[int, int]],
    n_phase: int = 101,
) -> np.ndarray:
    """Resample each (start,end) segment to a fixed phase grid [0,1]."""

    x = np.asarray(x, dtype=float)
    phase = np.linspace(0.0, 1.0, n_phase)

    out: List[np.ndarray] = []
    for s, e in cycles:
        seg = x[s:e]
        if seg.size < 2:
            continue
        seg_t = np.linspace(0.0, 1.0, seg.size)
        out.append(np.interp(phase, seg_t, seg))

    if not out:
        return np.empty((0, n_phase), dtype=float)
    return np.stack(out, axis=0)


def compute_mean_std(x_cycles: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if x_cycles.size == 0:
        return np.array([]), np.array([])
    return np.mean(x_cycles, axis=0), np.std(x_cycles, axis=0)


def _default_joint_pairs(df: pd.DataFrame) -> List[Tuple[str, str, str]]:
    """(right, left, title)"""
    pairs = [
        ("hip_flexion_r", "hip_flexion_l", "Hip Flexion Torque"),
        ("knee_angle_r", "knee_angle_l", "Knee Torque"),
        ("ankle_angle_r", "ankle_angle_l", "Ankle Torque"),
    ]
    return [p for p in pairs if p[0] in df.columns and p[1] in df.columns]


def _default_activation_pairs(df: pd.DataFrame) -> List[Tuple[str, str, str]]:
    """Default muscle activation pairs (right, left, title)."""

    pairs = [
        ("act:soleus_r", "act:soleus_l", "Soleus Activation"),
        ("act:med_gas_r", "act:med_gas_l", "Medial Gastrocnemius Activation"),
        ("act:lat_gas_r", "act:lat_gas_l", "Lateral Gastrocnemius Activation"),
        ("act:tib_ant_r", "act:tib_ant_l", "Tibialis Anterior Activation"),
    ]
    return [p for p in pairs if p[0] in df.columns and p[1] in df.columns]


def _derive_out_paths(out: str) -> Tuple[Optional[Path], Optional[Path]]:
    """Derive (torque_out, act_out) from a single --out.

    If out is empty -> (None, None).
    If out has suffix -> insert _torque/_act before suffix.
    """

    if not out:
        return None, None
    p = Path(out)
    suffix = p.suffix
    stem = p.stem
    parent = p.parent
    if suffix:
        return parent / f"{stem}_torque{suffix}", parent / f"{stem}_act{suffix}"
    # No suffix: treat as directory-like stem
    return parent / f"{stem}_torque.png", parent / f"{stem}_act.png"


def _major_activation_columns(side: str) -> List[Tuple[str, str]]:
    """Return [(column_name, pretty_name), ...] for major muscles."""

    if side not in {"right", "left"}:
        raise ValueError("side must be 'right' or 'left'")

    suf = "_r" if side == "right" else "_l"
    return [
        (f"act:tib_ant{suf}", "TA"),
        (f"act:soleus{suf}", "SL"),
        (f"act:med_gas{suf}", "MG"),
        (f"act:lat_gas{suf}", "LG"),
    ]


def _activation_color_map():
    """High-contrast but pleasant colors for activation curves."""
    return {
        # Unified cool color family (blue/cyan/purple/teal), with clear separation.
        "TA": "tab:purple",
        "SL": "tab:blue",
        "MG": "tab:cyan",
        "LG": "tab:green",
    }


def _plot_activation_means_overlay(
    ax,
    df: pd.DataFrame,
    cycles: Sequence[Tuple[int, int]],
    *,
    fs: float,
    lowpass_hz: float,
    filter_order: int,
    n_phase: int,
    side: str,
    show_legend: bool,
):
    phase_pct = np.linspace(0.0, 100.0, n_phase)

    color_map = _activation_color_map()
    muscle_specs = _major_activation_columns(side)

    plotted = 0
    for idx, (col, pretty) in enumerate(muscle_specs):
        if col not in df.columns:
            continue

        y = lowpass_filter(
            df[col].to_numpy(dtype=float),
            cutoff_hz=lowpass_hz,
            fs=fs,
            order=int(filter_order),
        )
        y_cycles = resample_cycles_to_phase(y, cycles, n_phase=n_phase)
        y_mean, _y_std = compute_mean_std(y_cycles)
        if y_mean.size == 0:
            continue

        ax.plot(
            phase_pct,
            y_mean,
            label=f"{pretty}",
            linewidth=3.0,
            color=color_map.get(pretty, "tab:gray"),
        )
        plotted += 1

    ax.set_title(f"Major Muscle Activations ({side.capitalize()})")
    ax.set_xlabel("Gait phase (%)")
    ax.set_ylabel("Activation")
    _tight_phase_xlim(ax)
    if show_legend and plotted > 0:
        ax.legend(
            loc="lower left",
            frameon=True,
            fancybox=True,
            framealpha=1.0,
            ncol=2,
        )


def _legend_inside(ax, ncol: int = 1, *, enabled: bool = True):
    if not enabled:
        return
    ax.legend(
        loc="lower left",
        frameon=True,
        fancybox=True,
        framealpha=1.0,
        ncol=int(ncol),
    )


def _apply_plot_style(save_dpi: int):
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.size": 20,
            "axes.titlesize": 30,
            "axes.labelsize": 25,
            "legend.fontsize": 20,
            "xtick.labelsize": 25,
            "ytick.labelsize": 25,
            "axes.grid": False,
            "grid.alpha": 0.0,
            # Frame/spines
            "axes.linewidth": 2.0,
            # Ticks: inward, thicker, a bit longer
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.major.size": 7,
            "ytick.major.size": 7,
            "xtick.minor.size": 4,
            "ytick.minor.size": 4,
            "xtick.major.width": 1.8,
            "ytick.major.width": 1.8,
            "xtick.minor.width": 1.4,
            "ytick.minor.width": 1.4,
            "lines.antialiased": True,
            "lines.linewidth": 3.0,
            "figure.dpi": 120,
            "savefig.dpi": int(save_dpi),
        }
    )


def _densify_y_ticks(ax, *, major_nbins: int = 4):
    """Increase y-axis tick density using more major ticks (no minor ticks)."""

    from matplotlib.ticker import MaxNLocator

    ax.yaxis.set_major_locator(MaxNLocator(nbins=int(major_nbins)))
    ax.minorticks_off()


def _tight_phase_xlim(ax):
    """Make gait-phase x-axis run exactly 0..100 with no side margins."""

    ax.set_xlim(0.0, 100.0)
    ax.margins(x=0.0)


def _save_or_show(out_path: Optional[Path], save_dpi: int):
    import matplotlib.pyplot as plt

    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=int(save_dpi), bbox_inches="tight")
        print(f"Saved figure: {out_path}")
    else:
        plt.show()


def plot_torque_figure(
    df: pd.DataFrame,
    detection: CycleDetectionResult,
    *,
    time_col: str,
    torque_pairs: Sequence[Tuple[str, str, str]],
    lowpass_hz: float,
    filter_order: int,
    ref_filter_order: int,
    n_phase: int,
    show_debug: bool,
    figwidth: float,
    figheight_per_row: float,
    save_dpi: int,
    out_path: Optional[Path],
    show_legend: bool,
):
    import matplotlib.pyplot as plt

    _apply_plot_style(save_dpi=save_dpi)

    color_r = "tab:blue"
    color_l = "tab:red"

    cycles = detection.cycles
    if len(cycles) < 2:
        raise RuntimeError(
            "Detected too few cycles to compute plots. "
            "Try changing --ref/--event or adjust --min-period/--max-period/--tol."
        )

    phase_pct = np.linspace(0.0, 100.0, n_phase)

    nrows = len(torque_pairs) + (1 if show_debug else 0)
    fig = plt.figure(figsize=(float(figwidth), float(figheight_per_row) * nrows))
    axes: List["plt.Axes"] = []

    row = 1
    if show_debug:
        t = df[time_col].to_numpy(dtype=float)
        ref = df[detection.ref_col].to_numpy(dtype=float)
        ref_f = lowpass_filter(
            ref,
            cutoff_hz=min(6.0, lowpass_hz),
            fs=detection.fs,
            order=int(ref_filter_order),
        )
        ax = plt.subplot(nrows, 1, row)
        ax.plot(t, ref_f, label=f"ref_filt: {detection.ref_col}")
        peak_t = t[detection.peak_indices] if detection.peak_indices.size > 0 else np.array([])
        if peak_t.size > 0:
            ax.scatter(peak_t, ref_f[detection.peak_indices], s=20, c="r", label="events")
        ax.set_title(
            f"Reference & detected events | est_period={detection.est_period_samples} samples, "
            f"cycles={len(detection.cycles)} (rejected={detection.rejected_cycles})"
        )
        ax.set_xlabel("Time (s)")
        _densify_y_ticks(ax)
        _legend_inside(ax, enabled=bool(show_legend))
        axes.append(ax)
        row += 1

    for right_col, left_col, title in torque_pairs:
        y_r = lowpass_filter(
            df[right_col].to_numpy(dtype=float),
            cutoff_hz=float(lowpass_hz),
            fs=detection.fs,
            order=int(filter_order),
        )
        y_l = lowpass_filter(
            df[left_col].to_numpy(dtype=float),
            cutoff_hz=float(lowpass_hz),
            fs=detection.fs,
            order=int(filter_order),
        )

        r_cycles = resample_cycles_to_phase(y_r, cycles, n_phase=n_phase)
        l_cycles = resample_cycles_to_phase(y_l, cycles, n_phase=n_phase)
        r_mean, r_std = compute_mean_std(r_cycles)
        l_mean, l_std = compute_mean_std(l_cycles)

        ax = plt.subplot(nrows, 1, row)
        ax.plot(phase_pct, r_mean, label=f"Right", color=color_r, linewidth=3.0)
        ax.fill_between(phase_pct, r_mean - r_std, r_mean + r_std, color=color_r, alpha=0.25, linewidth=0)
        ax.plot(phase_pct, l_mean, label=f"Left", color=color_l, linewidth=3.0)
        ax.fill_between(phase_pct, l_mean - l_std, l_mean + l_std, color=color_l, alpha=0.25, linewidth=0)
        ax.set_title(f"{title}")
        ax.set_xlabel("Gait phase (%)")
        ax.set_ylabel("Torque")
        _tight_phase_xlim(ax)
        _densify_y_ticks(ax)
        _legend_inside(ax, enabled=bool(show_legend))
        axes.append(ax)
        row += 1

    # Align y-labels across subplots.
    fig.align_ylabels(axes)
    plt.tight_layout()
    _save_or_show(out_path=out_path, save_dpi=save_dpi)


def plot_activation_figure(
    df: pd.DataFrame,
    detection: CycleDetectionResult,
    *,
    activation_pairs: Sequence[Tuple[str, str, str]],
    act_lowpass_hz: float,
    act_filter_order: int,
    n_phase: int,
    plot_activation_means: bool,
    activation_side: str,
    figwidth: float,
    figheight_per_row: float,
    save_dpi: int,
    out_path: Optional[Path],
    show_legend: bool,
):
    import matplotlib.pyplot as plt

    _apply_plot_style(save_dpi=save_dpi)

    cycles = detection.cycles
    if len(cycles) < 2:
        raise RuntimeError("Detected too few cycles to compute plots")

    phase_pct = np.linspace(0.0, 100.0, n_phase)

    # Rows: activation error bands (optional, per muscle-pair) + activation means (optional)
    extra_rows = 0
    if plot_activation_means:
        extra_rows += 2 if activation_side == "both" else 1

    nrows = len(activation_pairs) + extra_rows
    if nrows <= 0:
        return

    fig = plt.figure(figsize=(float(figwidth), float(figheight_per_row) * nrows))
    axes: List["plt.Axes"] = []
    row = 1

    # Error bands for act: pairs (optional)
    for right_col, left_col, title in activation_pairs:
        y_r = lowpass_filter(
            df[right_col].to_numpy(dtype=float),
            cutoff_hz=float(act_lowpass_hz),
            fs=detection.fs,
            order=int(act_filter_order),
        )
        y_l = lowpass_filter(
            df[left_col].to_numpy(dtype=float),
            cutoff_hz=float(act_lowpass_hz),
            fs=detection.fs,
            order=int(act_filter_order),
        )

        r_cycles = resample_cycles_to_phase(y_r, cycles, n_phase=n_phase)
        l_cycles = resample_cycles_to_phase(y_l, cycles, n_phase=n_phase)
        r_mean, r_std = compute_mean_std(r_cycles)
        l_mean, l_std = compute_mean_std(l_cycles)

        ax = plt.subplot(nrows, 1, row)
        ax.plot(phase_pct, r_mean, label=f"Right ", color="tab:blue", linewidth=3.0)
        ax.fill_between(phase_pct, r_mean - r_std, r_mean + r_std, color="tab:blue", alpha=0.25, linewidth=0)
        ax.plot(phase_pct, l_mean, label=f"Left ", color="tab:red", linewidth=3.0)
        ax.fill_between(phase_pct, l_mean - l_std, l_mean + l_std, color="tab:red", alpha=0.25, linewidth=0)
        ax.set_title(f"{title}")
        ax.set_xlabel("Gait phase (%)")
        ax.set_ylabel("Activation")
        _tight_phase_xlim(ax)
        _densify_y_ticks(ax)
        _legend_inside(ax, enabled=bool(show_legend))
        axes.append(ax)
        row += 1

    # Means overlay (requested)
    if plot_activation_means:
        if activation_side not in {"right", "left", "both"}:
            raise ValueError("activation_side must be one of: right, left, both")

        if activation_side in {"right", "both"}:
            ax = plt.subplot(nrows, 1, row)
            _plot_activation_means_overlay(
                ax,
                df,
                cycles,
                fs=detection.fs,
                lowpass_hz=float(act_lowpass_hz),
                filter_order=int(act_filter_order),
                n_phase=n_phase,
                side="right",
                show_legend=bool(show_legend),
            )
            _densify_y_ticks(ax)
            axes.append(ax)
            row += 1

        if activation_side in {"left", "both"}:
            ax = plt.subplot(nrows, 1, row)
            _plot_activation_means_overlay(
                ax,
                df,
                cycles,
                fs=detection.fs,
                lowpass_hz=float(act_lowpass_hz),
                filter_order=int(act_filter_order),
                n_phase=n_phase,
                side="left",
                show_legend=bool(show_legend),
            )
            _densify_y_ticks(ax)
            axes.append(ax)
            row += 1

    # Align y-labels across subplots.
    fig.align_ylabels(axes)
    plt.tight_layout()
    _save_or_show(out_path=out_path, save_dpi=save_dpi)


 


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csv", type=str, required=True, help="Input CSV (e.g., runs/run_001/torques_and_ctrl.csv)")
    p.add_argument("--time", type=str, default="time", help="Time column name")
    p.add_argument("--fs", type=float, default=0.0, help="Sampling frequency (Hz). If 0, estimate from time column.")

    p.add_argument("--ref", type=str, default="ankle_angle_r", help="Reference column used to segment gait cycles")
    p.add_argument("--event", type=str, default="peak", choices=["peak", "trough"], help="Event type on reference")

    p.add_argument("--ref-lowpass", type=float, default=6.0, help="Low-pass cutoff for reference signal (Hz)")
    p.add_argument("--lowpass", type=float, default=6.0, help="Low-pass cutoff for torque signals (Hz)")
    p.add_argument(
        "--act-lowpass",
        type=float,
        default=6.0,
        help="Low-pass cutoff for activation signals (Hz). Lower = smoother.",
    )
    p.add_argument(
        "--filter-order",
        type=int,
        default=4,
        help="Butterworth low-pass order for torque signals (higher = stronger roll-off)",
    )
    p.add_argument(
        "--act-filter-order",
        type=int,
        default=4,
        help="Butterworth low-pass order for activation signals",
    )
    p.add_argument(
        "--ref-filter-order",
        type=int,
        default=4,
        help="Butterworth low-pass order for reference signal",
    )

    p.add_argument("--min-period", type=float, default=0.4, help="Min expected gait period (s)")
    p.add_argument("--max-period", type=float, default=2.0, help="Max expected gait period (s)")
    p.add_argument("--tol", type=float, default=0.35, help="Allowed relative deviation from estimated period for cycle durations")

    p.add_argument("--n-phase", type=int, default=101, help="Number of samples per normalized gait cycle")
    p.add_argument("--debug", action="store_true", help="Show reference + detected events plot")
    p.add_argument(
        "--plot-activations",
        action="store_true",
        help="Also plot activation (act:) error bands for soleus/gastrocnemius/tib_ant",
    )
    # Backward/typo-friendly alias (user often types singular).
    p.add_argument(
        "--plot-activation",
        dest="plot_activations",
        action="store_true",
        help="Alias of --plot-activations",
    )
    p.add_argument(
        "--plot-activation-means",
        action="store_true",
        help="Plot major muscle activations mean curves on one subplot (no error bands)",
    )
    p.add_argument(
        "--activation-side",
        type=str,
        default="right",
        choices=["right", "left", "both"],
        help="Which side to plot for --plot-activation-means",
    )

    p.add_argument("--out", type=str, default="", help="Base output path. If empty, show interactively.")
    p.add_argument("--out-torque", type=str, default="", help="Explicit output path for torque figure")
    p.add_argument("--out-act", type=str, default="", help="Explicit output path for activation figure")
    p.add_argument(
        "--dpi",
        type=int,
        default=350,
        help="Output DPI for raster formats (e.g., png). Ignored for vector formats like pdf/svg.",
    )
    p.add_argument(
        "--no-legend",
        action="store_true",
        help="Do not draw legends on figures.",
    )

    args = p.parse_args(argv)

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    df = pd.read_csv(csv_path)

    if args.fs and args.fs > 0:
        fs = float(args.fs)
    else:
        fs = estimate_fs_from_time(df[args.time].to_numpy(dtype=float))

    detection = detect_gait_cycles(
        df,
        time_col=args.time,
        ref_col=args.ref,
        fs=fs,
        event=args.event,
        ref_lowpass_hz=float(args.ref_lowpass),
        ref_filter_order=int(args.ref_filter_order),
        min_period_s=float(args.min_period),
        max_period_s=float(args.max_period),
        tol_ratio=float(args.tol),
    )

    print(
        "Cycle detection summary:\n"
        f"  fs                : {detection.fs:.3f} Hz\n"
        f"  ref               : {detection.ref_col} ({detection.event})\n"
        f"  est_period_samples : {detection.est_period_samples}\n"
        f"  events_detected    : {int(detection.peak_indices.size)}\n"
        f"  cycles_kept        : {len(detection.cycles)}\n"
        f"  cycles_rejected    : {detection.rejected_cycles}"
    )

    torque_pairs = _default_joint_pairs(df)
    activation_pairs: List[Tuple[str, str, str]] = []
    if args.plot_activations:
        activation_pairs = _default_activation_pairs(df)
    if not torque_pairs:
        # Fallback: just plot any *_r/*_l pairs with common base.
        rights = [c for c in df.columns if c.endswith("_r")]
        torque_pairs = []
        for r in rights:
            l = r[:-2] + "_l"
            if l in df.columns:
                torque_pairs.append((r, l, r[:-2]))

    derived_torque_out, derived_act_out = _derive_out_paths(args.out)
    torque_out = Path(args.out_torque) if args.out_torque else derived_torque_out
    act_out = Path(args.out_act) if args.out_act else derived_act_out

    # If user only wants one figure (no activations), keep backward behavior:
    # use --out as torque output if provided.
    activations_requested = bool(args.plot_activations or args.plot_activation_means)
    if not activations_requested:
        torque_out = Path(args.out) if args.out else None

    plot_torque_figure(
        df,
        detection,
        time_col=args.time,
        torque_pairs=torque_pairs,
        lowpass_hz=float(args.lowpass),
        filter_order=int(args.filter_order),
        ref_filter_order=int(args.ref_filter_order),
        n_phase=int(args.n_phase),
        show_debug=bool(args.debug),
        figwidth=10,
        figheight_per_row=4.0,
        save_dpi=int(args.dpi),
        out_path=torque_out,
        show_legend=not bool(args.no_legend),
    )

    plot_activation_figure(
        df,
        detection,
        activation_pairs=activation_pairs,
        act_lowpass_hz=float(args.act_lowpass),
        act_filter_order=int(args.act_filter_order),
        n_phase=int(args.n_phase),
        plot_activation_means=bool(args.plot_activation_means),
        activation_side=str(args.activation_side),
        figwidth=10,
        figheight_per_row=4.0,
        save_dpi=int(args.dpi),
        out_path=act_out,
        show_legend=not bool(args.no_legend),
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
