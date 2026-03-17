"""Analyze gait-cycle error bands for BOTH joint angles and joint torques.

This script:
1) Reads a CSV (e.g., from record_joint_torques.py).
2) Detects gait cycles from a chosen reference signal (e.g., q:ankle_angle_r).
3) Resamples each cycle to a fixed 0–100% gait phase grid.
4) Computes mean +/- band (std/sem/95%CI) across cycles.
5) Plots error-band figures for angles and torques.
6) Optionally exports the aggregated bands to CSV/NPZ.

Example
-------
cd eval
python analyze_gait_bands.py \
  --csv ./runs/walk_006/torques_and_ctrl.csv \
  --ref q:ankle_angle_r --event peak \
  --out ./runs/walk_006/bands \
  --out-data ./runs/walk_006/bands_summary.npz

Notes
-----
- Angle columns are expected to be named like: q:hip_flexion_r / q:hip_flexion_l.
- Torque columns are expected to be named like: hip_flexion_r / hip_flexion_l.
- You can customize joint list and column patterns via CLI.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CycleDetectionResult:
    fs: float
    ref_col: str
    event: str
    est_period_samples: int
    event_indices: np.ndarray
    cycles: List[Tuple[int, int]]
    rejected_cycles: int


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


def estimate_period_samples_autocorr(
    x: np.ndarray,
    fs: float,
    min_period_s: float = 0.4,
    max_period_s: float = 2.0,
) -> int:
    x0 = np.asarray(x, dtype=float)
    x0 = x0[np.isfinite(x0)]
    if x0.size < 10:
        raise ValueError("Not enough finite samples to estimate period")

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
    if time_col not in df.columns:
        raise KeyError(f"Missing time column: {time_col}")
    if ref_col not in df.columns:
        raise KeyError(f"Missing reference column: {ref_col}")
    if event not in {"peak", "trough"}:
        raise ValueError("event must be 'peak' or 'trough'")

    ref_raw = df[ref_col].to_numpy(dtype=float)
    ref_f = lowpass_filter(ref_raw, cutoff_hz=ref_lowpass_hz, fs=fs, order=int(ref_filter_order))

    est_period = estimate_period_samples_autocorr(ref_f, fs=fs, min_period_s=min_period_s, max_period_s=max_period_s)

    from scipy.signal import find_peaks

    target = ref_f if event == "peak" else -ref_f
    distance = max(1, int(round(distance_ratio * est_period)))
    prominence = float(0.25 * np.nanstd(target))

    events, _ = find_peaks(target, distance=distance, prominence=prominence)
    events = np.asarray(events, dtype=int)

    cycles: List[Tuple[int, int]] = []
    rejected = 0
    if events.size >= 2:
        min_len = int(round((1.0 - tol_ratio) * est_period))
        max_len = int(round((1.0 + tol_ratio) * est_period))
        for s, e in zip(events[:-1], events[1:]):
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
        event_indices=events,
        cycles=cycles,
        rejected_cycles=int(rejected),
    )


def resample_cycles_to_phase(x: np.ndarray, cycles: Sequence[Tuple[int, int]], n_phase: int = 101) -> np.ndarray:
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


def compute_band(x_cycles: np.ndarray, band: str = "std") -> Tuple[np.ndarray, np.ndarray]:
    """Return (mean, half_band).

    band:
      - std: mean +/- std
      - sem: mean +/- standard error
      - ci95: mean +/- 1.96 * sem
    """

    if x_cycles.size == 0:
        return np.array([]), np.array([])

    mean = np.mean(x_cycles, axis=0)
    std = np.std(x_cycles, axis=0)

    if band == "std":
        return mean, std

    n = x_cycles.shape[0]
    sem = std / max(1.0, np.sqrt(float(n)))

    if band == "sem":
        return mean, sem
    if band == "ci95":
        return mean, 1.96 * sem

    raise ValueError("band must be one of: std, sem, ci95")


def _build_pairs(joints: Sequence[str], right_suffix: str, left_suffix: str, prefix: str) -> List[Tuple[str, str, str]]:
    pairs: List[Tuple[str, str, str]] = []
    for j in joints:
        r = f"{prefix}{j}{right_suffix}"
        l = f"{prefix}{j}{left_suffix}"
        pairs.append((r, l, j))
    return pairs


def _existing_pairs(df: pd.DataFrame, pairs: Sequence[Tuple[str, str, str]]) -> List[Tuple[str, str, str]]:
    out: List[Tuple[str, str, str]] = []
    for r, l, title in pairs:
        if r in df.columns and l in df.columns:
            out.append((r, l, title))
    return out


def _save_band_data(out_path: Path, phase_pct: np.ndarray, bands: Dict[str, Dict[str, np.ndarray]]):
    """Save in NPZ (recommended) or CSV (wide format) depending on suffix."""

    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.suffix.lower() == ".npz":
        # Flatten nested dict into keys: <group>/<signal>/<stat>
        flat: Dict[str, np.ndarray] = {"phase_pct": phase_pct}
        for key, d in bands.items():
            for stat, arr in d.items():
                flat[f"{key}/{stat}"] = arr
        np.savez(out_path, **flat)
        return

    # CSV wide: phase_pct + each band array as a column
    data: Dict[str, np.ndarray] = {"phase_pct": phase_pct}
    for key, d in bands.items():
        for stat, arr in d.items():
            data[f"{key}__{stat}"] = arr
    pd.DataFrame(data).to_csv(out_path, index=False)


def plot_pairs_band(
    df: pd.DataFrame,
    cycles: Sequence[Tuple[int, int]],
    pairs: Sequence[Tuple[str, str, str]],
    *,
    fs: float,
    lowpass_hz: float,
    filter_order: int,
    n_phase: int,
    band: str,
    ylabel: str,
    title_prefix: str,
    out_path: Optional[Path],
    dpi: int,
    show_legend: bool,
    side: str = "both",
    toe_off: float = 60.0,
    show_phases: bool = True,
    y_lim: Optional[Tuple[float, float]] = None,
) -> Dict[str, Dict[str, np.ndarray]]:
    import matplotlib.pyplot as plt

    if not pairs:
        return {}

    phase_pct = np.linspace(0.0, 100.0, n_phase)

    nrows = len(pairs)
    fig = plt.figure(figsize=(9, 3.2 * nrows), constrained_layout=False)

    bands_out: Dict[str, Dict[str, np.ndarray]] = {}

    for i, (r_col, l_col, base_title) in enumerate(pairs, start=1):
        ax = plt.subplot(nrows, 1, i)
        entry: Dict[str, np.ndarray] = {}

        if side in ["both", "right"] and r_col and r_col in df.columns:
            y_r = lowpass_filter(df[r_col].to_numpy(dtype=float), cutoff_hz=lowpass_hz, fs=fs, order=filter_order)
            r_cycles = resample_cycles_to_phase(y_r, cycles, n_phase=n_phase)
            r_mean, r_half = compute_band(r_cycles, band=band)
            ax.plot(phase_pct, r_mean, label="Right", color="tab:blue", linewidth=2.5)
            ax.fill_between(phase_pct, r_mean - r_half, r_mean + r_half, color="tab:blue", alpha=0.25, linewidth=0)
            entry["right_mean"] = r_mean
            entry["right_band"] = r_half

        if side in ["both", "left"] and l_col and l_col in df.columns:
            y_l = lowpass_filter(df[l_col].to_numpy(dtype=float), cutoff_hz=lowpass_hz, fs=fs, order=filter_order)
            l_cycles = resample_cycles_to_phase(y_l, cycles, n_phase=n_phase)
            l_mean, l_half = compute_band(l_cycles, band=band)
            ax.plot(phase_pct, l_mean, label="Left", color="tab:red", linewidth=2.5)
            ax.fill_between(phase_pct, l_mean - l_half, l_mean + l_half, color="tab:red", alpha=0.25, linewidth=0)
            entry["left_mean"] = l_mean
            entry["left_band"] = l_half

        ax.set_xlim(0, 100)
        if y_lim:
            ax.set_ylim(y_lim)
        ax.set_xlabel("Gait phase (%)")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{title_prefix}{base_title}")
        ax.grid(True, alpha=0.15)

        if show_phases:
            # Add vertical line for Toe-Off
            ax.axvline(x=toe_off, color="black", linestyle="--", alpha=0.5, linewidth=1.2)
            
            # Add text labels for Stance and Swing
            y_min, y_max = ax.get_ylim()
            label_y = y_min + (y_max - y_min) * 0.02
            ax.text(toe_off / 2, label_y, "STANCE", ha="center", fontsize=9, fontweight="bold", alpha=0.6)
            ax.text(toe_off + (100 - toe_off) / 2, label_y, "SWING", ha="center", fontsize=9, fontweight="bold", alpha=0.6)

        if show_legend:
            ax.legend(loc="upper left", frameon=True, framealpha=0.6)

        key = f"{title_prefix.strip()}::{base_title}".strip(":")
        bands_out[key] = entry

    plt.tight_layout()

    if out_path is None:
        plt.show()
    else:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=dpi)
        plt.close(fig)

    return bands_out


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csv", type=str, required=True, help="Input CSV")
    p.add_argument("--time", type=str, default="time", help="Time column")
    p.add_argument("--fs", type=float, default=0.0, help="Sampling rate (Hz). If 0, estimate from time.")

    p.add_argument("--ref", type=str, default="q:ankle_angle_r", help="Reference column for cycle detection")
    p.add_argument("--event", type=str, default="peak", choices=["peak", "trough"], help="Event type")

    p.add_argument("--ref-lowpass", type=float, default=6.0, help="Low-pass cutoff for ref (Hz)")
    p.add_argument("--ref-filter-order", type=int, default=4, help="Filter order for ref")
    p.add_argument("--min-period", type=float, default=0.4, help="Min gait period (s)")
    p.add_argument("--max-period", type=float, default=2.0, help="Max gait period (s)")
    p.add_argument("--tol", type=float, default=0.35, help="Cycle length tolerance ratio")

    p.add_argument("--joints", type=str, default="hip_flexion,knee_angle,ankle_angle,subtalar_angle,mtp_angle", help="Comma-separated joints")
    p.add_argument("--right-suffix", type=str, default="_r", help="Right side suffix")
    p.add_argument("--left-suffix", type=str, default="_l", help="Left side suffix")
    p.add_argument("--angle-prefix", type=str, default="q:", help="Prefix for angle columns")
    p.add_argument("--torque-prefix", type=str, default="", help="Prefix for torque columns")

    p.add_argument("--angle-lowpass", type=float, default=6.0, help="Low-pass cutoff for angles (Hz)")
    p.add_argument("--torque-lowpass", type=float, default=6.0, help="Low-pass cutoff for torques (Hz)")
    p.add_argument("--activation-lowpass", type=float, default=10.0, help="Low-pass cutoff for activations (Hz)")
    p.add_argument("--filter-order", type=int, default=4, help="Filter order for angles/torques/activations")

    p.add_argument("--n-phase", type=int, default=101, help="Samples per normalized cycle")
    p.add_argument("--band", type=str, default="std", choices=["std", "sem", "ci95"], help="Band type")
    p.add_argument("--side", type=str, default="both", choices=["both", "right", "left"], help="Leg side to plot")

    p.add_argument("--out", type=str, default="", help="Output base path: dir or file stem. If empty, show plots.")
    p.add_argument("--out-angles", type=str, default="", help="Output path for angle figure")
    p.add_argument("--out-torques", type=str, default="", help="Output path for torque figure")
    p.add_argument("--out-activations", type=str, default="", help="Output path for activations figure")
    p.add_argument("--out-data", type=str, default="", help="Optional export of bands to .npz or .csv")
    p.add_argument("--dpi", type=int, default=350)
    p.add_argument("--no-legend", action="store_true")

    p.add_argument("--toe-off", type=float, default=60.0, help="Gait phase percentage for Toe-Off vertical line")
    p.add_argument("--no-phases", action="store_true", help="Do not draw stance/swing phase markers")

    args = p.parse_args(argv)

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    df = pd.read_csv(csv_path)

    fs = float(args.fs) if args.fs and args.fs > 0 else estimate_fs_from_time(df[args.time].to_numpy(dtype=float))

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
        f"  fs                 : {detection.fs:.3f} Hz\n"
        f"  ref                : {detection.ref_col} ({detection.event})\n"
        f"  est_period_samples  : {detection.est_period_samples}\n"
        f"  events_detected     : {int(detection.event_indices.size)}\n"
        f"  cycles_kept         : {len(detection.cycles)}\n"
        f"  cycles_rejected     : {detection.rejected_cycles}"
    )

    if len(detection.cycles) < 3:
        raise RuntimeError(
            "Too few cycles detected (<3). Try changing --ref/--event, or relax --tol, or adjust --min-period/--max-period."
        )

    joints = [j.strip() for j in str(args.joints).split(",") if j.strip()]

    angle_pairs = _existing_pairs(
        df,
        _build_pairs(joints, right_suffix=args.right_suffix, left_suffix=args.left_suffix, prefix=str(args.angle_prefix)),
    )
    torque_pairs = _existing_pairs(
        df,
        _build_pairs(joints, right_suffix=args.right_suffix, left_suffix=args.left_suffix, prefix=str(args.torque_prefix)),
    )

    # output path resolution
    base_out = Path(args.out) if args.out else None

    def _derive(base: Optional[Path], name: str) -> Optional[Path]:
        if base is None:
            return None
        if base.suffix:
            return base.parent / f"{base.stem}_{name}{base.suffix}"
        # directory-like
        return base / f"{name}.png"

    out_angles = Path(args.out_angles) if args.out_angles else _derive(base_out, "angles")
    out_torques = Path(args.out_torques) if args.out_torques else _derive(base_out, "torques")
    out_activations = Path(args.out_activations) if args.out_activations else _derive(base_out, "activations")

    bands_all: Dict[str, Dict[str, np.ndarray]] = {}

    if angle_pairs:
        bands_angles = plot_pairs_band(
            df,
            detection.cycles,
            angle_pairs,
            fs=detection.fs,
            lowpass_hz=float(args.angle_lowpass),
            filter_order=int(args.filter_order),
            n_phase=int(args.n_phase),
            band=str(args.band),
            ylabel="Angle (rad)",
            title_prefix="Joint Angle: ",
            out_path=out_angles,
            dpi=int(args.dpi),
            show_legend=not bool(args.no_legend),
            side=str(args.side),
            toe_off=float(args.toe_off),
            show_phases=not bool(args.no_phases),
        )
        bands_all.update({f"angles/{k}": v for k, v in bands_angles.items()})
    else:
        print("[warn] No angle pairs found. Check column names and --angle-prefix.")

    if torque_pairs:
        bands_torques = plot_pairs_band(
            df,
            detection.cycles,
            torque_pairs,
            fs=detection.fs,
            lowpass_hz=float(args.torque_lowpass),
            filter_order=int(args.filter_order),
            n_phase=int(args.n_phase),
            band=str(args.band),
            ylabel="Torque (Nm)",
            title_prefix="Joint Torque: ",
            out_path=out_torques,
            dpi=int(args.dpi),
            show_legend=not bool(args.no_legend),
            side=str(args.side),
            toe_off=float(args.toe_off),
            show_phases=not bool(args.no_phases),
            y_lim=(-150, 150),
        )
        bands_all.update({f"torques/{k}": v for k, v in bands_torques.items()})
    else:
        print("[warn] No torque pairs found. Check column names and --torque-prefix.")

    # Activation plotting
    muscles = ["med_gas", "lat_gas", "soleus", "tib_ant", "vas_lat", "bifemsh"]
    activation_pairs = _existing_pairs(
        df,
        _build_pairs(muscles, right_suffix=args.right_suffix, left_suffix=args.left_suffix, prefix="act:"),
    )
    if activation_pairs:
        bands_activations = plot_pairs_band(
            df,
            detection.cycles,
            activation_pairs,
            fs=detection.fs,
            lowpass_hz=float(args.activation_lowpass),
            filter_order=int(args.filter_order),
            n_phase=int(args.n_phase),
            band=str(args.band),
            ylabel="Activation",
            title_prefix="Activation: ",
            out_path=out_activations,
            dpi=int(args.dpi),
            show_legend=not bool(args.no_legend),
            side=str(args.side),
            toe_off=float(args.toe_off),
            show_phases=not bool(args.no_phases),
        )
        bands_all.update({f"activations/{k}": v for k, v in bands_activations.items()})
    else:
        print("[info] No activation pairs found for requested muscles.")

    if args.out_data:
        phase_pct = np.linspace(0.0, 100.0, int(args.n_phase))
        out_data = Path(args.out_data)
        _save_band_data(out_data, phase_pct, bands_all)
        print(f"Saved band data to: {out_data}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
