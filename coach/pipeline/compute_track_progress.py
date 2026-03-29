from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _close_curve(points: np.ndarray) -> np.ndarray:
    if np.linalg.norm(points[0] - points[-1]) > 1e-6:
        return np.vstack([points, points[0]])
    return points


def _resample_closed(points: np.ndarray, n: int) -> np.ndarray:
    points = _close_curve(points)
    seg = np.linalg.norm(np.diff(points, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    target = np.linspace(0.0, s[-1], n, endpoint=False)
    out = np.empty((n, points.shape[1]), dtype=float)
    for dim in range(points.shape[1]):
        out[:, dim] = np.interp(target, s, points[:, dim])
    return out


def _align_right_to_left(left: np.ndarray, right: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    n = len(left)
    coarse_step = max(1, n // 128)
    best = None
    for reversed_flag in (False, True):
        candidate = right[::-1] if reversed_flag else right
        for shift in range(0, n, coarse_step):
            rolled = np.roll(candidate, shift, axis=0)
            dist = np.linalg.norm(left - rolled, axis=1)
            score = float(np.median(dist))
            if best is None or score < best[0]:
                best = (score, shift, reversed_flag, float(np.mean(dist)))
        lo = best[1] - coarse_step
        hi = best[1] + coarse_step + 1
        for shift in range(lo, hi):
            rolled = np.roll(candidate, shift, axis=0)
            dist = np.linalg.norm(left - rolled, axis=1)
            score = float(np.median(dist))
            if score < best[0]:
                best = (score, shift, reversed_flag, float(np.mean(dist)))
    _, shift, reversed_flag, mean_dist = best
    aligned = np.roll(right[::-1] if reversed_flag else right, shift, axis=0)
    return aligned, {
        'right_shift': int(shift),
        'right_reversed': bool(reversed_flag),
        'mean_border_gap_m': float(mean_dist),
        'median_border_gap_m': float(best[0]),
    }


def load_track_geometry(boundary_json_path: str | Path, n_points: int) -> dict[str, Any]:
    obj = json.loads(Path(boundary_json_path).read_text())
    left = np.asarray(obj['boundaries']['left_border'], dtype=float)
    right = np.asarray(obj['boundaries']['right_border'], dtype=float)
    left_rs = _resample_closed(left, n_points)
    right_rs = _resample_closed(right, n_points)
    right_aligned, alignment = _align_right_to_left(left_rs, right_rs)
    center = 0.5 * (left_rs + right_aligned)
    seg = np.linalg.norm(np.diff(np.vstack([center, center[0]]), axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg[:-1])])
    width = np.linalg.norm(left_rs - right_aligned, axis=1)
    return {
        'left': left_rs,
        'right': right_aligned,
        'center': center,
        's': s,
        'lap_length_m': float(np.sum(seg)),
        'width_m': width,
        'alignment': alignment,
    }


def _nearest_index(x: np.ndarray, y: np.ndarray, center: np.ndarray) -> np.ndarray:
    points = np.column_stack([x, y])
    idx = np.empty(len(points), dtype=int)
    for start in range(0, len(points), 256):
        chunk = points[start:start + 256]
        dist = ((chunk[:, None, :] - center[None, :, :]) ** 2).sum(axis=2)
        idx[start:start + len(chunk)] = np.argmin(dist, axis=1)
    return idx


def annotate_track_progress(telemetry: pd.DataFrame, geometry: dict[str, Any]) -> pd.DataFrame:
    data = telemetry.sort_values('timestamp_ns').reset_index(drop=True).copy()
    n_points = len(geometry['center'])
    idx = pd.to_numeric(data.get('track_idx'), errors='coerce')
    if idx.isna().mean() > 0.5:
        idx = pd.Series(_nearest_index(data['x'].to_numpy(), data['y'].to_numpy(), geometry['center']), index=data.index)
    idx = idx.fillna(0).round().astype(int).clip(0, n_points - 1)
    ds = pd.to_numeric(data.get('track_ds'), errors='coerce').fillna(0.0)
    s_center = geometry['s'][idx.to_numpy()]
    lap_length = float(geometry['lap_length_m'])
    s_mod = np.mod(s_center + ds.to_numpy(), lap_length)
    lap_id = np.zeros(len(data), dtype=int)
    wraps = 0
    for i in range(1, len(data)):
        delta = s_mod[i] - s_mod[i - 1]
        if delta < -0.5 * lap_length:
            wraps += 1
        lap_id[i] = wraps
    data['track_idx_aligned'] = idx
    data['s_mod'] = s_mod
    data['lap_id'] = lap_id
    data['s_total'] = s_mod + lap_id * lap_length
    data['lap_progress'] = s_mod / lap_length
    data['track_x'] = geometry['center'][idx.to_numpy(), 0]
    data['track_y'] = geometry['center'][idx.to_numpy(), 1]
    data['track_width_m'] = geometry['width_m'][idx.to_numpy()]
    data['lateral_offset_m'] = pd.to_numeric(data.get('track_n'), errors='coerce')
    data['inside_track'] = data.get('track_inside', True)
    return data


def save_track_products(track_df: pd.DataFrame, geometry: dict[str, Any], output_dir: str | Path) -> dict[str, str]:
    output_dir = Path(output_dir)
    track_df.to_parquet(output_dir / 'track_samples.parquet', index=False)
    centerline = pd.DataFrame({
        'track_idx': np.arange(len(geometry['center'])),
        's_m': geometry['s'],
        'center_x': geometry['center'][:, 0],
        'center_y': geometry['center'][:, 1],
        'left_x': geometry['left'][:, 0],
        'left_y': geometry['left'][:, 1],
        'right_x': geometry['right'][:, 0],
        'right_y': geometry['right'][:, 1],
        'track_width_m': geometry['width_m'],
    })
    centerline.to_parquet(output_dir / 'centerline.parquet', index=False)
    metadata = {
        'lap_length_m': geometry['lap_length_m'],
        **geometry['alignment'],
    }
    (output_dir / 'track_metadata.json').write_text(json.dumps(metadata, indent=2))
    return {
        'track_samples_path': str(output_dir / 'track_samples.parquet'),
        'centerline_path': str(output_dir / 'centerline.parquet'),
        'track_metadata_path': str(output_dir / 'track_metadata.json'),
    }
