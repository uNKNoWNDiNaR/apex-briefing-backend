from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _lap_length(centerline: pd.DataFrame) -> float:
    ds = np.median(np.diff(centerline['s_m'].to_numpy()))
    return float(centerline['s_m'].iloc[-1] + ds)


def _compute_curvature(centerline: pd.DataFrame) -> np.ndarray:
    xy = centerline[['center_x', 'center_y']].to_numpy()
    prev_xy = np.roll(xy, 1, axis=0)
    next_xy = np.roll(xy, -1, axis=0)
    headings = np.unwrap(np.arctan2(next_xy[:, 1] - prev_xy[:, 1], next_xy[:, 0] - prev_xy[:, 0]))
    ds = np.gradient(centerline['s_m'].to_numpy())
    ds[ds <= 0] = np.median(ds[ds > 0]) if np.any(ds > 0) else 1.0
    return np.gradient(headings) / ds


def _circular_smooth(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return values.copy()
    if window % 2 == 0:
        window += 1
    pad = window // 2
    extended = np.pad(values, pad_width=pad, mode='wrap')
    kernel = np.ones(window, dtype=float) / float(window)
    smoothed = np.convolve(extended, kernel, mode='valid')
    return smoothed[: len(values)]


def _segments_from_mask(mask: np.ndarray) -> list[tuple[int, int, bool]]:
    segments: list[tuple[int, int, bool]] = []
    start = 0
    current = bool(mask[0])
    for idx in range(1, len(mask)):
        flag = bool(mask[idx])
        if flag != current:
            segments.append((start, idx - 1, current))
            start = idx
            current = flag
    segments.append((start, len(mask) - 1, current))
    if len(segments) > 1 and segments[0][2] == segments[-1][2]:
        first = segments.pop(0)
        last = segments.pop(-1)
        segments.insert(0, (last[0], first[1], first[2]))
    return segments


def _span_indices(start_idx: int, end_idx: int, n: int) -> np.ndarray:
    if start_idx <= end_idx:
        return np.arange(start_idx, end_idx + 1)
    return np.concatenate([np.arange(start_idx, n), np.arange(0, end_idx + 1)])


def _span_length_m(start_idx: int, end_idx: int, s: np.ndarray, lap_length: float) -> tuple[float, float, float]:
    start_s = float(s[start_idx])
    end_exclusive_idx = (end_idx + 1) % len(s)
    end_s = float(s[end_exclusive_idx])
    if end_exclusive_idx <= start_idx:
        end_s += lap_length
    return start_s, end_s % lap_length, max(end_s - start_s, 1.0)


def _annotate_mask(mask: np.ndarray, s: np.ndarray, curvature: np.ndarray, curvature_threshold: float, lap_length: float) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for start_idx, end_idx, is_corner in _segments_from_mask(mask):
        idxs = _span_indices(start_idx, end_idx, len(mask))
        start_s, end_s_mod, length_m = _span_length_m(start_idx, end_idx, s, lap_length)
        curv = curvature[idxs]
        rows.append({
            'start_idx': int(start_idx),
            'end_idx': int(end_idx),
            'is_corner': bool(is_corner),
            's_start_m': float(start_s % lap_length),
            's_end_m': float(end_s_mod),
            's_start_unwrapped_m': float(start_s),
            'segment_length_m': float(length_m),
            'mean_abs_curvature': float(np.mean(np.abs(curv))),
            'max_abs_curvature': float(np.max(np.abs(curv))),
            'curvature_threshold': float(curvature_threshold),
        })
    return rows


def _normalize_corner_mask(mask: np.ndarray, s: np.ndarray, curvature: np.ndarray, config: dict[str, Any], lap_length: float) -> list[dict[str, Any]]:
    threshold = max(float(np.quantile(np.abs(curvature), float(config['segmentation']['curvature_quantile']))), float(config['segmentation']['curvature_threshold_min']))
    min_corner_length = float(config['segmentation']['min_corner_length_m'])
    min_straight_length = float(config['segmentation']['min_straight_length_m'])
    min_corner_peak = float(config['segmentation']['min_corner_peak_curvature'])
    merge_curvature_factor = float(config['segmentation']['merge_curvature_factor'])
    for _ in range(10):
        rows = _annotate_mask(mask, s, curvature, threshold, lap_length)
        changed = False
        for idx, row in enumerate(rows):
            prev_row = rows[idx - 1]
            next_row = rows[(idx + 1) % len(rows)]
            indices = _span_indices(row['start_idx'], row['end_idx'], len(mask))
            if row['is_corner']:
                if row['segment_length_m'] < min_corner_length and row['max_abs_curvature'] < min_corner_peak:
                    mask[indices] = False
                    changed = True
            else:
                if prev_row['is_corner'] and next_row['is_corner'] and (
                    row['segment_length_m'] < min_straight_length or row['mean_abs_curvature'] > threshold * merge_curvature_factor
                ):
                    mask[indices] = True
                    changed = True
        if not changed:
            break
    return _annotate_mask(mask, s, curvature, threshold, lap_length)


def _span_unwrapped_positions(indices: np.ndarray, s: np.ndarray, lap_length: float, start_unwrapped: float) -> np.ndarray:
    out = np.empty(len(indices), dtype=float)
    out[0] = start_unwrapped
    offset = start_unwrapped - float(s[indices[0]])
    prev_raw = float(s[indices[0]])
    for i, idx in enumerate(indices[1:], start=1):
        raw = float(s[idx])
        if raw < prev_raw - 0.5 * lap_length:
            offset += lap_length
        out[i] = raw + offset
        prev_raw = raw
    return out


def _find_prominent_peaks(values: np.ndarray, config: dict[str, Any]) -> list[int]:
    if len(values) < 5:
        return []
    peak_floor = max(float(np.quantile(values, float(config['segmentation']['split_peak_quantile']))), float(config['segmentation']['min_corner_peak_curvature']))
    candidates: list[int] = []
    for i in range(1, len(values) - 1):
        if values[i] > values[i - 1] and values[i] >= values[i + 1] and values[i] >= peak_floor:
            candidates.append(i)
    if not candidates:
        return []
    min_sep = int(config['segmentation']['split_min_peak_separation_points'])
    merged: list[int] = []
    for idx in candidates:
        if not merged or idx - merged[-1] >= min_sep:
            merged.append(idx)
        elif values[idx] > values[merged[-1]]:
            merged[-1] = idx
    return merged


def _split_complex_corner_rows(rows: list[dict[str, Any]], s: np.ndarray, curvature: np.ndarray, config: dict[str, Any], lap_length: float) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    min_complex = float(config['segmentation']['split_complex_min_length_m'])
    valley_ratio = float(config['segmentation']['split_valley_ratio'])
    min_subcorner = float(config['segmentation']['split_min_subcorner_length_m'])
    for row in rows:
        if not row['is_corner'] or row['segment_length_m'] < min_complex:
            out.append(row)
            continue
        indices = _span_indices(row['start_idx'], row['end_idx'], len(s))
        vals = curvature[indices]
        peaks = _find_prominent_peaks(vals, config)
        if len(peaks) < 2:
            out.append(row)
            continue
        unwrapped = _span_unwrapped_positions(indices, s, lap_length, row['s_start_unwrapped_m'])
        split_rel_points: list[int] = []
        for left_peak, right_peak in zip(peaks[:-1], peaks[1:]):
            trough_rel = int(left_peak + np.argmin(vals[left_peak:right_peak + 1]))
            trough_val = float(vals[trough_rel])
            if trough_val > valley_ratio * min(float(vals[left_peak]), float(vals[right_peak])):
                continue
            left_len = unwrapped[trough_rel] - unwrapped[0 if not split_rel_points else split_rel_points[-1]]
            right_len = unwrapped[-1] - unwrapped[trough_rel]
            if left_len < min_subcorner or right_len < min_subcorner:
                continue
            split_rel_points.append(trough_rel)
        if not split_rel_points:
            out.append(row)
            continue
        boundaries = [0] + split_rel_points + [len(indices) - 1]
        built: list[dict[str, Any]] = []
        valid = True
        for start_rel, end_rel in zip(boundaries[:-1], boundaries[1:]):
            start_idx = int(indices[start_rel])
            end_idx = int(indices[end_rel])
            start_s = float(unwrapped[start_rel])
            end_s = float(unwrapped[end_rel])
            length_m = max(end_s - start_s, 1.0)
            if length_m < min_subcorner:
                valid = False
                break
            curv_slice = vals[start_rel:end_rel + 1]
            built.append({
                'start_idx': start_idx,
                'end_idx': end_idx,
                'is_corner': True,
                's_start_m': float(start_s % lap_length),
                's_end_m': float(end_s % lap_length),
                's_start_unwrapped_m': float(start_s),
                'segment_length_m': float(length_m),
                'mean_abs_curvature': float(np.mean(np.abs(curv_slice))),
                'max_abs_curvature': float(np.max(np.abs(curv_slice))),
                'curvature_threshold': row['curvature_threshold'],
            })
        if valid and len(built) >= 2:
            out.extend(built)
        else:
            out.append(row)
    return out


def _relative_window(track_df: pd.DataFrame, start_s: float, segment_length_m: float, lap_length: float, approach_cap: float) -> pd.DataFrame:
    rel_unwrapped = track_df['s_total'].to_numpy() - float(start_s)
    pass_id = np.floor((rel_unwrapped + 0.5 * lap_length) / lap_length).astype(int)
    local_rel = rel_unwrapped - pass_id * lap_length
    mask = (local_rel >= -approach_cap) & (local_rel <= segment_length_m + 60.0)
    window = track_df.loc[mask, ['timestamp_ns', 's_mod', 's_total', 'speed', 'brake', 'throttle']].copy()
    window['pass_id'] = pass_id[mask]
    window['local_rel_m'] = local_rel[mask]
    return window.sort_values(['pass_id', 'local_rel_m']).reset_index(drop=True)


def _infer_corner_events(track_df: pd.DataFrame, start_s: float, segment_length_m: float, lap_length: float, config: dict[str, Any]) -> dict[str, float]:
    approach_cap = float(config['segmentation']['brake_approach_cap_m'])
    brake_threshold = float(config['segmentation']['brake_threshold'])
    throttle_threshold = float(config['segmentation']['throttle_pickup_threshold'])
    apex_window_min = float(config['segmentation']['apex_window_min_m'])
    apex_window_max = float(config['segmentation']['apex_window_max_m'])
    window = _relative_window(track_df, start_s, segment_length_m, lap_length, approach_cap)
    brake_onsets = []
    brake_releases = []
    apex_positions = []
    throttle_pickups = []
    for _, group in window.groupby('pass_id'):
        corner = group[(group['local_rel_m'] >= 0.0) & (group['local_rel_m'] <= segment_length_m)]
        if len(corner) < 8:
            continue
        apex_idx = corner['speed'].astype(float).idxmin()
        apex_rel = float(corner.loc[apex_idx, 'local_rel_m'])
        apex_positions.append(apex_rel)
        brake_window = group[(group['local_rel_m'] <= apex_rel) & (group['brake'].astype(float) > brake_threshold)]
        if not brake_window.empty:
            brake_onsets.append(float(brake_window['local_rel_m'].iloc[0]))
            brake_releases.append(float(brake_window['local_rel_m'].iloc[-1]))
        throttle_window = corner[(corner['local_rel_m'] >= apex_rel) & (corner['throttle'].astype(float) > throttle_threshold)]
        if not throttle_window.empty:
            throttle_pickups.append(float(throttle_window['local_rel_m'].iloc[0]))
    apex_rel = float(np.median(apex_positions)) if apex_positions else 0.5 * segment_length_m
    if brake_onsets:
        brake_on_rel = float(np.median(brake_onsets))
        brake_release_rel = float(np.median(brake_releases)) if brake_releases else 0.0
    else:
        brake_on_rel = 0.0
        brake_release_rel = 0.0
    brake_release_rel = float(np.clip(brake_release_rel, 0.0, max(apex_rel - 4.0, 0.0)))
    throttle_rel = float(np.median(throttle_pickups)) if throttle_pickups else apex_rel + max(8.0, 0.20 * segment_length_m)
    throttle_rel = float(np.clip(throttle_rel, apex_rel + 4.0, segment_length_m))
    apex_width = float(np.clip(0.18 * segment_length_m, apex_window_min, apex_window_max))
    apex_start_rel = float(np.clip(apex_rel - 0.5 * apex_width, brake_release_rel + 1.0, max(segment_length_m - 6.0, 0.0)))
    apex_end_rel = float(np.clip(apex_rel + 0.5 * apex_width, apex_start_rel + 4.0, segment_length_m))
    exit_start_rel = float(max(apex_end_rel, throttle_rel))
    return {
        'brake_zone_start_rel_m': brake_on_rel,
        'entry_start_rel_m': brake_release_rel,
        'apex_rel_m': apex_rel,
        'apex_start_rel_m': apex_start_rel,
        'apex_end_rel_m': apex_end_rel,
        'exit_start_rel_m': exit_start_rel,
    }


def _attach_corner_phase_metadata(rows: list[dict[str, Any]], track_df: pd.DataFrame, lap_length: float, config: dict[str, Any]) -> list[dict[str, Any]]:
    for row in rows:
        if not row['is_corner']:
            row.update({
                'brake_zone_start_m': row['s_start_m'],
                'entry_start_s_m': row['s_start_m'],
                'apex_s_m': row['s_start_m'],
                'apex_start_s_m': row['s_start_m'],
                'apex_end_s_m': row['s_end_m'],
                'exit_start_s_m': row['s_end_m'],
            })
            continue
        inferred = _infer_corner_events(track_df, row['s_start_unwrapped_m'], row['segment_length_m'], lap_length, config)
        row['brake_zone_start_m'] = float((row['s_start_unwrapped_m'] + inferred['brake_zone_start_rel_m']) % lap_length)
        row['entry_start_s_m'] = float((row['s_start_unwrapped_m'] + inferred['entry_start_rel_m']) % lap_length)
        row['apex_s_m'] = float((row['s_start_unwrapped_m'] + inferred['apex_rel_m']) % lap_length)
        row['apex_start_s_m'] = float((row['s_start_unwrapped_m'] + inferred['apex_start_rel_m']) % lap_length)
        row['apex_end_s_m'] = float((row['s_start_unwrapped_m'] + inferred['apex_end_rel_m']) % lap_length)
        row['exit_start_s_m'] = float((row['s_start_unwrapped_m'] + inferred['exit_start_rel_m']) % lap_length)
    return rows


def _finalize_segment_rows(rows: list[dict[str, Any]]) -> pd.DataFrame:
    frame_rows = []
    corner_count = 0
    straight_count = 0
    for row in rows:
        segment_type = 'corner' if row['is_corner'] else 'straight'
        if row['is_corner']:
            corner_count += 1
            name = f'C{corner_count:02d}'
        else:
            straight_count += 1
            name = f'S{straight_count:02d}'
        frame_rows.append({'segment_name': name, 'segment_type': segment_type, **row})
    segments = pd.DataFrame(frame_rows)
    segments.insert(0, 'segment_id', np.arange(1, len(segments) + 1))
    segments['s_end_unwrapped_m'] = segments['s_start_unwrapped_m'] + segments['segment_length_m']
    prev_lengths = []
    next_lengths = []
    for idx, row in segments.iterrows():
        prev_row = segments.iloc[idx - 1]
        next_row = segments.iloc[(idx + 1) % len(segments)]
        prev_lengths.append(float(prev_row['segment_length_m']) if prev_row['segment_type'] == 'straight' else 0.0)
        next_lengths.append(float(next_row['segment_length_m']) if next_row['segment_type'] == 'straight' else 0.0)
    segments['previous_straight_length_m'] = prev_lengths
    segments['next_straight_length_m'] = next_lengths
    keep = [
        'segment_id', 'segment_name', 'segment_type', 'start_idx', 'end_idx',
        's_start_m', 's_end_m', 's_start_unwrapped_m', 's_end_unwrapped_m',
        'segment_length_m', 'mean_abs_curvature', 'max_abs_curvature',
        'brake_zone_start_m', 'entry_start_s_m', 'apex_s_m', 'apex_start_s_m',
        'apex_end_s_m', 'exit_start_s_m', 'previous_straight_length_m', 'next_straight_length_m',
    ]
    return segments[keep]


def build_segments(track_df: pd.DataFrame, centerline: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    lap_length = _lap_length(centerline)
    curvature_raw = np.abs(_compute_curvature(centerline))
    curvature = _circular_smooth(curvature_raw, int(config['segmentation']['curvature_smoothing_window_points']))
    threshold = max(float(np.quantile(curvature, float(config['segmentation']['curvature_quantile']))), float(config['segmentation']['curvature_threshold_min']))
    mask = curvature >= threshold
    gap_points = int(config['segmentation']['merge_gap_points'])
    start = 0
    while start < len(mask):
        if mask[start]:
            start += 1
            continue
        end = start
        while end < len(mask) and not mask[end]:
            end += 1
        if start > 0 and end < len(mask) and (end - start) <= gap_points:
            mask[start:end] = True
        start = end
    rows = _normalize_corner_mask(mask, centerline['s_m'].to_numpy(), curvature, config, lap_length)
    rows = _split_complex_corner_rows(rows, centerline['s_m'].to_numpy(), curvature, config, lap_length)
    rows = _attach_corner_phase_metadata(rows, track_df, lap_length, config)
    rows = sorted(rows, key=lambda row: row['s_start_unwrapped_m'])
    return _finalize_segment_rows(rows)


def _interval_mask(s: np.ndarray, start: float, end: float, lap_length: float) -> np.ndarray:
    s = np.mod(s, lap_length)
    start = start % lap_length
    end = end % lap_length
    if start <= end:
        return (s >= start) & (s < end)
    return (s >= start) | (s < end)


def apply_segments(track_df: pd.DataFrame, segments: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    data = track_df.copy()
    lap_length = float(data['s_mod'].max() + np.median(np.diff(np.sort(data['s_mod'].unique()))))
    data['segment_id'] = -1
    data['segment_name'] = 'UNKNOWN'
    data['segment_type'] = 'unknown'
    data['track_phase'] = 'unassigned'
    s_vals = data['s_mod'].to_numpy()
    for row in segments.itertuples(index=False):
        if row.segment_type == 'straight':
            mask = _interval_mask(s_vals, row.s_start_m, row.s_end_m, lap_length)
            data.loc[mask, 'segment_id'] = row.segment_id
            data.loc[mask, 'segment_name'] = row.segment_name
            data.loc[mask, 'segment_type'] = row.segment_type
            data.loc[mask, 'track_phase'] = 'straight'
            continue
        braking_mask = _interval_mask(s_vals, row.brake_zone_start_m, row.entry_start_s_m, lap_length)
        entry_mask = _interval_mask(s_vals, row.entry_start_s_m, row.apex_start_s_m, lap_length)
        apex_mask = _interval_mask(s_vals, row.apex_start_s_m, row.apex_end_s_m, lap_length)
        exit_mask = _interval_mask(s_vals, row.apex_end_s_m, row.s_end_m, lap_length)
        full_mask = braking_mask | entry_mask | apex_mask | exit_mask
        data.loc[full_mask, 'segment_id'] = row.segment_id
        data.loc[full_mask, 'segment_name'] = row.segment_name
        data.loc[full_mask, 'segment_type'] = row.segment_type
        data.loc[braking_mask, 'track_phase'] = 'braking'
        data.loc[entry_mask, 'track_phase'] = 'entry'
        data.loc[apex_mask, 'track_phase'] = 'apex'
        data.loc[exit_mask, 'track_phase'] = 'exit'
    data.loc[data['segment_id'] < 0, 'track_phase'] = 'straight'
    return data
