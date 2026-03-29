from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def build_lap_windows(track_df: pd.DataFrame, lap_length: float, config: dict[str, Any]) -> list[tuple[int, float, float, pd.DataFrame]]:
    data = track_df.sort_values('timestamp_ns').reset_index(drop=True)
    if data.empty:
        return []
    start_anchor = float(data['s_total'].iloc[0])
    end_total = float(data['s_total'].iloc[-1])
    window_count = int((end_total - start_anchor) // lap_length)
    windows: list[tuple[int, float, float, pd.DataFrame]] = []
    for lap_id in range(window_count):
        lap_start = start_anchor + lap_id * lap_length
        lap_end = lap_start + lap_length
        group = data[(data['s_total'] >= lap_start) & (data['s_total'] <= lap_end)].copy()
        if group.empty:
            continue
        group['lap_id'] = lap_id
        group['lap_s_m'] = group['s_total'] - lap_start
        windows.append((lap_id, lap_start, lap_end, group))
    return windows


def summarize_laps(track_df: pd.DataFrame, lap_length: float, config: dict[str, Any]) -> pd.DataFrame:
    rows = []
    min_coverage = float(config['track']['min_complete_lap_coverage'])
    for lap_id, lap_start, lap_end, group in build_lap_windows(track_df, lap_length, config):
        coverage = float(group['lap_s_m'].max() - group['lap_s_m'].min())
        coverage_ratio = coverage / lap_length if lap_length else 0.0
        rows.append({
            'lap_id': int(lap_id),
            'lap_start_s_total': float(lap_start),
            'lap_end_s_total': float(lap_end),
            'lap_start_timestamp': float(group['timestamp'].iloc[0]),
            'lap_end_timestamp': float(group['timestamp'].iloc[-1]),
            'lap_time_s': float(group['timestamp'].iloc[-1] - group['timestamp'].iloc[0]),
            'sample_count': int(len(group)),
            'coverage_ratio': float(coverage_ratio),
            'is_complete': bool(coverage_ratio >= min_coverage and len(group) > 1000),
            'mean_speed_mps': float(group['speed'].mean()),
            'max_speed_mps': float(group['speed'].max()),
        })
    return pd.DataFrame(rows)


def _first_s(group: pd.DataFrame, mask: pd.Series) -> float | None:
    subset = group.loc[mask]
    if subset.empty:
        return None
    return float(subset['s_mod'].iloc[0])


def _last_s(group: pd.DataFrame, mask: pd.Series) -> float | None:
    subset = group.loc[mask]
    if subset.empty:
        return None
    return float(subset['s_mod'].iloc[-1])


def _primary_run(group: pd.DataFrame, max_gap_s: float) -> tuple[pd.DataFrame, float]:
    group = group.sort_values('timestamp_ns').reset_index(drop=True)
    if group.empty:
        return group, 0.0
    dt = group['timestamp'].diff().fillna(0.0)
    run_id = (dt > max_gap_s).cumsum()
    group = group.assign(_run_id=run_id)
    best_run = None
    best_duration = -1.0
    total_duration = 0.0
    for _, run in group.groupby('_run_id'):
        duration = float(run['timestamp'].diff().clip(lower=0.0, upper=max_gap_s).sum())
        total_duration += duration
        if duration > best_duration:
            best_run = run.copy()
            best_duration = duration
    primary = best_run.drop(columns='_run_id') if best_run is not None else group.drop(columns='_run_id')
    return primary.reset_index(drop=True), total_duration


def _compute_feature_row(group: pd.DataFrame, brake_threshold: float, throttle_threshold: float, contiguous_gap_s: float) -> dict[str, Any]:
    group = group.sort_values('timestamp_ns').reset_index(drop=True)
    primary, duration_s = _primary_run(group, contiguous_gap_s)
    brake_mask = pd.to_numeric(primary['brake'], errors='coerce').fillna(0.0) > brake_threshold
    throttle_mask = pd.to_numeric(primary['throttle'], errors='coerce').fillna(0.0) > throttle_threshold
    throttle_diff = np.abs(np.diff(group['throttle'].ffill().fillna(0.0).to_numpy()))
    steering_diff = np.abs(np.diff(group['steering'].ffill().fillna(0.0).to_numpy()))
    slip_cols = [col for col in ['slip_ratio_fl', 'slip_ratio_fr', 'slip_ratio_rl', 'slip_ratio_rr'] if col in group.columns]
    tyre_cols = [col for col in ['tyre_temp_fl', 'tyre_temp_fr', 'tyre_temp_rl', 'tyre_temp_rr'] if col in group.columns]
    brake_temp_cols = [col for col in ['brake_temp_fl', 'brake_temp_fr', 'brake_temp_rl', 'brake_temp_rr'] if col in group.columns]
    lateral = pd.to_numeric(group.get('lateral_offset_m'), errors='coerce')
    primary_speed_idx = primary['speed'].astype(float).idxmin() if not primary.empty else None
    primary_brake_idx = primary['brake'].astype(float).idxmax() if not primary.empty else None
    return {
        'start_timestamp': float(primary['timestamp'].iloc[0]),
        'end_timestamp': float(primary['timestamp'].iloc[-1]),
        'time_s': float(duration_s),
        'sample_count': int(len(group)),
        'mean_speed_mps': float(group['speed'].mean()),
        'min_speed_mps': float(group['speed'].min()),
        'max_speed_mps': float(group['speed'].max()),
        'start_speed_mps': float(primary['speed'].iloc[0]),
        'end_speed_mps': float(primary['speed'].iloc[-1]),
        'mean_brake': float(group['brake'].mean()),
        'max_brake': float(group['brake'].max()),
        'mean_throttle': float(group['throttle'].mean()),
        'max_throttle': float(group['throttle'].max()),
        'throttle_smoothness': float(throttle_diff.mean()) if len(throttle_diff) else 0.0,
        'steering_rms': float(np.sqrt(np.mean(np.square(group['steering'].to_numpy())))),
        'steering_smoothness': float(steering_diff.mean()) if len(steering_diff) else 0.0,
        'mean_gear': float(group['gear'].mean()),
        'mean_rpm': float(group['rpm'].mean()),
        'max_rpm': float(group['rpm'].max()),
        'mean_slip_abs': float(group[slip_cols].abs().mean().mean()) if slip_cols else np.nan,
        'mean_tyre_temp_c': float(group[tyre_cols].mean().mean()) if tyre_cols else np.nan,
        'max_brake_temp_c': float(group[brake_temp_cols].max().max()) if brake_temp_cols else np.nan,
        'mean_abs_lateral_offset_m': float(lateral.abs().mean()) if lateral is not None and not lateral.dropna().empty else np.nan,
        'max_abs_lateral_offset_m': float(lateral.abs().max()) if lateral is not None and not lateral.dropna().empty else np.nan,
        'lateral_offset_std_m': float(lateral.std()) if lateral is not None and not lateral.dropna().empty else np.nan,
        'brake_start_s': _first_s(primary, brake_mask),
        'brake_release_s': _last_s(primary, brake_mask),
        'throttle_pickup_s': _first_s(primary, throttle_mask),
        'min_speed_s': None if primary_speed_idx is None else float(primary.loc[primary_speed_idx, 's_mod']),
        'peak_brake_s': None if primary_brake_idx is None else float(primary.loc[primary_brake_idx, 's_mod']),
    }


def extract_features(track_df: pd.DataFrame, segments_df: pd.DataFrame, lap_length: float, config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    laps = summarize_laps(track_df, lap_length, config)
    complete_laps = laps[laps['is_complete']]
    rows = []
    brake_threshold = float(config['segmentation']['brake_threshold'])
    throttle_threshold = float(config['segmentation']['throttle_pickup_threshold'])
    contiguous_gap_s = float(config['features']['contiguous_gap_s'])
    windows = {lap_id: group for lap_id, _, _, group in build_lap_windows(track_df, lap_length, config)}
    for lap_id in complete_laps['lap_id'].tolist():
        lap_group = windows[int(lap_id)].sort_values('timestamp_ns').copy()
        for (segment_id, phase), group in lap_group.groupby(['segment_id', 'track_phase'], sort=False):
            if segment_id < 0 or group.empty:
                continue
            info = segments_df.loc[segments_df['segment_id'] == segment_id].iloc[0]
            metrics = _compute_feature_row(group, brake_threshold, throttle_threshold, contiguous_gap_s)
            rows.append({
                'lap_id': int(lap_id),
                'segment_id': int(segment_id),
                'segment_name': info['segment_name'],
                'segment_type': info['segment_type'],
                'phase': str(phase),
                'segment_length_m': float(info['segment_length_m']),
                'brake_zone_start_m': float(info['brake_zone_start_m']),
                'entry_start_s_m': float(info['entry_start_s_m']),
                'apex_s_m': float(info['apex_s_m']),
                'apex_start_s_m': float(info['apex_start_s_m']),
                'apex_end_s_m': float(info['apex_end_s_m']),
                'exit_start_s_m': float(info['exit_start_s_m']),
                'previous_straight_length_m': float(info['previous_straight_length_m']),
                'next_straight_length_m': float(info['next_straight_length_m']),
                **metrics,
            })
        for segment_id, group in lap_group.groupby('segment_id', sort=False):
            if segment_id < 0 or group.empty:
                continue
            info = segments_df.loc[segments_df['segment_id'] == segment_id].iloc[0]
            metrics = _compute_feature_row(group, brake_threshold, throttle_threshold, contiguous_gap_s)
            rows.append({
                'lap_id': int(lap_id),
                'segment_id': int(segment_id),
                'segment_name': info['segment_name'],
                'segment_type': info['segment_type'],
                'phase': 'segment_total',
                'segment_length_m': float(info['segment_length_m']),
                'brake_zone_start_m': float(info['brake_zone_start_m']),
                'entry_start_s_m': float(info['entry_start_s_m']),
                'apex_s_m': float(info['apex_s_m']),
                'apex_start_s_m': float(info['apex_start_s_m']),
                'apex_end_s_m': float(info['apex_end_s_m']),
                'exit_start_s_m': float(info['exit_start_s_m']),
                'previous_straight_length_m': float(info['previous_straight_length_m']),
                'next_straight_length_m': float(info['next_straight_length_m']),
                **metrics,
            })
    return laps, pd.DataFrame(rows)


def get_lap_window(track_df: pd.DataFrame, lap_length: float, config: dict[str, Any], lap_id: int) -> pd.DataFrame:
    for candidate_lap_id, _, _, group in build_lap_windows(track_df, lap_length, config):
        if int(candidate_lap_id) == int(lap_id):
            return group.sort_values('timestamp_ns').reset_index(drop=True)
    raise KeyError(f'Lap window {lap_id} not found')


def select_reference_lap(lap_summary: pd.DataFrame) -> int:
    complete = lap_summary[lap_summary['is_complete']]
    if complete.empty:
        raise RuntimeError('No complete laps available for reference selection')
    return int(complete.sort_values('lap_time_s').iloc[0]['lap_id'])
