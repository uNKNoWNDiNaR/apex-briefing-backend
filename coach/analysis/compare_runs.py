
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _prepare_interp(df: pd.DataFrame, x_col: str, value_col: str) -> tuple[np.ndarray, np.ndarray]:
    temp = df[[x_col, value_col]].dropna().sort_values(x_col).drop_duplicates(x_col, keep='last')
    return temp[x_col].to_numpy(), temp[value_col].to_numpy()


def _interval_mask(s: np.ndarray, start: float, end: float, lap_length: float) -> np.ndarray:
    s = np.mod(s, lap_length)
    start = start % lap_length
    end = end % lap_length
    if start <= end:
        return (s >= start) & (s < end)
    return (s >= start) | (s < end)


def build_overlay(target_lap: pd.DataFrame, reference_lap: pd.DataFrame, segments_df: pd.DataFrame, lap_length: float, step_m: float) -> pd.DataFrame:
    grid = np.arange(0.0, lap_length, step_m)
    target_elapsed = target_lap['timestamp'] - float(target_lap['timestamp'].iloc[0])
    reference_elapsed = reference_lap['timestamp'] - float(reference_lap['timestamp'].iloc[0])
    target_interp = target_lap.assign(elapsed_s=target_elapsed)
    reference_interp = reference_lap.assign(elapsed_s=reference_elapsed)
    out = pd.DataFrame({'s_m': grid})
    for prefix, lap in [('target', target_interp), ('reference', reference_interp)]:
        for value_col in ['elapsed_s', 'speed', 'brake', 'throttle', 'steering']:
            xp, fp = _prepare_interp(lap, 'lap_s_m', value_col)
            out[f'{prefix}_{value_col}'] = np.interp(grid, xp, fp)
    out['delta_time_s'] = out['target_elapsed_s'] - out['reference_elapsed_s']
    out['segment_name'] = ''
    out['segment_id'] = -1
    out['segment_type'] = 'straight'
    out['phase'] = 'straight'
    for row in segments_df.itertuples(index=False):
        full_mask = _interval_mask(out['s_m'].to_numpy(), row.brake_zone_start_m if row.segment_type == 'corner' else row.s_start_m, row.s_end_m, lap_length)
        out.loc[full_mask, 'segment_name'] = row.segment_name
        out.loc[full_mask, 'segment_id'] = row.segment_id
        out.loc[full_mask, 'segment_type'] = row.segment_type
        if row.segment_type == 'straight':
            out.loc[full_mask, 'phase'] = 'straight'
            continue
        braking_mask = _interval_mask(out['s_m'].to_numpy(), row.brake_zone_start_m, row.entry_start_s_m, lap_length)
        entry_mask = _interval_mask(out['s_m'].to_numpy(), row.entry_start_s_m, row.apex_start_s_m, lap_length)
        apex_mask = _interval_mask(out['s_m'].to_numpy(), row.apex_start_s_m, row.apex_end_s_m, lap_length)
        exit_mask = _interval_mask(out['s_m'].to_numpy(), row.apex_end_s_m, row.s_end_m, lap_length)
        out.loc[braking_mask, 'phase'] = 'braking'
        out.loc[entry_mask, 'phase'] = 'entry'
        out.loc[apex_mask, 'phase'] = 'apex'
        out.loc[exit_mask, 'phase'] = 'exit'
    return out


def compare_segment_features(target_features: pd.DataFrame, reference_features: pd.DataFrame) -> pd.DataFrame:
    keys = ['segment_id', 'segment_name', 'segment_type', 'phase']
    merged = target_features.merge(reference_features, on=keys, suffixes=('_target', '_reference'))
    merged['time_loss_s'] = merged['time_s_target'] - merged['time_s_reference']
    merged['braking_point_delta_m'] = merged['brake_start_s_target'] - merged['brake_start_s_reference']
    merged['brake_release_delta_m'] = merged['brake_release_s_target'] - merged['brake_release_s_reference']
    merged['min_speed_delta_mps'] = merged['min_speed_mps_target'] - merged['min_speed_mps_reference']
    merged['exit_speed_delta_mps'] = merged['end_speed_mps_target'] - merged['end_speed_mps_reference']
    merged['throttle_pickup_delta_m'] = merged['throttle_pickup_s_target'] - merged['throttle_pickup_s_reference']
    merged['throttle_smoothness_delta'] = merged['throttle_smoothness_target'] - merged['throttle_smoothness_reference']
    merged['steering_smoothness_ratio'] = merged['steering_smoothness_target'] / merged['steering_smoothness_reference'].replace(0, np.nan)
    merged['mean_rpm_delta'] = merged['mean_rpm_target'] - merged['mean_rpm_reference']
    merged['mean_gear_delta'] = merged['mean_gear_target'] - merged['mean_gear_reference']
    merged['mean_slip_abs_delta'] = merged['mean_slip_abs_target'] - merged['mean_slip_abs_reference']
    merged['line_offset_delta_m'] = merged['mean_abs_lateral_offset_m_target'] - merged['mean_abs_lateral_offset_m_reference']
    return merged


def build_comparison_summary(target_session: str, reference_session: str, target_lap_id: int, reference_lap_id: int, lap_summary_target: pd.Series, lap_summary_reference: pd.Series, segment_cmp: pd.DataFrame) -> dict[str, Any]:
    total_loss = float(segment_cmp.loc[segment_cmp['phase'] == 'segment_total', 'time_loss_s'].sum()) if not segment_cmp.empty else 0.0
    phase_breakdown = {}
    phase_normalized = {}
    if not segment_cmp.empty:
        non_total = segment_cmp[segment_cmp['phase'] != 'segment_total'].copy()
        grouped = non_total.groupby('phase')['time_loss_s'].sum().sort_values(ascending=False)
        phase_breakdown = {str(k): float(v) for k, v in grouped.items()}
        positive_grouped = non_total.assign(positive_time_loss_s=non_total['time_loss_s'].clip(lower=0.0)).groupby('phase')['positive_time_loss_s'].sum()
        positive_total = float(positive_grouped.sum())
        if positive_total > 1e-6:
            phase_normalized = {str(k): float(v / positive_total) for k, v in positive_grouped.sort_values(ascending=False).items()}
    return {
        'target_session': target_session,
        'reference_session': reference_session,
        'target_lap_id': int(target_lap_id),
        'reference_lap_id': int(reference_lap_id),
        'target_lap_time_s': float(lap_summary_target['lap_time_s']),
        'reference_lap_time_s': float(lap_summary_reference['lap_time_s']),
        'lap_time_delta_s': float(lap_summary_target['lap_time_s'] - lap_summary_reference['lap_time_s']),
        'segment_time_loss_total_s': total_loss,
        'phase_time_loss_s': phase_breakdown,
        'phase_time_loss_normalized': phase_normalized,
    }
