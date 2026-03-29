
from __future__ import annotations

from typing import Any

import pandas as pd


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _label_from_score(score: float, high_cut: float, medium_cut: float) -> str:
    if score >= high_cut:
        return 'high'
    if score >= medium_cut:
        return 'medium'
    return 'low'


def _trait_from_rows(
    rows: pd.DataFrame,
    trait_id: str,
    label: str,
    description: str,
    total_lap_loss: float,
    metric_name: str,
    threshold: float,
    summary_template: str,
    available_count: int,
) -> dict[str, Any]:
    if rows.empty or available_count <= 0:
        return {
            'trait_id': trait_id,
            'label': label,
            'description': description,
            'score': 0.0,
            'strength': 'low',
            'confidence': 'low',
            'repeat_ratio': 0.0,
            'time_share': 0.0,
            'affected_segments': [],
            'evidence': {},
            'summary': f'No repeat signal for {label.lower()}.',
        }
    affected_segments = sorted(rows['segment_name'].astype(str).unique().tolist())
    repeat_ratio = len(affected_segments) / float(max(available_count, 1))
    positive_loss = float(rows['time_loss_s'].clip(lower=0.0).sum())
    time_share = positive_loss / total_lap_loss if total_lap_loss > 1e-6 else 0.0
    margin = float((rows[metric_name].abs() / threshold).clip(upper=2.0).mean() / 2.0)
    score = _clamp(0.45 * repeat_ratio + 0.35 * time_share + 0.20 * margin)
    metric_mean = float(rows[metric_name].mean())
    strength = _label_from_score(score, 0.55, 0.30)
    confidence = _label_from_score(_clamp(0.55 * score + 0.45 * min(1.0, len(affected_segments) / 3.0)), 0.65, 0.40)
    return {
        'trait_id': trait_id,
        'label': label,
        'description': description,
        'score': round(score, 3),
        'strength': strength,
        'confidence': confidence,
        'repeat_ratio': round(repeat_ratio, 3),
        'time_share': round(time_share, 3),
        'affected_segments': affected_segments,
        'evidence': {
            'segment_count': len(affected_segments),
            'mean_metric': round(metric_mean, 3),
            'positive_time_loss_s': round(positive_loss, 3),
        },
        'summary': summary_template.format(segment_count=len(affected_segments), metric_value=round(abs(metric_mean), 1)),
    }


def _build_racecraft_profile(
    racecraft_cards: list[dict[str, Any]],
    racecraft_summary: dict[str, Any],
    summary: dict[str, Any],
) -> dict[str, Any]:
    label_map = {
        'defend_entry': ('Too defensive on entry', 'Braking defensively too early gives away rotation and exit speed.'),
        'late_attack': ('Late attack compromise', 'Late attack attempts are costing the exit.'),
        'attack_exit': ('Missed attack window', 'Exit hesitation is preventing a clean run onto the straight.'),
        'line_compromise': ('Line compromise under pressure', 'Defensive or compromised line choices are costing speed through the corner.'),
    }
    total_racecraft = max(float(racecraft_summary.get('racecraft_time_loss_s', 0.0)), 1e-6)
    grouped: dict[str, dict[str, Any]] = {}
    for card in racecraft_cards:
        category = str(card['category'])
        info = grouped.setdefault(category, {'time_loss_s': 0.0, 'segments': [], 'cards': [], 'expected_gain_s': 0.0})
        info['time_loss_s'] += float(card.get('time_loss_s', 0.0))
        if card['segment_name'] not in info['segments']:
            info['segments'].append(card['segment_name'])
        info['cards'].append(card['card_id'])
        info['expected_gain_s'] = max(info['expected_gain_s'], float(card.get('expected_gain_s', 0.0)))
    traits = []
    for category, info in grouped.items():
        label, description = label_map.get(category, ('Racecraft issue', 'Racecraft compromise detected.'))
        score = _clamp(info['time_loss_s'] / total_racecraft)
        traits.append({
            'trait_id': category,
            'label': label,
            'description': description,
            'score': round(score, 3),
            'strength': _label_from_score(score, 0.55, 0.30),
            'confidence': 'high' if len(info['segments']) >= 1 else 'medium',
            'repeat_ratio': round(len(info['segments']) / float(max(1, len(grouped))), 3),
            'time_share': round(info['time_loss_s'] / total_racecraft, 3),
            'affected_segments': sorted(info['segments']),
            'evidence': {
                'segment_count': len(info['segments']),
                'positive_time_loss_s': round(info['time_loss_s'], 3),
                'expected_gain_s': round(info['expected_gain_s'], 3),
            },
            'summary': f'{label} shows up in {len(info["segments"])} segments and explains about {round(info["time_loss_s"], 2)} s of racecraft loss.',
        })
    traits.sort(key=lambda item: (-float(item['score']), item['trait_id']))
    dominant_traits = traits[:3]
    return {
        'profile_version': '1.0',
        'target_session': summary['target_session'],
        'reference_session': summary['reference_session'],
        'lap_time_delta_s': float(summary['lap_time_delta_s']),
        'dominant_traits': dominant_traits,
        'traits': traits,
        'trait_count': len(traits),
        'top_card_titles': [card['title'] for card in racecraft_cards[:5]],
        'profile_summary': [trait['label'] for trait in dominant_traits],
        'profile_mode': 'racecraft',
    }


def build_driver_profile(
    segment_cmp: pd.DataFrame,
    summary: dict[str, Any],
    cards: list[dict[str, Any]],
    config: dict[str, Any],
    racecraft_cards: list[dict[str, Any]] | None = None,
    racecraft_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if summary.get('target_session') == 'wheel_to_wheel' and racecraft_cards and racecraft_summary and float(racecraft_summary.get('racecraft_time_loss_s', 0.0)) > max(1.0, 2.0 * float(racecraft_summary.get('pace_like_time_loss_s', 0.0))):
        return _build_racecraft_profile(racecraft_cards, racecraft_summary, summary)

    phases = segment_cmp[segment_cmp['phase'] != 'segment_total'].copy()
    total_lap_loss = max(float(summary.get('lap_time_delta_s', 0.0)), 0.0)
    comparison_cfg = config['comparison']
    steering_threshold = float(comparison_cfg['steering_smoothness_ratio_threshold'])
    brake_threshold = float(comparison_cfg['brake_point_delta_threshold_m'])
    brake_release_threshold = float(comparison_cfg['brake_release_delta_threshold_m'])
    apex_threshold = float(comparison_cfg['apex_speed_delta_threshold_mps'])
    exit_threshold = float(comparison_cfg['exit_speed_delta_threshold_mps'])
    throttle_threshold = float(comparison_cfg['throttle_pickup_delta_threshold_m'])

    braking_rows = phases[phases['phase'].isin(['braking', 'entry'])]
    apex_rows = phases[phases['phase'].isin(['entry', 'apex'])]
    exit_rows = phases[phases['phase'].isin(['exit', 'straight'])]
    control_rows = phases[phases['phase'].isin(['entry', 'apex', 'exit'])]

    early_braker = _trait_from_rows(
        braking_rows[braking_rows['braking_point_delta_m'] < -brake_threshold],
        'early_braker',
        'Early braker',
        'Brakes consistently earlier than the faster reference.',
        total_lap_loss,
        'braking_point_delta_m',
        brake_threshold,
        'Braking starts too early in {segment_count} segments, by about {metric_value} m on average.',
        int(max(1, braking_rows['segment_name'].nunique())),
    )
    over_slow_apex = _trait_from_rows(
        apex_rows[apex_rows['min_speed_delta_mps'] < -apex_threshold],
        'over_slow_apex',
        'Over-slow apex',
        'Drops too much speed near the middle of the corner.',
        total_lap_loss,
        'min_speed_delta_mps',
        apex_threshold,
        'Apex speed is consistently down in {segment_count} segments by about {metric_value} m/s.',
        int(max(1, apex_rows['segment_name'].nunique())),
    )
    weak_exit_throttle = _trait_from_rows(
        exit_rows[(exit_rows['throttle_pickup_delta_m'] > throttle_threshold) | (exit_rows['exit_speed_delta_mps'] < -exit_threshold)],
        'weak_exit_throttle',
        'Weak exit throttle',
        'Gets back to throttle later and gives away exit speed.',
        total_lap_loss,
        'throttle_pickup_delta_m',
        throttle_threshold,
        'Throttle pickup is delayed in {segment_count} segments by about {metric_value} m.',
        int(max(1, exit_rows['segment_name'].nunique())),
    )
    abrupt_steering = _trait_from_rows(
        control_rows[control_rows['steering_smoothness_ratio'] > steering_threshold],
        'abrupt_steering',
        'Abrupt steering',
        'Steering inputs are rougher than the reference lap.',
        total_lap_loss,
        'steering_smoothness_ratio',
        steering_threshold,
        'Steering roughness is elevated in {segment_count} segments, by about {metric_value}x.',
        int(max(1, control_rows['segment_name'].nunique())),
    )
    conservative_entry_rows = braking_rows.merge(
        apex_rows[['segment_id', 'segment_name', 'min_speed_delta_mps']],
        on=['segment_id', 'segment_name'],
        suffixes=('', '_apex'),
    )
    conservative_entry_rows = conservative_entry_rows[
        (conservative_entry_rows['braking_point_delta_m'] < -brake_threshold)
        & (conservative_entry_rows['brake_release_delta_m'] > brake_release_threshold)
        & (conservative_entry_rows['min_speed_delta_mps'] < -1.0)
    ]
    conservative_entry = _trait_from_rows(
        conservative_entry_rows,
        'conservative_entry_style',
        'Conservative entry style',
        'Brakes early, hangs on to brake too long, and arrives at apex too slowly.',
        total_lap_loss,
        'brake_release_delta_m',
        brake_release_threshold,
        'Entry is too conservative in {segment_count} segments, with brake release about {metric_value} m late.',
        int(max(1, braking_rows['segment_name'].nunique())),
    )
    poor_exit_rows = segment_cmp[
        (segment_cmp['phase'].isin(['exit', 'straight', 'segment_total']))
        & (segment_cmp['next_straight_length_m_reference'].fillna(0.0) >= 150.0)
        & ((segment_cmp['exit_speed_delta_mps'] < -1.5) | (segment_cmp['throttle_pickup_delta_m'] > throttle_threshold))
    ]
    poor_exit_compromise = _trait_from_rows(
        poor_exit_rows,
        'poor_exit_compromise',
        'Poor exit compromise',
        'Exit losses are carrying into the next straight.',
        total_lap_loss,
        'exit_speed_delta_mps',
        1.5,
        'Exit compromise shows up in {segment_count} segments, with exit speed down about {metric_value} m/s.',
        int(max(1, exit_rows['segment_name'].nunique())),
    )

    traits = [
        early_braker,
        over_slow_apex,
        weak_exit_throttle,
        abrupt_steering,
        conservative_entry,
        poor_exit_compromise,
    ]
    active_traits = [trait for trait in traits if trait['score'] >= 0.12]
    active_traits.sort(key=lambda item: (-float(item['score']), item['trait_id']))
    dominant_traits = active_traits[:3]
    card_titles = [card['title'] for card in cards[:5]]
    return {
        'profile_version': '1.0',
        'target_session': summary['target_session'],
        'reference_session': summary['reference_session'],
        'lap_time_delta_s': float(summary['lap_time_delta_s']),
        'dominant_traits': dominant_traits,
        'traits': active_traits,
        'trait_count': len(active_traits),
        'top_card_titles': card_titles,
        'profile_summary': [trait['label'] for trait in dominant_traits],
        'profile_mode': 'pace',
    }
