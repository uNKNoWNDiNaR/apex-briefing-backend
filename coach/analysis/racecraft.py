
from __future__ import annotations

from collections import Counter
from typing import Any

import pandas as pd


def _severity(expected_gain_s: float) -> str:
    if expected_gain_s >= 1.0:
        return 'high'
    if expected_gain_s >= 0.35:
        return 'medium'
    return 'low'


def _confidence(trigger_count: int) -> str:
    if trigger_count >= 3:
        return 'high'
    if trigger_count >= 2:
        return 'medium'
    return 'low'


def _recommended_action(category: str) -> str:
    mapping = {
        'defend_entry': 'Avoid committing to the defensive brake too early. Keep enough entry speed to preserve rotation and exit.',
        'late_attack': 'Do not force a late attack unless the car can still rotate and launch cleanly from the apex.',
        'attack_exit': 'Prioritize throttle timing and car placement to maximize the run onto the next straight.',
        'line_compromise': 'Use less defensive line offset than you can afford so the minimum speed loss stays under control.',
    }
    return mapping.get(category, 'Use the racecraft reference to reduce compromise and recover exit speed.')


def _expected_gain(time_loss_s: float, config: dict[str, object]) -> float:
    stats = config['stats']
    minimum = float(stats['expected_gain_min_s'])
    abs_cap = float(stats['expected_gain_abs_cap_s'])
    return round(max(minimum, min(time_loss_s * 0.18, abs_cap)), 3)


def racecraft_dominates(racecraft_summary: dict[str, Any]) -> bool:
    return float(racecraft_summary.get('racecraft_time_loss_s', 0.0)) > max(1.0, 2.0 * float(racecraft_summary.get('pace_like_time_loss_s', 0.0)))


def blend_wheel_to_wheel_cards(
    pace_cards: list[dict[str, object]],
    racecraft_cards: list[dict[str, object]],
    racecraft_summary: dict[str, object],
    config: dict[str, object],
) -> list[dict[str, object]]:
    max_cards = int(config['export']['top_cards'])
    if not racecraft_dominates(racecraft_summary):
        return pace_cards[:max_cards]
    combined: list[dict[str, object]] = []
    seen: set[tuple[int, str]] = set()
    primary = list(racecraft_cards) + list(pace_cards)
    for card in primary:
        key = (int(card['segment_id']), str(card['title']))
        if key in seen:
            continue
        seen.add(key)
        combined.append(card)
        if len(combined) >= max_cards:
            break
    return combined


def generate_racecraft_cards(segment_cmp: pd.DataFrame, config: dict[str, object]) -> tuple[list[dict[str, object]], dict[str, object]]:
    if segment_cmp.empty:
        return [], {'card_count': 0, 'categories': {}, 'racecraft_time_loss_s': 0.0, 'pace_like_time_loss_s': 0.0}
    thresholds = config['racecraft']
    rows = segment_cmp[segment_cmp['phase'] == 'segment_total'].sort_values('time_loss_s', ascending=False)
    cards: list[dict[str, object]] = []
    category_counts: dict[str, int] = {}
    racecraft_segments: set[int] = set()
    for row in rows.itertuples(index=False):
        if row.segment_type != 'corner':
            continue
        title = None
        message = None
        category = None
        trigger_count = 0
        if pd.notna(row.braking_point_delta_m) and row.braking_point_delta_m < float(thresholds['defensive_brake_delta_m']) and pd.notna(row.exit_speed_delta_mps) and row.exit_speed_delta_mps < float(thresholds['compromised_exit_delta_mps']):
            title = 'Too defensive on entry'
            message = f'You brake {abs(row.braking_point_delta_m):.1f} m earlier and give away {abs(row.exit_speed_delta_mps):.1f} m/s on exit.'
            category = 'defend_entry'
            trigger_count = 3
        elif pd.notna(row.braking_point_delta_m) and row.braking_point_delta_m > float(thresholds['late_attack_brake_delta_m']) and pd.notna(row.exit_speed_delta_mps) and row.exit_speed_delta_mps < float(thresholds['compromised_exit_delta_mps']):
            title = 'Late attack compromised exit'
            message = f'You stay in too deep by {row.braking_point_delta_m:.1f} m and hurt exit speed by {abs(row.exit_speed_delta_mps):.1f} m/s.'
            category = 'late_attack'
            trigger_count = 3
        elif pd.notna(row.throttle_pickup_delta_m) and row.throttle_pickup_delta_m > float(thresholds['follow_hesitation_delta_m']) and row.next_straight_length_m_reference >= float(thresholds['long_straight_threshold_m']):
            title = 'Missed attack onto the straight'
            message = f'Throttle comes in {row.throttle_pickup_delta_m:.1f} m late before a {row.next_straight_length_m_reference:.0f} m straight.'
            category = 'attack_exit'
            trigger_count = 2
        elif pd.notna(row.line_offset_delta_m) and row.line_offset_delta_m > float(thresholds['line_offset_delta_m']) and pd.notna(row.min_speed_delta_mps) and row.min_speed_delta_mps < -1.5:
            title = 'Line compromise under pressure'
            message = f'Line offset grows by {row.line_offset_delta_m:.2f} m and minimum speed drops by {abs(row.min_speed_delta_mps):.1f} m/s.'
            category = 'line_compromise'
            trigger_count = 2
        if not title:
            continue
        category_counts[category] = category_counts.get(category, 0) + 1
        racecraft_segments.add(int(row.segment_id))
        time_loss_s = float(max(row.time_loss_s, 0.0))
        expected_gain_s = _expected_gain(time_loss_s, config)
        cards.append({
            'card_id': f'racecraft_{row.segment_name.lower()}_{len(cards)+1}',
            'segment_id': int(row.segment_id),
            'segment_name': row.segment_name,
            'segment_type': row.segment_type,
            'phase': 'segment_total',
            'title': title,
            'message': message,
            'category': category,
            'loss_context': 'racecraft',
            'recommended_action': _recommended_action(category),
            'time_loss_s': time_loss_s,
            'expected_gain_s': expected_gain_s,
            'severity': _severity(expected_gain_s),
            'confidence': _confidence(trigger_count),
            'phase_loss_share': None,
            'segment_loss_share': 1.0,
            'evidence': {
                'braking_point_delta_m': None if pd.isna(row.braking_point_delta_m) else float(row.braking_point_delta_m),
                'exit_speed_delta_mps': None if pd.isna(row.exit_speed_delta_mps) else float(row.exit_speed_delta_mps),
                'throttle_pickup_delta_m': None if pd.isna(row.throttle_pickup_delta_m) else float(row.throttle_pickup_delta_m),
                'line_offset_delta_m': None if pd.isna(row.line_offset_delta_m) else float(row.line_offset_delta_m),
                'next_straight_length_m': float(row.next_straight_length_m_reference),
            },
        })
        if len(cards) >= int(thresholds['max_cards']):
            break
    total_positive = float(rows['time_loss_s'].clip(lower=0.0).sum())
    racecraft_time_loss_s = float(rows[rows['segment_id'].isin(racecraft_segments)]['time_loss_s'].clip(lower=0.0).sum()) if racecraft_segments else 0.0
    summary = {
        'card_count': len(cards),
        'categories': category_counts,
        'racecraft_time_loss_s': round(racecraft_time_loss_s, 3),
        'pace_like_time_loss_s': round(max(total_positive - racecraft_time_loss_s, 0.0), 3),
    }
    return cards, summary
