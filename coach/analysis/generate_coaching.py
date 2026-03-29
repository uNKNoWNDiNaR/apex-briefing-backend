
from __future__ import annotations

from collections import Counter
from typing import Any

import pandas as pd


DUPLICATE_GROUPS = {
    'brake_later': 'entry_braking',
    'release_earlier': 'entry_braking',
    'apex_speed': 'midcorner_speed',
    'exit_speed': 'exit_drive',
    'throttle_pickup': 'exit_drive',
    'straight_compromise': 'exit_drive',
    'steering_smoothness': 'control_smoothness',
}


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _confidence_label(score: float, config: dict[str, object]) -> str:
    high = float(config['stats']['confidence_high_score'])
    medium = float(config['stats']['confidence_medium_score'])
    if score >= high:
        return 'high'
    if score >= medium:
        return 'medium'
    return 'low'


def _severity(expected_gain_s: float) -> str:
    if expected_gain_s >= 0.45:
        return 'high'
    if expected_gain_s >= 0.18:
        return 'medium'
    return 'low'


def _positive_phase_loss_maps(segment_cmp: pd.DataFrame) -> tuple[float, dict[tuple[int, str], float], dict[str, float], dict[int, float]]:
    rows = segment_cmp.copy()
    rows['positive_time_loss_s'] = rows['time_loss_s'].clip(lower=0.0)
    non_total = rows[rows['phase'] != 'segment_total']
    total_positive = float(non_total['positive_time_loss_s'].sum())
    by_segment_phase = {
        (int(row.segment_id), str(row.phase)): float(row.positive_time_loss_s)
        for row in non_total.itertuples(index=False)
    }
    by_phase = {
        str(phase): float(value)
        for phase, value in non_total.groupby('phase')['positive_time_loss_s'].sum().items()
    }
    segment_total = {
        int(row.segment_id): float(row.positive_time_loss_s)
        for row in rows[rows['phase'] == 'segment_total'].itertuples(index=False)
    }
    return total_positive, by_segment_phase, by_phase, segment_total


def _metric_completeness(row, fields: list[str]) -> float:
    present = 0
    for field in fields:
        value = getattr(row, field)
        if pd.notna(value):
            present += 1
    return present / float(len(fields)) if fields else 1.0


def _confidence_score(metric_margin: float, time_loss_s: float, phase_share: float, segment_share: float, completeness: float) -> float:
    margin_score = _clamp(metric_margin / 2.0)
    time_score = _clamp(time_loss_s / 0.35)
    return _clamp(0.35 * margin_score + 0.25 * time_score + 0.20 * phase_share + 0.10 * segment_share + 0.10 * completeness)


def _expected_gain(time_loss_s: float, segment_total_s: float, phase_share: float, confidence_score: float, config: dict[str, object]) -> float:
    scale = float(config['comparison']['expected_gain_scale'])
    minimum = float(config['stats']['expected_gain_min_s'])
    cap_ratio = float(config['stats']['expected_gain_cap_ratio'])
    abs_cap = float(config['stats']['expected_gain_abs_cap_s'])
    recoverability = _clamp(0.18 + 0.22 * confidence_score + 0.18 * phase_share)
    capped_loss = min(time_loss_s, max(0.0, segment_total_s) if segment_total_s > 0 else time_loss_s)
    gain = capped_loss * scale * recoverability
    capped = min(gain, capped_loss * cap_ratio if capped_loss > 0 else gain, abs_cap)
    return round(max(minimum, capped), 3)


def _build_card(
    row,
    category: str,
    title: str,
    summary: str,
    action: str,
    metric_margin: float,
    score: float,
    phase_share: float,
    segment_share: float,
    segment_total_s: float,
    config: dict[str, object],
) -> dict[str, object]:
    completeness = _metric_completeness(row, ['braking_point_delta_m', 'brake_release_delta_m', 'min_speed_delta_mps', 'exit_speed_delta_mps', 'throttle_pickup_delta_m', 'steering_smoothness_ratio'])
    confidence_score = _confidence_score(metric_margin, float(row.time_loss_s), phase_share, segment_share, completeness)
    expected_gain_s = _expected_gain(float(row.time_loss_s), segment_total_s, phase_share, confidence_score, config)
    return {
        'card_id': f'{row.segment_name.lower()}_{row.phase}_{category}',
        'segment_id': int(row.segment_id),
        'segment_name': row.segment_name,
        'segment_type': row.segment_type,
        'phase': row.phase,
        'category': category,
        'duplicate_group': DUPLICATE_GROUPS.get(category, category),
        'loss_context': 'pace',
        'title': title,
        'message': summary,
        'recommended_action': action,
        'time_loss_s': float(row.time_loss_s),
        'expected_gain_s': expected_gain_s,
        'severity': _severity(expected_gain_s),
        'confidence': _confidence_label(confidence_score, config),
        'confidence_score': round(confidence_score, 3),
        'phase_loss_share': round(phase_share, 3),
        'segment_loss_share': round(segment_share, 3),
        'score': score,
        'evidence': {
            'braking_point_delta_m': None if pd.isna(row.braking_point_delta_m) else float(row.braking_point_delta_m),
            'brake_release_delta_m': None if pd.isna(row.brake_release_delta_m) else float(row.brake_release_delta_m),
            'min_speed_delta_mps': None if pd.isna(row.min_speed_delta_mps) else float(row.min_speed_delta_mps),
            'exit_speed_delta_mps': None if pd.isna(row.exit_speed_delta_mps) else float(row.exit_speed_delta_mps),
            'throttle_pickup_delta_m': None if pd.isna(row.throttle_pickup_delta_m) else float(row.throttle_pickup_delta_m),
            'steering_smoothness_ratio': None if pd.isna(row.steering_smoothness_ratio) else float(row.steering_smoothness_ratio),
            'line_offset_delta_m': None if pd.isna(row.line_offset_delta_m) else float(row.line_offset_delta_m),
        },
    }


def _candidate_cards(segment_cmp: pd.DataFrame, config: dict[str, object]) -> list[dict[str, object]]:
    thresholds = config['comparison']
    total_positive, by_segment_phase, by_phase, segment_totals = _positive_phase_loss_maps(segment_cmp)
    out: list[dict[str, object]] = []
    ordered = segment_cmp[(segment_cmp['phase'] != 'segment_total') & (segment_cmp['time_loss_s'] > 0)].sort_values(['time_loss_s', 'segment_id'], ascending=[False, True])
    for row in ordered.itertuples(index=False):
        phase_share = by_segment_phase.get((int(row.segment_id), str(row.phase)), 0.0) / max(by_phase.get(str(row.phase), 0.0), 1e-6)
        segment_total_s = segment_totals.get(int(row.segment_id), max(float(row.time_loss_s), 0.0))
        segment_share = max(float(row.time_loss_s), 0.0) / max(segment_total_s, 1e-6)
        base_score = float(row.time_loss_s) + 0.15 * phase_share + 0.10 * segment_share
        if float(row.time_loss_s) < float(thresholds['min_time_loss_card_s']):
            continue
        if row.phase in {'braking', 'entry'} and pd.notna(row.braking_point_delta_m) and row.braking_point_delta_m < -float(thresholds['brake_point_delta_threshold_m']):
            margin = abs(float(row.braking_point_delta_m)) / float(thresholds['brake_point_delta_threshold_m'])
            out.append(_build_card(
                row, 'brake_later', 'Brake later',
                f'You are braking {abs(row.braking_point_delta_m):.1f} m earlier than the faster reference.',
                'Move the initial brake point deeper and keep the release progressive into entry.',
                margin, base_score + 0.04 * abs(float(row.braking_point_delta_m)), phase_share, segment_share, segment_total_s, config,
            ))
        if row.phase in {'braking', 'entry'} and pd.notna(row.brake_release_delta_m) and row.brake_release_delta_m > float(thresholds['brake_release_delta_threshold_m']) and pd.notna(row.min_speed_delta_mps) and row.min_speed_delta_mps < -1.0:
            margin = float(row.brake_release_delta_m) / float(thresholds['brake_release_delta_threshold_m'])
            out.append(_build_card(
                row, 'release_earlier', 'Release brake earlier',
                f'Brake release is {row.brake_release_delta_m:.1f} m later than the reference and apex speed is down {abs(row.min_speed_delta_mps):.1f} m/s.',
                'Bleed off brake earlier so the car rotates sooner before apex.',
                margin, base_score + 0.03 * float(row.brake_release_delta_m), phase_share, segment_share, segment_total_s, config,
            ))
        if row.phase in {'entry', 'apex'} and pd.notna(row.min_speed_delta_mps) and row.min_speed_delta_mps < -float(thresholds['apex_speed_delta_threshold_mps']):
            margin = abs(float(row.min_speed_delta_mps)) / float(thresholds['apex_speed_delta_threshold_mps'])
            out.append(_build_card(
                row, 'apex_speed', 'Carry more apex speed',
                f'Minimum speed is {abs(row.min_speed_delta_mps):.1f} m/s below the reference.',
                'Commit more speed through mid-corner and avoid over-slowing before the apex.',
                margin, base_score + 0.03 * abs(float(row.min_speed_delta_mps)), phase_share, segment_share, segment_total_s, config,
            ))
        if row.phase == 'exit' and pd.notna(row.exit_speed_delta_mps) and row.exit_speed_delta_mps < -float(thresholds['exit_speed_delta_threshold_mps']):
            margin = abs(float(row.exit_speed_delta_mps)) / float(thresholds['exit_speed_delta_threshold_mps'])
            out.append(_build_card(
                row, 'exit_speed', 'Protect exit speed',
                f'Exit speed is {abs(row.exit_speed_delta_mps):.1f} m/s down on the reference.',
                'Prioritize a cleaner exit line and earlier acceleration so the next straight starts stronger.',
                margin, base_score + 0.03 * abs(float(row.exit_speed_delta_mps)), phase_share, segment_share, segment_total_s, config,
            ))
        if row.phase == 'exit' and pd.notna(row.throttle_pickup_delta_m) and row.throttle_pickup_delta_m > float(thresholds['throttle_pickup_delta_threshold_m']):
            margin = float(row.throttle_pickup_delta_m) / float(thresholds['throttle_pickup_delta_threshold_m'])
            out.append(_build_card(
                row, 'throttle_pickup', 'Get on throttle earlier',
                f'Throttle comes in {row.throttle_pickup_delta_m:.1f} m later than the faster lap.',
                'Open the throttle earlier after apex once the car is pointed out of the corner.',
                margin, base_score + 0.02 * float(row.throttle_pickup_delta_m), phase_share, segment_share, segment_total_s, config,
            ))
        if row.phase == 'straight' and pd.notna(row.exit_speed_delta_mps) and row.exit_speed_delta_mps < -float(thresholds['straight_exit_delta_threshold_mps']):
            margin = abs(float(row.exit_speed_delta_mps)) / float(thresholds['straight_exit_delta_threshold_mps'])
            out.append(_build_card(
                row, 'straight_compromise', 'Poor exit compromises the straight',
                f'Straight-line loss is tied to an exit speed deficit of {abs(row.exit_speed_delta_mps):.1f} m/s.',
                'Shift focus to the corner before this straight and optimize the run onto the throttle.',
                margin, base_score + 0.025 * abs(float(row.exit_speed_delta_mps)), phase_share, segment_share, segment_total_s, config,
            ))
        if pd.notna(row.steering_smoothness_ratio) and row.steering_smoothness_ratio > float(thresholds['steering_smoothness_ratio_threshold']) and row.phase in {'entry', 'apex', 'exit'}:
            margin = float(row.steering_smoothness_ratio) / float(thresholds['steering_smoothness_ratio_threshold'])
            out.append(_build_card(
                row, 'steering_smoothness', 'Steering input is too abrupt',
                f'Steering smoothness is {row.steering_smoothness_ratio:.2f}x rougher than the reference.',
                'Trim the peak steering input and unwind the wheel more progressively.',
                margin, base_score + 0.04 * float(row.steering_smoothness_ratio), phase_share, segment_share, segment_total_s, config,
            ))
    return out


def _rank_cards(candidates: list[dict[str, object]], config: dict[str, object]) -> list[dict[str, object]]:
    max_cards = int(config['export']['top_cards'])
    max_per_segment = int(config['comparison']['max_cards_per_segment'])
    title_soft_cap = int(config['stats']['duplicate_title_soft_cap'])
    ordered = sorted(candidates, key=lambda item: (-float(item['score']), int(item['segment_id']), str(item['phase'])))
    chosen: list[dict[str, object]] = []
    seen = set()
    per_segment = Counter()
    per_title = Counter()
    for card in ordered:
        key = (card['segment_id'], card['duplicate_group'])
        if key in seen:
            continue
        if per_segment[card['segment_id']] >= max_per_segment:
            continue
        if per_title[card['title']] >= title_soft_cap and float(card['expected_gain_s']) < 0.25:
            continue
        seen.add(key)
        per_segment[card['segment_id']] += 1
        per_title[card['title']] += 1
        chosen.append(card)
        if len(chosen) >= max_cards:
            break
    for card in chosen:
        card.pop('score', None)
        card.pop('duplicate_group', None)
        card.pop('confidence_score', None)
    return chosen


def generate_coach_cards(segment_cmp: pd.DataFrame, config: dict[str, object]) -> list[dict[str, object]]:
    return _rank_cards(_candidate_cards(segment_cmp, config), config)


def build_coach_evidence(cards: list[dict[str, object]], segment_cmp: pd.DataFrame) -> list[dict[str, object]]:
    rows = []
    for card in cards:
        matches = segment_cmp[(segment_cmp['segment_id'] == card['segment_id']) & (segment_cmp['phase'] == card['phase'])]
        if matches.empty:
            continue
        row = matches.sort_values('time_loss_s', ascending=False).iloc[0]
        rows.append({
            'card_id': card['card_id'],
            'segment_name': card['segment_name'],
            'phase': card['phase'],
            'loss_context': card.get('loss_context', 'pace'),
            'time_loss_s': float(row['time_loss_s']),
            'expected_gain_s': card['expected_gain_s'],
            'confidence': card['confidence'],
            'phase_loss_share': card.get('phase_loss_share'),
            'segment_loss_share': card.get('segment_loss_share'),
            'target': {
                'brake_start_s': None if pd.isna(row['brake_start_s_target']) else float(row['brake_start_s_target']),
                'brake_release_s': None if pd.isna(row['brake_release_s_target']) else float(row['brake_release_s_target']),
                'min_speed_mps': None if pd.isna(row['min_speed_mps_target']) else float(row['min_speed_mps_target']),
                'exit_speed_mps': None if pd.isna(row['end_speed_mps_target']) else float(row['end_speed_mps_target']),
                'throttle_pickup_s': None if pd.isna(row['throttle_pickup_s_target']) else float(row['throttle_pickup_s_target']),
                'steering_smoothness': None if pd.isna(row['steering_smoothness_target']) else float(row['steering_smoothness_target']),
            },
            'reference': {
                'brake_start_s': None if pd.isna(row['brake_start_s_reference']) else float(row['brake_start_s_reference']),
                'brake_release_s': None if pd.isna(row['brake_release_s_reference']) else float(row['brake_release_s_reference']),
                'min_speed_mps': None if pd.isna(row['min_speed_mps_reference']) else float(row['min_speed_mps_reference']),
                'exit_speed_mps': None if pd.isna(row['end_speed_mps_reference']) else float(row['end_speed_mps_reference']),
                'throttle_pickup_s': None if pd.isna(row['throttle_pickup_s_reference']) else float(row['throttle_pickup_s_reference']),
                'steering_smoothness': None if pd.isna(row['steering_smoothness_reference']) else float(row['steering_smoothness_reference']),
            },
            'deltas': card['evidence'],
        })
    return rows


def build_corner_brief(segment_cmp: pd.DataFrame, cards: list[dict[str, object]]) -> list[dict[str, object]]:
    card_map = {}
    for card in cards:
        card_map.setdefault(card['segment_name'], []).append(card['title'])
    rows = []
    totals = segment_cmp[(segment_cmp['phase'] == 'segment_total') & (segment_cmp['segment_type'] == 'corner')].sort_values('segment_id')
    for row in totals.itertuples(index=False):
        rows.append({
            'segment_id': int(row.segment_id),
            'segment_name': row.segment_name,
            'segment_length_m': float(row.segment_length_m_target),
            'time_loss_s': float(row.time_loss_s),
            'entry_delta_s': float(segment_cmp[(segment_cmp['segment_id'] == row.segment_id) & (segment_cmp['phase'] == 'entry')]['time_loss_s'].sum()),
            'apex_delta_s': float(segment_cmp[(segment_cmp['segment_id'] == row.segment_id) & (segment_cmp['phase'] == 'apex')]['time_loss_s'].sum()),
            'exit_delta_s': float(segment_cmp[(segment_cmp['segment_id'] == row.segment_id) & (segment_cmp['phase'] == 'exit')]['time_loss_s'].sum()),
            'top_issues': card_map.get(row.segment_name, []),
        })
    return rows


def build_session_takeaways(summary: dict[str, object], cards: list[dict[str, object]]) -> dict[str, object]:
    top_cards = cards[:3]
    top_issue_titles = [card['title'] for card in top_cards]
    return {
        'lap_time_delta_s': summary['lap_time_delta_s'],
        'dominant_phase_losses': summary['phase_time_loss_s'],
        'dominant_phase_loss_normalized': summary.get('phase_time_loss_normalized', {}),
        'top_takeaways': [
            {
                'segment_name': card['segment_name'],
                'title': card['title'],
                'expected_gain_s': card['expected_gain_s'],
                'confidence': card['confidence'],
            }
            for card in top_cards
        ],
        'card_count': len(cards),
        'top_issue_titles': top_issue_titles,
    }
