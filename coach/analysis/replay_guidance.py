from __future__ import annotations

import numpy as np
import pandas as pd


def _time_at_s(overlay: pd.DataFrame, s_m: float, prefix: str) -> float:
    frame = overlay[['s_m', f'{prefix}_elapsed_s']].dropna().sort_values('s_m').drop_duplicates('s_m', keep='last')
    return float(np.interp([s_m], frame['s_m'].to_numpy(), frame[f'{prefix}_elapsed_s'].to_numpy())[0])


def _wrap_s(value: float, lap_length: float) -> float:
    return float(value % lap_length)


def build_replay_guidance(cards: list[dict[str, object]], segment_cmp: pd.DataFrame, overlay: pd.DataFrame, lap_length: float, config: dict[str, object]) -> list[dict[str, object]]:
    if not cards:
        return []
    replay_cfg = config['replay']
    guidance = []
    for card in cards:
        matches = segment_cmp[(segment_cmp['segment_id'] == card['segment_id']) & (segment_cmp['phase'] == card['phase'])]
        if matches.empty:
            continue
        row = matches.sort_values('time_loss_s', ascending=False).iloc[0]
        phase = str(card['phase'])
        if phase == 'braking':
            event_s = row.get('brake_start_s_reference') if pd.notna(row.get('brake_start_s_reference')) else row.get('entry_start_s_m_reference')
            trigger_distance = float(replay_cfg['braking_trigger_distance_m'])
        elif phase == 'entry':
            event_s = row.get('entry_start_s_m_reference')
            trigger_distance = float(replay_cfg['entry_trigger_distance_m'])
        elif phase == 'apex':
            event_s = row.get('apex_s_m_reference')
            trigger_distance = float(replay_cfg['apex_trigger_distance_m'])
        else:
            event_s = row.get('exit_start_s_m_reference') if pd.notna(row.get('exit_start_s_m_reference')) else row.get('throttle_pickup_s_reference')
            trigger_distance = float(replay_cfg['exit_trigger_distance_m'])
        if pd.isna(event_s):
            continue
        event_s = float(event_s)
        trigger_s = _wrap_s(event_s - trigger_distance, lap_length)
        event_time = _time_at_s(overlay, event_s, 'reference')
        trigger_time = _time_at_s(overlay, trigger_s, 'reference')
        if trigger_time > event_time:
            trigger_lead = (event_time + overlay['reference_elapsed_s'].max()) - trigger_time
        else:
            trigger_lead = event_time - trigger_time
        trigger_lead = float(np.clip(trigger_lead, float(replay_cfg['min_trigger_lead_s']), float(replay_cfg['max_trigger_lead_s'])))
        guidance.append({
            'card_id': card['card_id'],
            'segment_name': card['segment_name'],
            'phase': phase,
            'title': card['title'],
            'message': card['message'],
            'recommended_action': card.get('recommended_action'),
            'severity': card.get('severity'),
            'confidence': card.get('confidence'),
            'expected_gain_s': card.get('expected_gain_s'),
            'trigger_s_m': trigger_s,
            'event_s_m': event_s,
            'trigger_time_s_ref': float(event_time - trigger_lead),
            'event_time_s_ref': float(event_time),
            'trigger_lead_time_s': trigger_lead,
            'active_from_s_m': trigger_s,
            'active_until_s_m': event_s,
        })
    return guidance
