
from __future__ import annotations

from typing import Any


def _focus_for_trait(trait_id: str) -> tuple[str, str, str]:
    mapping = {
        'early_braker': (
            'Brake later on approach',
            'Early braking gives away free track distance before turn-in.',
            'Use the reference as a brake-marker target and move the first hit deeper by one marker step.',
        ),
        'over_slow_apex': (
            'Carry more apex speed',
            'Over-slowing in the middle of the corner hurts the entire segment.',
            'Aim to hold more minimum speed through apex while keeping the car settled.',
        ),
        'weak_exit_throttle': (
            'Commit earlier to throttle on exit',
            'Late throttle pickup compromises the next straight and magnifies lap loss.',
            'Start squeezing throttle as soon as the car is pointed out, not after the car is fully straight.',
        ),
        'abrupt_steering': (
            'Smooth the steering trace',
            'Abrupt steering costs grip and makes the car harder to balance.',
            'Reduce steering peak rate and unwind more progressively through apex and exit.',
        ),
        'conservative_entry_style': (
            'Free up entry rotation',
            'Conservative entry style stacks losses before the apex even arrives.',
            'Brake slightly later, release earlier, and let the car rotate sooner before apex.',
        ),
        'poor_exit_compromise': (
            'Protect exits onto long straights',
            'Exit deficits carry down the next straight and are expensive to recover.',
            'Prioritize line and throttle timing on the corners that feed the longest straights.',
        ),
        'defend_entry': (
            'Be less defensive on entry',
            'Defensive entry timing is costing rotation and exit speed in traffic.',
            'Delay the defensive compromise until it is actually needed and preserve more corner entry speed.',
        ),
        'late_attack': (
            'Stop forcing late attacks',
            'Going too deep is costing the exit and killing the next opportunity.',
            'Only commit to the attack if you can still rotate and launch cleanly from apex.',
        ),
        'attack_exit': (
            'Maximize the attack exit',
            'Exit hesitation is stopping the run onto the next straight.',
            'Prioritize earlier throttle and better placement when following another car.',
        ),
        'line_compromise': (
            'Reduce line compromise under pressure',
            'Line offset under pressure is causing bigger speed loss than necessary.',
            'Give away less line than you can afford and keep the minimum speed loss under control.',
        ),
    }
    return mapping.get(trait_id, ('General pace gain', 'Repeated loss pattern detected.', 'Use the top coach cards as the next-session checklist.'))


def build_next_session_plan(
    driver_profile: dict[str, Any],
    cards: list[dict[str, Any]],
    summary: dict[str, Any],
) -> dict[str, Any]:
    focuses = []
    top_traits = driver_profile.get('dominant_traits', [])[:3]
    for idx, trait in enumerate(top_traits, start=1):
        title, why, action = _focus_for_trait(str(trait['trait_id']))
        related_cards = [card for card in cards if card['segment_name'] in trait.get('affected_segments', [])][:3]
        expected_gain = round(max((float(card.get('expected_gain_s', 0.0)) for card in related_cards), default=0.0), 3)
        focuses.append({
            'rank': idx,
            'focus_id': str(trait['trait_id']),
            'title': title,
            'why_it_matters': why,
            'what_to_do_next_session': action,
            'related_segments': trait.get('affected_segments', []),
            'related_card_ids': [card['card_id'] for card in related_cards],
            'expected_gain_s': expected_gain,
            'confidence': trait.get('confidence', 'low'),
        })
    return {
        'plan_version': '1.0',
        'target_session': driver_profile.get('target_session', summary['target_session']),
        'reference_session': driver_profile.get('reference_session', summary['reference_session']),
        'lap_time_delta_s': float(summary['lap_time_delta_s']),
        'top_3_focus_areas': focuses,
        'session_goal': 'Arrive at the next session with a short, segment-specific improvement plan rather than a long list of notes.',
    }
