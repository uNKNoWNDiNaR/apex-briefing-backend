from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .grounded_llm import call_json_completion, generation_enabled


REQUIRED_INPUTS = [
    'driver_profile.json',
    'next_session_plan.json',
    'session_takeaways.json',
    'coach_cards.json',
    'coach_evidence.json',
    'run_summary.json',
]


ALLOWED_REF_PREFIXES = ('card:', 'plan:', 'trait:', 'summary:phase:')


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def load_debrief_inputs(comparison_dir: str | Path) -> dict[str, Any]:
    comparison_dir = Path(comparison_dir)
    payload = {'comparison_dir': str(comparison_dir), 'inputs': {}}
    for name in REQUIRED_INPUTS:
        path = comparison_dir / name
        if not path.exists():
            raise FileNotFoundError(f'Missing required debrief input: {path}')
        payload['inputs'][name] = _read_json(path)
    return payload


def _cards(payload: dict[str, Any]) -> list[dict[str, Any]]:
    cards = payload['inputs']['coach_cards.json']
    return cards if isinstance(cards, list) else cards.get('cards', [])


def _evidence_map(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    entries = payload['inputs']['coach_evidence.json']
    entries = entries if isinstance(entries, list) else entries.get('evidence', [])
    return {entry['card_id']: entry for entry in entries if 'card_id' in entry}


def _dominant_traits(payload: dict[str, Any]) -> list[dict[str, Any]]:
    profile = payload['inputs']['driver_profile.json']
    return profile.get('dominant_traits') or profile.get('traits') or []


def _top_focus(payload: dict[str, Any]) -> list[dict[str, Any]]:
    plan = payload['inputs']['next_session_plan.json']
    return plan.get('top_3_focus_areas', [])


def _summary(payload: dict[str, Any]) -> dict[str, Any]:
    return payload['inputs']['run_summary.json']


def _segment_name(card: dict[str, Any]) -> str:
    return str(card.get('segment_name') or f"segment_{card.get('segment_id', 'unknown')}")


def _round(value: Any, digits: int = 2) -> float | None:
    if value is None:
        return None
    try:
        return round(float(value), digits)
    except Exception:
        return None


def _confidence_rank(label: str) -> int:
    return {'high': 2, 'medium': 1, 'low': 0}.get(str(label).lower(), -1)


def _top_cards(payload: dict[str, Any], limit: int = 3) -> list[dict[str, Any]]:
    cards = list(_cards(payload))
    cards.sort(
        key=lambda card: (
            _round(card.get('expected_gain_s'), 4) or 0.0,
            _confidence_rank(str(card.get('confidence', 'low'))),
            _round(card.get('time_loss_s'), 4) or 0.0,
        ),
        reverse=True,
    )
    return cards[:limit]


def _phase_label(phase: str) -> str:
    return {
        'entry': 'entry',
        'braking': 'braking',
        'apex': 'apex',
        'exit': 'exit',
        'straight': 'straight',
        'segment_total': 'whole segment',
    }.get(phase, phase)


def _build_strengths(payload: dict[str, Any]) -> list[dict[str, Any]]:
    strengths: list[dict[str, Any]] = []
    evidence_map = _evidence_map(payload)
    for card in _top_cards(payload, limit=4):
        evidence = evidence_map.get(card['card_id'], {})
        deltas = evidence.get('deltas', {})
        segment = _segment_name(card)
        if abs(float(deltas.get('brake_release_delta_m', 999.0))) <= 2.0:
            strengths.append({
                'title': f'Brake release is already close in {segment}',
                'detail': f"Brake release differs by only {_round(deltas.get('brake_release_delta_m'))} m, so the bigger opportunity is the initial approach rather than the release point.",
                'evidence_refs': [f"card:{card['card_id']}"]
            })
        if abs(float(deltas.get('line_offset_delta_m', 999.0))) <= 0.5:
            strengths.append({
                'title': f'Line placement is relatively close in {segment}',
                'detail': f"Line offset is only {_round(deltas.get('line_offset_delta_m'))} m away from the reference here, so the main gain is coming from timing and speed management.",
                'evidence_refs': [f"card:{card['card_id']}"]
            })
        ratio = deltas.get('steering_smoothness_ratio')
        if ratio is not None and 0.85 <= float(ratio) <= 1.15:
            strengths.append({
                'title': f'Steering smoothness is already near the reference in {segment}',
                'detail': f"Steering smoothness is at {_round(ratio, 3)}x the reference, which is close enough that bigger gains are elsewhere.",
                'evidence_refs': [f"card:{card['card_id']}"]
            })
    normalized = _summary(payload).get('phase_time_loss_normalized', {})
    for phase, share in sorted(normalized.items(), key=lambda item: item[1]):
        if share is None:
            continue
        strengths.append({
            'title': f'{phase.capitalize()} losses are relatively contained',
            'detail': f"{phase.capitalize()} accounts for only {_round(float(share) * 100.0, 1)}% of the total lap delta in this comparison.",
            'evidence_refs': [f'summary:phase:{phase}']
        })
    deduped: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in strengths:
        if item['title'] in seen:
            continue
        deduped.append(item)
        seen.add(item['title'])
        if len(deduped) == 3:
            break
    return deduped


def _build_weaknesses(payload: dict[str, Any]) -> list[dict[str, Any]]:
    weaknesses = []
    evidence_map = _evidence_map(payload)
    for card in _top_cards(payload, limit=3):
        evidence = evidence_map.get(card['card_id'], {})
        deltas = evidence.get('deltas', {})
        details = [card.get('message', '').rstrip('.')]
        if deltas.get('braking_point_delta_m') is not None:
            details.append(f"brake point delta {_round(deltas.get('braking_point_delta_m'))} m")
        if deltas.get('min_speed_delta_mps') is not None:
            details.append(f"min speed delta {_round(deltas.get('min_speed_delta_mps'))} m/s")
        if deltas.get('exit_speed_delta_mps') is not None:
            details.append(f"exit speed delta {_round(deltas.get('exit_speed_delta_mps'))} m/s")
        weaknesses.append({
            'title': f"{card.get('title')} in {_segment_name(card)}",
            'detail': '; '.join([part for part in details if part]),
            'evidence_refs': [f"card:{card['card_id']}"]
        })
    return weaknesses


def _build_next_focus(payload: dict[str, Any]) -> list[dict[str, Any]]:
    focus_items = []
    for item in _top_focus(payload):
        refs = [f"plan:{item.get('focus_id')}"]
        refs.extend([f'card:{card_id}' for card_id in item.get('related_card_ids', [])])
        focus_items.append({
            'title': item.get('title'),
            'why_it_matters': item.get('why_it_matters'),
            'what_to_do_next_session': item.get('what_to_do_next_session'),
            'evidence_refs': refs,
        })
    return focus_items


def _explain_card(card: dict[str, Any], evidence: dict[str, Any]) -> str:
    segment = _segment_name(card)
    title = str(card.get('title', 'Issue'))
    if title == 'Brake later':
        return (
            f"In {segment} {_phase_label(str(card.get('phase')))}, the car is slowing too early. "
            f"That gives away track distance before turn-in and then drags speed down through the rest of the corner."
        )
    if title == 'Carry more apex speed':
        return (
            f"In {segment}, too much speed is being scrubbed before the middle of the corner. "
            f"That smaller minimum speed then limits how quickly the car can be driven back out."
        )
    if title == 'Release brake earlier':
        return (
            f"In {segment}, the brake is being held longer than the faster reference. "
            f"That keeps extra load on the front axle deeper into the corner and delays rotation or throttle pickup."
        )
    if title == 'Poor exit compromises the straight':
        return (
            f"The exit from {segment} is costing more than just the corner itself. "
            f"A slower release back to full speed leaves time on the following straight that is hard to recover later."
        )
    if title == 'Steering input is too abrupt':
        return (
            f"In {segment}, the steering build-up is rougher than the reference. "
            f"That makes the car harder to settle and can cost confidence on the way out of the corner."
        )
    if title == 'Too defensive on entry':
        return (
            f"In {segment}, too much is being given away to protect the corner entry. "
            f"The defensive brake timing and line offset are costing rotation and then hurting the exit that follows."
        )
    if title == 'Line compromise under pressure':
        return (
            f"In {segment}, the line choice under pressure is costing more speed than necessary. "
            f"The car is carrying a larger line offset and that turns into a bigger minimum-speed or exit-speed penalty."
        )
    return (
        f"In {segment} {_phase_label(str(card.get('phase')))}, this is a repeatable loss against the reference. "
        f"The structured evidence says the issue is big enough to matter and clear enough to coach directly."
    )


def _build_plain_english(payload: dict[str, Any]) -> list[dict[str, Any]]:
    explanations = []
    evidence_map = _evidence_map(payload)
    for card in _top_cards(payload, limit=3):
        explanations.append({
            'issue': card.get('title'),
            'segment_name': _segment_name(card),
            'explanation': _explain_card(card, evidence_map.get(card['card_id'], {})),
            'evidence_refs': [f"card:{card['card_id']}"]
        })
    return explanations


def _build_short_summary(payload: dict[str, Any]) -> str:
    summary = _summary(payload)
    traits = _dominant_traits(payload)
    top_trait = traits[0] if traits else {}
    top_card = _top_cards(payload, limit=1)
    top_card = top_card[0] if top_card else {}
    phase_norm = summary.get('phase_time_loss_normalized', {})
    dominant_phase = None
    dominant_share = None
    if phase_norm:
        dominant_phase, dominant_share = max(phase_norm.items(), key=lambda item: item[1])
    text = f"You are {_round(summary.get('lap_time_delta_s'))} s off the {summary.get('reference_session')} reference. "
    if dominant_phase is not None and dominant_share is not None:
        text += f"Most of the gap sits in {dominant_phase} ({_round(float(dominant_share) * 100.0, 1)}%). "
    if top_trait:
        text += f"The profile points first to {top_trait.get('label').lower()}"
        if top_trait.get('affected_segments'):
            text += f" in {', '.join(top_trait.get('affected_segments', [])[:2])}"
        text += '. '
    if top_card:
        text += f"The clearest coaching opportunity is {_segment_name(top_card)} {_phase_label(str(top_card.get('phase')))}: {str(top_card.get('title', '')).lower()}."
    return text.strip()


def _build_motivational_close(payload: dict[str, Any]) -> str:
    focus = _top_focus(payload)
    if not focus:
        return 'The gap is clear enough to coach directly. Keep the next session focused on one repeatable change at a time.'
    titles = [item.get('title') for item in focus[:2] if item.get('title')]
    if len(titles) == 1:
        return f"The upside is that the main opportunity is concentrated. If you execute on '{titles[0]}' next session, the car should feel better immediately."
    return f"The upside is that the main opportunities are concentrated, not spread everywhere. Start with '{titles[0]}' and '{titles[1]}' and the next session should feel noticeably cleaner."


def _build_evidence_refs(payload: dict[str, Any], debrief: dict[str, Any]) -> list[dict[str, Any]]:
    refs_needed: list[str] = []
    for key in ['top_strengths', 'top_weaknesses', 'next_session_focus', 'plain_english_explanations']:
        for item in debrief.get(key, []):
            refs_needed.extend(item.get('evidence_refs', []))
    refs_needed = list(dict.fromkeys(refs_needed))
    evidence_map = _evidence_map(payload)
    cards = {card['card_id']: card for card in _cards(payload) if 'card_id' in card}
    traits = {trait.get('trait_id'): trait for trait in _dominant_traits(payload) if trait.get('trait_id')}
    focus = {item.get('focus_id'): item for item in _top_focus(payload) if item.get('focus_id')}
    summary = _summary(payload)
    refs: list[dict[str, Any]] = []
    for ref_id in refs_needed:
        kind, _, value = ref_id.partition(':')
        if kind == 'card' and value in cards:
            card = cards[value]
            evidence = evidence_map.get(value, {})
            refs.append({
                'ref_id': ref_id,
                'source_file': 'coach_cards.json',
                'card_id': value,
                'segment_name': card.get('segment_name'),
                'phase': card.get('phase'),
                'metrics': {
                    'time_loss_s': _round(card.get('time_loss_s'), 3),
                    'expected_gain_s': _round(card.get('expected_gain_s'), 3),
                    'confidence': card.get('confidence'),
                    'braking_point_delta_m': _round(evidence.get('deltas', {}).get('braking_point_delta_m'), 3),
                    'min_speed_delta_mps': _round(evidence.get('deltas', {}).get('min_speed_delta_mps'), 3),
                    'exit_speed_delta_mps': _round(evidence.get('deltas', {}).get('exit_speed_delta_mps'), 3),
                    'line_offset_delta_m': _round(evidence.get('deltas', {}).get('line_offset_delta_m'), 3),
                },
            })
        elif kind == 'plan' and value in focus:
            item = focus[value]
            refs.append({
                'ref_id': ref_id,
                'source_file': 'next_session_plan.json',
                'focus_id': value,
                'title': item.get('title'),
                'metrics': {
                    'expected_gain_s': _round(item.get('expected_gain_s'), 3),
                    'confidence': item.get('confidence'),
                    'related_segments': item.get('related_segments'),
                },
            })
        elif kind == 'summary' and value.startswith('phase:'):
            phase = value.split(':', 1)[1]
            refs.append({
                'ref_id': ref_id,
                'source_file': 'run_summary.json',
                'phase': phase,
                'metrics': {
                    'phase_time_loss_s': _round(summary.get('phase_time_loss_s', {}).get(phase), 3),
                    'phase_time_loss_normalized': _round(summary.get('phase_time_loss_normalized', {}).get(phase), 4),
                },
            })
        elif kind == 'trait' and value in traits:
            item = traits[value]
            refs.append({
                'ref_id': ref_id,
                'source_file': 'driver_profile.json',
                'trait_id': value,
                'label': item.get('label'),
                'metrics': {
                    'score': _round(item.get('score'), 3),
                    'confidence': item.get('confidence'),
                    'positive_time_loss_s': _round(item.get('evidence', {}).get('positive_time_loss_s'), 3),
                },
            })
    return refs


def _fallback_debrief(payload: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    summary = _summary(payload)
    debrief = {
        'debrief_version': '1.0',
        'target_session': summary.get('target_session'),
        'reference_session': summary.get('reference_session'),
        'generation_mode': 'template_fallback',
        'short_summary': _build_short_summary(payload),
        'top_strengths': _build_strengths(payload),
        'top_weaknesses': _build_weaknesses(payload),
        'next_session_focus': _build_next_focus(payload),
        'plain_english_explanations': _build_plain_english(payload),
        'motivational_close': _build_motivational_close(payload),
    }
    debrief['evidence_refs'] = _build_evidence_refs(payload, debrief)
    return debrief


def build_bounded_prompt(payload: dict[str, Any], config: dict[str, Any]) -> tuple[str, str]:
    settings = config.get('ai_debrief', {})
    limit = int(settings.get('max_cards_in_prompt', 6))
    compact = {
        'driver_profile': payload['inputs']['driver_profile.json'],
        'next_session_plan': payload['inputs']['next_session_plan.json'],
        'session_takeaways': payload['inputs']['session_takeaways.json'],
        'coach_cards': _top_cards(payload, limit=limit),
        'coach_evidence': list(_evidence_map(payload).values())[:limit],
        'run_summary': payload['inputs']['run_summary.json'],
    }
    system = (
        'You are writing a grounded race-engineer debrief. Use only the supplied JSON facts. '
        'Do not invent telemetry, numbers, or coaching claims. Keep the output concise and human-readable. '
        'Every evidence_refs entry must use one of the allowed reference formats exactly. Return JSON only.'
    )
    user = json.dumps({
        'task': 'Generate ai_session_debrief.json from the provided backend outputs.',
        'required_schema': {
            'short_summary': 'string',
            'top_strengths': [{'title': 'string', 'detail': 'string', 'evidence_refs': ['string']}],
            'top_weaknesses': [{'title': 'string', 'detail': 'string', 'evidence_refs': ['string']}],
            'next_session_focus': [{'title': 'string', 'why_it_matters': 'string', 'what_to_do_next_session': 'string', 'evidence_refs': ['string']}],
            'plain_english_explanations': [{'issue': 'string', 'segment_name': 'string', 'explanation': 'string', 'evidence_refs': ['string']}],
            'motivational_close': 'string'
        },
        'allowed_evidence_refs': [
            'card:<card_id>',
            'plan:<focus_id>',
            'trait:<trait_id>',
            'summary:phase:<phase>'
        ],
        'facts': compact,
    }, indent=2)
    return system, user


def _clean_refs(refs: Any) -> list[str]:
    cleaned: list[str] = []
    for ref in refs if isinstance(refs, list) else []:
        ref = str(ref).strip()
        if ref.startswith(ALLOWED_REF_PREFIXES):
            cleaned.append(ref)
    return list(dict.fromkeys(cleaned))


def _normalize_items(items: Any, allowed_fields: list[str]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for item in items if isinstance(items, list) else []:
        if not isinstance(item, dict):
            continue
        row: dict[str, Any] = {}
        for field in allowed_fields:
            if field == 'evidence_refs':
                row[field] = _clean_refs(item.get(field, []))
            else:
                row[field] = str(item.get(field, '') or '').strip()
        normalized.append(row)
    return normalized


def _normalize_debrief(parsed: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any] | None:
    summary = _summary(payload)
    debrief = {
        'debrief_version': '1.0',
        'target_session': summary.get('target_session'),
        'reference_session': summary.get('reference_session'),
        'generation_mode': 'llm_grounded',
        'short_summary': str(parsed.get('short_summary', '') or '').strip(),
        'top_strengths': _normalize_items(parsed.get('top_strengths', []), ['title', 'detail', 'evidence_refs'])[:3],
        'top_weaknesses': _normalize_items(parsed.get('top_weaknesses', []), ['title', 'detail', 'evidence_refs'])[:3],
        'next_session_focus': _normalize_items(parsed.get('next_session_focus', []), ['title', 'why_it_matters', 'what_to_do_next_session', 'evidence_refs'])[:3],
        'plain_english_explanations': _normalize_items(parsed.get('plain_english_explanations', []), ['issue', 'segment_name', 'explanation', 'evidence_refs'])[:3],
        'motivational_close': str(parsed.get('motivational_close', '') or '').strip(),
    }
    if not debrief['short_summary']:
        return None
    if not debrief['top_weaknesses']:
        return None
    debrief['evidence_refs'] = _build_evidence_refs(payload, debrief)
    return debrief


def _try_remote_generation(payload: dict[str, Any], config: dict[str, Any]) -> dict[str, Any] | None:
    settings = config.get('ai_debrief', {})
    if not generation_enabled(settings, 'COACH_AI_DEBRIEF_ENABLED'):
        return None
    system, user = build_bounded_prompt(payload, config)
    parsed = call_json_completion(
        system=system,
        user=user,
        settings=settings,
        default_model='gpt-4.1-mini',
        response_schema_hint={
            'short_summary': 'string',
            'top_strengths': 'array',
            'top_weaknesses': 'array',
            'next_session_focus': 'array',
            'plain_english_explanations': 'array',
            'motivational_close': 'string',
        },
    )
    if not parsed:
        return None
    return _normalize_debrief(parsed, payload)


def generate_ai_session_debrief(comparison_dir: str | Path, config: dict[str, Any]) -> tuple[dict[str, Any], Path]:
    payload = load_debrief_inputs(comparison_dir)
    debrief = _try_remote_generation(payload, config)
    if debrief is None:
        debrief = _fallback_debrief(payload, config)
    output_path = Path(comparison_dir) / 'ai_session_debrief.json'
    output_path.write_text(json.dumps(debrief, indent=2))
    return debrief, output_path
