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
    'corner_brief.json',
    'replay_guidance.json',
]

ALLOWED_REF_PREFIXES = ('card:', 'plan:', 'trait:', 'summary:phase:', 'corner:', 'replay:')


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def load_selected_detail_inputs(comparison_dir: str | Path) -> dict[str, Any]:
    comparison_dir = Path(comparison_dir)
    payload = {'comparison_dir': str(comparison_dir), 'inputs': {}}
    for name in REQUIRED_INPUTS:
        path = comparison_dir / name
        if not path.exists():
            raise FileNotFoundError(f'Missing required selected-detail input: {path}')
        payload['inputs'][name] = _read_json(path)
    return payload


def _cards(payload: dict[str, Any]) -> list[dict[str, Any]]:
    value = payload['inputs']['coach_cards.json']
    return value if isinstance(value, list) else value.get('cards', [])


def _card_evidence(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    value = payload['inputs']['coach_evidence.json']
    entries = value if isinstance(value, list) else value.get('evidence', [])
    return {entry['card_id']: entry for entry in entries if 'card_id' in entry}


def _corners(payload: dict[str, Any]) -> list[dict[str, Any]]:
    value = payload['inputs']['corner_brief.json']
    return value if isinstance(value, list) else value.get('corners', [])


def _replay(payload: dict[str, Any]) -> list[dict[str, Any]]:
    value = payload['inputs']['replay_guidance.json']
    return value if isinstance(value, list) else value.get('replay_items', [])


def _summary(payload: dict[str, Any]) -> dict[str, Any]:
    return payload['inputs']['run_summary.json']


def _round(value: Any, digits: int = 2) -> float | None:
    if value is None:
        return None
    try:
        return round(float(value), digits)
    except Exception:
        return None


def _clean_refs(refs: Any) -> list[str]:
    cleaned: list[str] = []
    for ref in refs if isinstance(refs, list) else []:
        ref = str(ref).strip()
        if ref.startswith(ALLOWED_REF_PREFIXES):
            cleaned.append(ref)
    return list(dict.fromkeys(cleaned))


def _fallback_cards(payload: dict[str, Any]) -> list[dict[str, Any]]:
    evidence = _card_evidence(payload)
    rows = []
    for card in _cards(payload):
        card_id = str(card.get('card_id'))
        ev = evidence.get(card_id, {})
        delta = ev.get('deltas', {})
        rows.append({
            'card_id': card_id,
            'segment_name': card.get('segment_name'),
            'phase': card.get('phase'),
            'title': card.get('title'),
            'explanation': str(card.get('message', '')).strip(),
            'why_it_matters': f"This card carries about {_round(card.get('time_loss_s'), 3)} s of loss with an expected recoverable gain of {_round(card.get('expected_gain_s'), 3)} s.",
            'recommended_action': card.get('recommended_action'),
            'evidence_refs': [f'card:{card_id}'],
            'metrics': {
                'time_loss_s': _round(card.get('time_loss_s'), 3),
                'expected_gain_s': _round(card.get('expected_gain_s'), 3),
                'confidence': card.get('confidence'),
                'braking_point_delta_m': _round(delta.get('braking_point_delta_m'), 3),
                'min_speed_delta_mps': _round(delta.get('min_speed_delta_mps'), 3),
                'exit_speed_delta_mps': _round(delta.get('exit_speed_delta_mps'), 3),
            },
        })
    return rows


def _fallback_corners(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for corner in _corners(payload):
        rows.append({
            'corner_id': f"corner:{corner.get('segment_id')}",
            'segment_id': corner.get('segment_id'),
            'segment_name': corner.get('segment_name'),
            'explanation': f"{corner.get('segment_name')} is losing {_round(corner.get('time_loss_s'), 3)} s overall, with entry {_round(corner.get('entry_delta_s'), 3)} s, apex {_round(corner.get('apex_delta_s'), 3)} s, and exit {_round(corner.get('exit_delta_s'), 3)} s.",
            'why_it_matters': 'This corner-level split shows where the time is actually going inside the corner, not just that the corner is slow overall.',
            'focus_phase': max(
                [('entry', abs(float(corner.get('entry_delta_s', 0.0)))), ('apex', abs(float(corner.get('apex_delta_s', 0.0)))), ('exit', abs(float(corner.get('exit_delta_s', 0.0))))],
                key=lambda item: item[1],
            )[0],
            'top_issues': list(corner.get('top_issues', [])),
            'evidence_refs': [f"corner:{corner.get('segment_id')}", f"summary:phase:{max([('entry', abs(float(corner.get('entry_delta_s', 0.0)))), ('apex', abs(float(corner.get('apex_delta_s', 0.0)))), ('exit', abs(float(corner.get('exit_delta_s', 0.0))))], key=lambda item: item[1])[0]}"]
        })
    return rows


def _fallback_replay(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for item in _replay(payload):
        card_id = str(item.get('card_id'))
        rows.append({
            'replay_item_id': f'replay:{card_id}',
            'card_id': card_id,
            'segment_name': item.get('segment_name'),
            'phase': item.get('phase'),
            'trigger_s_m': _round(item.get('trigger_s_m'), 3),
            'event_s_m': _round(item.get('event_s_m'), 3),
            'approach_brief': f"Trigger this guidance about {_round(item.get('trigger_lead_time_s'), 2)} s before the event so the driver can act before {_round(item.get('event_s_m'), 1)} m on track progress.",
            'recommended_action': item.get('recommended_action'),
            'evidence_refs': [f'replay:{card_id}', f'card:{card_id}'],
        })
    return rows


def _fallback_detail(payload: dict[str, Any]) -> dict[str, Any]:
    summary = _summary(payload)
    return {
        'detail_version': '1.0',
        'target_session': summary.get('target_session'),
        'reference_session': summary.get('reference_session'),
        'generation_mode': 'template_fallback',
        'cards': _fallback_cards(payload),
        'corners': _fallback_corners(payload),
        'replay_items': _fallback_replay(payload),
    }


def build_bounded_prompt(payload: dict[str, Any], config: dict[str, Any]) -> tuple[str, str]:
    settings = config.get('ai_selected_detail', {})
    card_limit = int(settings.get('max_cards_in_prompt', 8))
    corner_limit = int(settings.get('max_corners_in_prompt', 8))
    replay_limit = int(settings.get('max_replay_items_in_prompt', 8))
    compact = {
        'run_summary': payload['inputs']['run_summary.json'],
        'driver_profile': payload['inputs']['driver_profile.json'],
        'next_session_plan': payload['inputs']['next_session_plan.json'],
        'session_takeaways': payload['inputs']['session_takeaways.json'],
        'coach_cards': _cards(payload)[:card_limit],
        'coach_evidence': list(_card_evidence(payload).values())[:card_limit],
        'corner_brief': _corners(payload)[:corner_limit],
        'replay_guidance': _replay(payload)[:replay_limit],
    }
    system = (
        'You are generating grounded selected-detail explanations for a race-coaching app. '
        'Use only the supplied JSON facts. Do not invent numbers, telemetry, or coaching claims. '
        'Return JSON only with the exact requested keys.'
    )
    user = json.dumps({
        'task': 'Generate a sidecar of selected-detail explanations for cards, corners, and replay items.',
        'required_schema': {
            'cards': [{'card_id': 'string', 'segment_name': 'string', 'phase': 'string', 'title': 'string', 'explanation': 'string', 'why_it_matters': 'string', 'recommended_action': 'string', 'evidence_refs': ['string']}],
            'corners': [{'corner_id': 'string', 'segment_id': 'number', 'segment_name': 'string', 'explanation': 'string', 'why_it_matters': 'string', 'focus_phase': 'string', 'top_issues': ['string'], 'evidence_refs': ['string']}],
            'replay_items': [{'replay_item_id': 'string', 'card_id': 'string', 'segment_name': 'string', 'phase': 'string', 'trigger_s_m': 'number', 'event_s_m': 'number', 'approach_brief': 'string', 'recommended_action': 'string', 'evidence_refs': ['string']}],
        },
        'allowed_evidence_refs': ['card:<card_id>', 'plan:<focus_id>', 'trait:<trait_id>', 'summary:phase:<phase>', 'corner:<segment_id>', 'replay:<card_id>'],
        'facts': compact,
    }, indent=2)
    return system, user


def _normalize_detail(parsed: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any] | None:
    fallback = _fallback_detail(payload)
    def clean_card(item: Any) -> dict[str, Any] | None:
        if not isinstance(item, dict) or not str(item.get('card_id', '')).strip():
            return None
        return {
            'card_id': str(item.get('card_id')).strip(),
            'segment_name': str(item.get('segment_name', '') or '').strip(),
            'phase': str(item.get('phase', '') or '').strip(),
            'title': str(item.get('title', '') or '').strip(),
            'explanation': str(item.get('explanation', '') or '').strip(),
            'why_it_matters': str(item.get('why_it_matters', '') or '').strip(),
            'recommended_action': str(item.get('recommended_action', '') or '').strip(),
            'evidence_refs': _clean_refs(item.get('evidence_refs', [])),
        }
    def clean_corner(item: Any) -> dict[str, Any] | None:
        if not isinstance(item, dict) or not str(item.get('segment_name', '')).strip():
            return None
        try:
            segment_id = int(item.get('segment_id'))
        except Exception:
            return None
        return {
            'corner_id': str(item.get('corner_id', f'corner:{segment_id}')).strip(),
            'segment_id': segment_id,
            'segment_name': str(item.get('segment_name', '') or '').strip(),
            'explanation': str(item.get('explanation', '') or '').strip(),
            'why_it_matters': str(item.get('why_it_matters', '') or '').strip(),
            'focus_phase': str(item.get('focus_phase', '') or '').strip(),
            'top_issues': [str(v).strip() for v in item.get('top_issues', []) if str(v).strip()],
            'evidence_refs': _clean_refs(item.get('evidence_refs', [])),
        }
    def clean_replay(item: Any) -> dict[str, Any] | None:
        if not isinstance(item, dict) or not str(item.get('card_id', '')).strip():
            return None
        return {
            'replay_item_id': str(item.get('replay_item_id', f"replay:{item.get('card_id')}")).strip(),
            'card_id': str(item.get('card_id')).strip(),
            'segment_name': str(item.get('segment_name', '') or '').strip(),
            'phase': str(item.get('phase', '') or '').strip(),
            'trigger_s_m': _round(item.get('trigger_s_m'), 3),
            'event_s_m': _round(item.get('event_s_m'), 3),
            'approach_brief': str(item.get('approach_brief', '') or '').strip(),
            'recommended_action': str(item.get('recommended_action', '') or '').strip(),
            'evidence_refs': _clean_refs(item.get('evidence_refs', [])),
        }
    cards = [clean_card(x) for x in parsed.get('cards', []) if clean_card(x)]
    corners = [clean_corner(x) for x in parsed.get('corners', []) if clean_corner(x)]
    replay_items = [clean_replay(x) for x in parsed.get('replay_items', []) if clean_replay(x)]
    if not cards or not corners or not replay_items:
        return None
    fallback['generation_mode'] = 'llm_grounded'
    fallback['cards'] = cards
    fallback['corners'] = corners
    fallback['replay_items'] = replay_items
    return fallback


def _try_remote_generation(payload: dict[str, Any], config: dict[str, Any]) -> dict[str, Any] | None:
    settings = config.get('ai_selected_detail', {})
    if not generation_enabled(settings, 'COACH_AI_DETAIL_ENABLED'):
        return None
    system, user = build_bounded_prompt(payload, config)
    parsed = call_json_completion(
        system=system,
        user=user,
        settings=settings,
        default_model='gpt-4.1-mini',
        response_schema_hint={'cards': 'array', 'corners': 'array', 'replay_items': 'array'},
    )
    if not parsed:
        return None
    return _normalize_detail(parsed, payload)


def generate_ai_selected_detail_sidecar(comparison_dir: str | Path, config: dict[str, Any]) -> tuple[dict[str, Any], Path]:
    payload = load_selected_detail_inputs(comparison_dir)
    detail = _try_remote_generation(payload, config)
    if detail is None:
        detail = _fallback_detail(payload)
    output_path = Path(comparison_dir) / 'ai_selected_detail.json'
    output_path.write_text(json.dumps(detail, indent=2))
    return detail, output_path
