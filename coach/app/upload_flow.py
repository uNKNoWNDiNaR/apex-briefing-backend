
from __future__ import annotations

from typing import Any


def build_upload_flow_notes(target_session: str, reference_session: str) -> dict[str, Any]:
    return {
        'flow_version': '1.0',
        'status': 'demo_ready',
        'story': 'A future user uploads a session, the system aligns it to the known Yas Marina track model, compares it to a chosen reference, and returns personalized coaching outputs.',
        'input_contract': {
            'accepted_session_format': ['mcap'],
            'required_track_asset': 'yas_marina_bnd.json',
            'required_signal_groups': ['state_estimation', 'can_telemetry'],
        },
        'pipeline_steps': [
            {'step_id': 'upload_session', 'title': 'Upload a driving session', 'backend_output': 'session_telemetry.parquet'},
            {'step_id': 'align_to_track', 'title': 'Compute track progress and segment membership', 'backend_output': 'track_samples.parquet'},
            {'step_id': 'compare_to_reference', 'title': 'Compare to a faster or racecraft reference', 'backend_output': 'segment_metrics.parquet'},
            {'step_id': 'generate_personalized_outputs', 'title': 'Generate coach cards, profile, and next-session plan', 'backend_output': ['coach_cards.json', 'driver_profile.json', 'next_session_plan.json']},
        ],
        'demo_reference_mode': {
            'target_session': target_session,
            'reference_session': reference_session,
        },
        'frozen_backend_rule': 'Frontend treats packaged comparison outputs as read-only source artifacts.',
    }
