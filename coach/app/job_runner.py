from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from ..config import load_config
from ..pipeline.pipeline_runner import register_session_path, run_comparison_pipeline, with_output_root
from .session_store import get_job, get_session_metadata, update_job, update_session_metadata, write_result_metadata


LOGGER = logging.getLogger('rpe_coach.job_runner')


RESULT_FILES = [
    'run_summary.json',
    'lap_summary.json',
    'coach_cards.json',
    'replay_guidance.json',
    'coach_evidence.json',
    'corner_brief.json',
    'session_takeaways.json',
    'time_loss_map.json',
    'telemetry_overlay.json',
    'driver_profile.json',
    'next_session_plan.json',
    'ai_session_debrief.json',
    'ai_selected_detail.json',
    'track_map_segments.json',
    'racecraft_summary.json',
    'racecraft_cards.json',
]


def process_job(job_id: str, config_path: str | None = None) -> dict[str, Any]:
    config = load_config(config_path)
    job = get_job(config, job_id)
    session = get_session_metadata(config, job['user_id'], job['session_id'])
    LOGGER.info('job_started job_id=%s session_id=%s analysis_mode=%s reference_session=%s', job_id, job.get('session_id'), job.get('analysis_mode'), job.get('reference_session'))
    update_job(config, job_id, {'status': 'running', 'error': None})
    update_session_metadata(config, job['user_id'], job['session_id'], {'status': 'running'})
    try:
        runtime_output_root = Path(job['result_dir']) / 'pipeline_outputs'
        target_session_name = f"user_{job['user_id']}_{job['session_id']}"
        cfg = with_output_root(config, runtime_output_root)
        cfg = register_session_path(cfg, target_session_name, session['uploaded_path'])
        LOGGER.info('job_pipeline_start job_id=%s target_session_name=%s uploaded_path=%s', job_id, target_session_name, session.get('uploaded_path'))
        result = run_comparison_pipeline(target_session_name, job['reference_session'], cfg, analysis_mode=job['analysis_mode'])
        LOGGER.info('job_pipeline_complete job_id=%s output_dir=%s coach_cards=%s replay_items=%s', job_id, result.get('output_dir'), result.get('coach_cards'), result.get('replay_guidance_items'))
        comparison_dir = Path(result['output_dir'])
        files = {name: str(comparison_dir / name) for name in RESULT_FILES if (comparison_dir / name).exists()}
        run_summary = {}
        run_summary_path = comparison_dir / 'run_summary.json'
        if run_summary_path.exists():
            run_summary = json.loads(run_summary_path.read_text())
        metadata = {
            'job_id': job_id,
            'user_id': job['user_id'],
            'session_id': job['session_id'],
            'reference_session': job['reference_session'],
            'analysis_mode': job['analysis_mode'],
            'status': 'success',
            'comparison_dir': str(comparison_dir),
            'user_profile': session.get('user_profile'),
            'result_files': files,
            'retrieval': {
                'session_path': f"/api/users/{job['user_id']}/sessions/{job['session_id']}",
                'result_metadata_path': f"/api/users/{job['user_id']}/sessions/{job['session_id']}/result",
            },
            'summary': {
                'target_session': run_summary.get('target_session'),
                'reference_session': run_summary.get('reference_session'),
                'lap_time_delta_s': run_summary.get('lap_time_delta_s'),
                'segment_time_loss_total_s': run_summary.get('segment_time_loss_total_s'),
                'segment_count': result.get('segment_count'),
                'coach_cards': result.get('coach_cards'),
                'replay_guidance_items': result.get('replay_guidance_items'),
                'racecraft_cards': result.get('racecraft_cards'),
            },
        }
        result_meta_path = write_result_metadata(config, job['user_id'], job['session_id'], metadata)
        update_job(config, job_id, {'status': 'success', 'result_metadata_path': str(result_meta_path), 'result_dir': str(comparison_dir)})
        update_session_metadata(config, job['user_id'], job['session_id'], {'status': 'success', 'result_metadata_path': str(result_meta_path), 'result_dir': str(comparison_dir)})
        LOGGER.info('job_succeeded job_id=%s result_metadata_path=%s', job_id, result_meta_path)
        return metadata
    except Exception as exc:
        LOGGER.exception('job_failed job_id=%s error=%s', job_id, exc)
        update_job(config, job_id, {'status': 'failure', 'error': str(exc)})
        update_session_metadata(config, job['user_id'], job['session_id'], {'status': 'failure', 'error': str(exc)})
        raise
