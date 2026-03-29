from __future__ import annotations

import argparse
import json
from pathlib import Path

from .analysis.ai_session_debrief import generate_ai_session_debrief
from .analysis.ai_selected_detail import generate_ai_selected_detail_sidecar
from .config import comparison_output_dir, load_config
from .app.job_runner import process_job
from .pipeline.pipeline_runner import run_comparison_pipeline
from .app.service import serve
from .app.session_store import create_job, create_user_session, get_job, get_session_metadata, list_user_sessions
from .analysis.track_mapping import export_track_map_sidecar
from .app.uploads import validate_upload


def cmd_ingest(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    from .pipeline.pipeline_runner import ingest_and_track
    sessions = args.sessions or list(config['sessions'].keys())
    results = {}
    for session_name in sessions:
        results[session_name] = ingest_and_track(session_name, config)
    print(json.dumps(results, indent=2))


def cmd_compare(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    result = run_comparison_pipeline(args.target, args.reference, config, analysis_mode='pace')
    print(json.dumps(result, indent=2))


def cmd_wheel_to_wheel(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    result = run_comparison_pipeline('wheel_to_wheel', args.reference, config, analysis_mode='racecraft')
    print(json.dumps(result, indent=2))


def cmd_track_map(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    result = export_track_map_sidecar(args.target, args.reference, config)
    output_dir = comparison_output_dir(config, args.target, args.reference)
    path = output_dir / 'track_map_segments.json'
    path.write_text(json.dumps(result, indent=2))
    print(json.dumps({'output_path': str(path), 'segment_count': len(result.get('segments', []))}, indent=2))


def cmd_ai_debrief(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    output_dir = comparison_output_dir(config, args.target, args.reference)
    debrief, output_path = generate_ai_session_debrief(output_dir, config)
    print(json.dumps({
        'output_path': str(output_path),
        'target_session': debrief.get('target_session'),
        'reference_session': debrief.get('reference_session'),
        'generation_mode': debrief.get('generation_mode'),
        'strength_count': len(debrief.get('top_strengths', [])),
        'weakness_count': len(debrief.get('top_weaknesses', [])),
        'focus_count': len(debrief.get('next_session_focus', [])),
    }, indent=2))


def cmd_ai_detail(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    output_dir = comparison_output_dir(config, args.target, args.reference)
    detail, output_path = generate_ai_selected_detail_sidecar(output_dir, config)
    print(json.dumps({
        'output_path': str(output_path),
        'target_session': detail.get('target_session'),
        'reference_session': detail.get('reference_session'),
        'generation_mode': detail.get('generation_mode'),
        'card_count': len(detail.get('cards', [])),
        'corner_count': len(detail.get('corners', [])),
        'replay_item_count': len(detail.get('replay_items', [])),
    }, indent=2))


def cmd_upload_session(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    analysis_mode = args.analysis_mode or config['user_flow']['default_analysis_mode']
    reference_session = args.reference or config['user_flow']['default_reference_session']
    validation = validate_upload(args.file, reference_session, analysis_mode, config)
    session = create_user_session(config, args.user_id, Path(args.file).stem, args.file, reference_session, analysis_mode)
    job = create_job(config, args.user_id, session['session_id'], reference_session, analysis_mode)
    payload = {'validation': validation, 'session': session, 'job': job}
    if args.run_now:
        payload['result'] = process_job(job['job_id'], args.config)
    print(json.dumps(payload, indent=2))


def cmd_run_job(args: argparse.Namespace) -> None:
    result = process_job(args.job_id, args.config)
    print(json.dumps(result, indent=2))


def cmd_job_status(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    print(json.dumps(get_job(config, args.job_id), indent=2))


def cmd_list_sessions(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    print(json.dumps({'user_id': args.user_id, 'sessions': list_user_sessions(config, args.user_id)}, indent=2))


def cmd_session_status(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    print(json.dumps(get_session_metadata(config, args.user_id, args.session_id), indent=2))


def cmd_serve(args: argparse.Namespace) -> None:
    serve(args.config, host=args.host, port=args.port)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='RPE driver coaching pipeline')
    parser.add_argument('--config', default=str(Path(__file__).resolve().parents[1] / 'configs' / 'default.yaml'))
    sub = parser.add_subparsers(dest='command', required=True)

    ingest = sub.add_parser('ingest', help='Ingest one or more hackathon sessions into aligned Parquet telemetry')
    ingest.add_argument('--sessions', nargs='*', choices=['good_lap', 'fast_laps', 'wheel_to_wheel'])
    ingest.set_defaults(func=cmd_ingest)

    compare = sub.add_parser('compare', help='Run end-to-end coaching comparison for a target session against a reference session')
    compare.add_argument('--target', default='good_lap', choices=['good_lap', 'fast_laps', 'wheel_to_wheel'])
    compare.add_argument('--reference', default='fast_laps', choices=['good_lap', 'fast_laps', 'wheel_to_wheel'])
    compare.set_defaults(func=cmd_compare)

    wheel = sub.add_parser('wheel-to-wheel', help='Run wheel-to-wheel racecraft analysis against a clean reference session')
    wheel.add_argument('--reference', default='fast_laps', choices=['good_lap', 'fast_laps'])
    wheel.set_defaults(func=cmd_wheel_to_wheel)

    track_map = sub.add_parser('track-map', help='Generate UI-ready track mapping sidecars for an existing comparison')
    track_map.add_argument('--target', default='good_lap', choices=['good_lap', 'fast_laps', 'wheel_to_wheel'])
    track_map.add_argument('--reference', default='fast_laps', choices=['good_lap', 'fast_laps', 'wheel_to_wheel'])
    track_map.set_defaults(func=cmd_track_map)

    ai_debrief = sub.add_parser('ai-debrief', help='Generate a bounded AI session debrief sidecar from existing comparison outputs')
    ai_debrief.add_argument('--target', default='good_lap', choices=['good_lap', 'fast_laps', 'wheel_to_wheel'])
    ai_debrief.add_argument('--reference', default='fast_laps', choices=['good_lap', 'fast_laps', 'wheel_to_wheel'])
    ai_debrief.set_defaults(func=cmd_ai_debrief)

    ai_detail = sub.add_parser('ai-detail', help='Generate bounded selected-detail AI explanations from existing comparison outputs')
    ai_detail.add_argument('--target', default='good_lap', choices=['good_lap', 'fast_laps', 'wheel_to_wheel'])
    ai_detail.add_argument('--reference', default='fast_laps', choices=['good_lap', 'fast_laps', 'wheel_to_wheel'])
    ai_detail.set_defaults(func=cmd_ai_detail)

    upload = sub.add_parser('upload-session', help='Validate and register a user upload, optionally processing it immediately')
    upload.add_argument('--user-id', required=True)
    upload.add_argument('--file', required=True)
    upload.add_argument('--reference', default=None, choices=['good_lap', 'fast_laps', 'wheel_to_wheel'])
    upload.add_argument('--analysis-mode', default=None, choices=['pace', 'racecraft'])
    upload.add_argument('--run-now', action='store_true')
    upload.set_defaults(func=cmd_upload_session)

    run_job = sub.add_parser('run-job', help='Run a queued upload processing job synchronously')
    run_job.add_argument('--job-id', required=True)
    run_job.set_defaults(func=cmd_run_job)

    job_status = sub.add_parser('job-status', help='Inspect a processing job')
    job_status.add_argument('--job-id', required=True)
    job_status.set_defaults(func=cmd_job_status)

    list_sessions = sub.add_parser('list-sessions', help='List processed and pending sessions for a user')
    list_sessions.add_argument('--user-id', required=True)
    list_sessions.set_defaults(func=cmd_list_sessions)

    session_status = sub.add_parser('session-status', help='Inspect one uploaded session record')
    session_status.add_argument('--user-id', required=True)
    session_status.add_argument('--session-id', required=True)
    session_status.set_defaults(func=cmd_session_status)

    serve_cmd = sub.add_parser('serve', help='Start the file-backed coaching service for uploads, jobs, and result retrieval')
    serve_cmd.add_argument('--host', default='127.0.0.1')
    serve_cmd.add_argument('--port', type=int, default=8080)
    serve_cmd.set_defaults(func=cmd_serve)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
