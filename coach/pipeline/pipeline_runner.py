from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pandas as pd

from ..analysis.ai_session_debrief import generate_ai_session_debrief
from ..analysis.compare_runs import build_comparison_summary, build_overlay, compare_segment_features
from .compute_track_progress import annotate_track_progress, load_track_geometry, save_track_products
from ..config import comparison_output_dir, get_boundary_path, get_session_path, session_output_dir
from ..analysis.driver_profile import build_driver_profile
from ..export_product import export_comparison
from ..analysis.extract_features import extract_features, get_lap_window, select_reference_lap
from ..analysis.generate_coaching import build_coach_evidence, build_corner_brief, build_session_takeaways, generate_coach_cards
from .ingest_mcap import ingest_session
from ..analysis.racecraft import blend_wheel_to_wheel_cards, generate_racecraft_cards
from ..analysis.replay_guidance import build_replay_guidance
from .segment_track import apply_segments, build_segments
from ..analysis.session_plan import build_next_session_plan
from ..analysis.track_mapping import export_track_map_sidecar
from ..app.upload_flow import build_upload_flow_notes


def clone_config(config: dict[str, Any]) -> dict[str, Any]:
    return copy.deepcopy(config)


def register_session_path(config: dict[str, Any], session_name: str, session_path: str | Path) -> dict[str, Any]:
    cfg = clone_config(config)
    cfg.setdefault('sessions', {})[session_name] = str(Path(session_path))
    return cfg


def with_output_root(config: dict[str, Any], output_root: str | Path) -> dict[str, Any]:
    cfg = clone_config(config)
    cfg.setdefault('paths', {})['output_root'] = str(Path(output_root))
    return cfg


def ingest_and_track(session_name: str, config: dict[str, Any]) -> dict[str, Any]:
    session_dir = session_output_dir(config, session_name)
    ingest_meta = ingest_session(get_session_path(config, session_name), session_dir, config)
    telemetry = pd.read_parquet(ingest_meta['telemetry_path'])
    geometry = load_track_geometry(get_boundary_path(config), int(config['track']['centerline_points']))
    track_df = annotate_track_progress(telemetry, geometry)
    track_meta = save_track_products(track_df, geometry, session_dir)
    return {
        **ingest_meta,
        **track_meta,
        'lap_length_m': geometry['lap_length_m'],
    }


def prepare_comparison(target_name: str, reference_name: str, config: dict[str, Any]) -> dict[str, Any]:
    reference_meta = ingest_and_track(reference_name, config)
    target_meta = ingest_and_track(target_name, config)
    reference_track = pd.read_parquet(reference_meta['track_samples_path'])
    target_track = pd.read_parquet(target_meta['track_samples_path'])
    centerline = pd.read_parquet(reference_meta['centerline_path'])
    lap_length = float(reference_meta['lap_length_m'])
    segments = build_segments(reference_track, centerline, config)
    reference_segmented = apply_segments(reference_track, segments, config)
    target_segmented = apply_segments(target_track, segments, config)
    segments.to_parquet(session_output_dir(config, reference_name) / 'track_segments.parquet', index=False)
    reference_segmented.to_parquet(session_output_dir(config, reference_name) / 'track_segmented.parquet', index=False)
    target_segmented.to_parquet(session_output_dir(config, target_name) / 'track_segmented.parquet', index=False)
    reference_laps, reference_features = extract_features(reference_segmented, segments, lap_length, config)
    target_laps, target_features = extract_features(target_segmented, segments, lap_length, config)
    reference_laps.to_parquet(session_output_dir(config, reference_name) / 'lap_summary.parquet', index=False)
    target_laps.to_parquet(session_output_dir(config, target_name) / 'lap_summary.parquet', index=False)
    reference_features.to_parquet(session_output_dir(config, reference_name) / 'segment_features.parquet', index=False)
    target_features.to_parquet(session_output_dir(config, target_name) / 'segment_features.parquet', index=False)
    reference_lap_id = select_reference_lap(reference_laps)
    target_complete = target_laps[target_laps['is_complete']]
    if target_complete.empty:
        raise RuntimeError('No complete target lap available for comparison')
    target_lap_id = int(target_complete.sort_values('lap_time_s').iloc[0]['lap_id'])
    reference_lap_track = get_lap_window(reference_segmented, lap_length, config, reference_lap_id)
    target_lap_track = get_lap_window(target_segmented, lap_length, config, target_lap_id)
    reference_lap_features = reference_features[reference_features['lap_id'] == reference_lap_id].copy()
    target_lap_features = target_features[target_features['lap_id'] == target_lap_id].copy()
    overlay = build_overlay(target_lap_track, reference_lap_track, segments, lap_length, float(config['comparison']['progress_step_m']))
    segment_cmp = compare_segment_features(target_lap_features, reference_lap_features)
    summary = build_comparison_summary(
        target_name,
        reference_name,
        target_lap_id,
        reference_lap_id,
        target_laps.loc[target_laps['lap_id'] == target_lap_id].iloc[0],
        reference_laps.loc[reference_laps['lap_id'] == reference_lap_id].iloc[0],
        segment_cmp,
    )
    return {
        'reference': reference_meta,
        'target': target_meta,
        'segments': segments,
        'target_laps': target_laps,
        'reference_laps': reference_laps,
        'segment_cmp': segment_cmp,
        'overlay': overlay,
        'summary': summary,
        'target_lap_id': target_lap_id,
        'reference_lap_id': reference_lap_id,
        'lap_length_m': lap_length,
    }


def coaching_payloads(prepared: dict[str, Any], config: dict[str, Any], racecraft_cards: list[dict] | None = None, racecraft_summary: dict[str, Any] | None = None) -> tuple[list[dict], list[dict], dict[str, Any]]:
    cards = generate_coach_cards(prepared['segment_cmp'], config)
    if racecraft_cards and racecraft_summary:
        cards = blend_wheel_to_wheel_cards(cards, racecraft_cards, racecraft_summary, config)
    replay_guidance = build_replay_guidance(cards, prepared['segment_cmp'], prepared['overlay'], prepared['lap_length_m'], config)
    driver_profile = build_driver_profile(prepared['segment_cmp'], prepared['summary'], cards, config, racecraft_cards=racecraft_cards, racecraft_summary=racecraft_summary)
    next_session_plan = build_next_session_plan(driver_profile, cards, prepared['summary'])
    sidecars = {
        'coach_evidence': build_coach_evidence(cards, prepared['segment_cmp']),
        'corner_brief': build_corner_brief(prepared['segment_cmp'], cards),
        'session_takeaways': build_session_takeaways(prepared['summary'], cards),
        'driver_profile': driver_profile,
        'next_session_plan': next_session_plan,
        'upload_flow_notes': build_upload_flow_notes(prepared['summary']['target_session'], prepared['summary']['reference_session']),
    }
    return cards, replay_guidance, sidecars


def run_comparison_pipeline(target_name: str, reference_name: str, config: dict[str, Any], analysis_mode: str = 'pace') -> dict[str, Any]:
    prepared = prepare_comparison(target_name, reference_name, config)
    racecraft_cards = None
    racecraft_summary = None
    if analysis_mode == 'racecraft':
        racecraft_cards, racecraft_summary = generate_racecraft_cards(prepared['segment_cmp'], config)
    coach_cards, replay_guidance, sidecars = coaching_payloads(prepared, config, racecraft_cards=racecraft_cards, racecraft_summary=racecraft_summary)
    output_dir = comparison_output_dir(config, target_name, reference_name)
    export_paths = export_comparison(
        output_dir,
        prepared['summary'],
        prepared['target_laps'],
        prepared['reference_laps'],
        prepared['segment_cmp'],
        prepared['overlay'],
        coach_cards,
        replay_guidance=replay_guidance,
        extra_payloads={
            **sidecars,
            **({'racecraft_cards': racecraft_cards, 'racecraft_summary': racecraft_summary} if racecraft_cards and racecraft_summary else {}),
        },
    )
    track_map = export_track_map_sidecar(target_name, reference_name, config)
    track_map_path = output_dir / 'track_map_segments.json'
    track_map_path.write_text(__import__('json').dumps(track_map, indent=2))
    ai_debrief, ai_debrief_path = generate_ai_session_debrief(output_dir, config)
    return {
        'reference': prepared['reference'],
        'target': prepared['target'],
        'comparison': export_paths,
        'target_lap_id': prepared['target_lap_id'],
        'reference_lap_id': prepared['reference_lap_id'],
        'segment_count': int(len(prepared['segments'])),
        'coach_cards': len(coach_cards),
        'replay_guidance_items': len(replay_guidance),
        'racecraft_cards': len(racecraft_cards) if racecraft_cards else 0,
        'track_map_path': str(track_map_path),
        'ai_debrief_path': str(ai_debrief_path),
        'output_dir': str(output_dir),
        'analysis_mode': analysis_mode,
    }
