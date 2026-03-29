
from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..config import comparison_output_dir, get_boundary_path, session_output_dir


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def _safe_float(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    return float(value)


def _bbox(points: list[tuple[float, float]]) -> dict[str, float] | None:
    if not points:
        return None
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return {
        'min_x': round(min(xs), 3),
        'max_x': round(max(xs), 3),
        'min_y': round(min(ys), 3),
        'max_y': round(max(ys), 3),
    }


def _segment_slice(centerline: pd.DataFrame, start_idx: int, end_idx: int) -> pd.DataFrame:
    if start_idx <= end_idx:
        return centerline.iloc[start_idx:end_idx + 1].copy()
    return pd.concat([centerline.iloc[start_idx:], centerline.iloc[:end_idx + 1]], ignore_index=True)


def _interp_centerline_point(centerline: pd.DataFrame, s_m: float | None, lap_length_m: float) -> dict[str, float] | None:
    if s_m is None:
        return None
    s_mod = float(s_m) % lap_length_m
    s_values = centerline['s_m'].to_numpy(dtype=float)
    x_values = centerline['center_x'].to_numpy(dtype=float)
    y_values = centerline['center_y'].to_numpy(dtype=float)
    s_ext = np.concatenate([s_values, [lap_length_m]])
    x_ext = np.concatenate([x_values, [x_values[0]]])
    y_ext = np.concatenate([y_values, [y_values[0]]])
    x = float(np.interp(s_mod, s_ext, x_ext))
    y = float(np.interp(s_mod, s_ext, y_ext))
    return {
        's_m': round(s_mod, 3),
        'x': round(x, 3),
        'y': round(y, 3),
    }


def _segment_record(row: pd.Series, centerline: pd.DataFrame, lap_length_m: float) -> dict[str, Any]:
    segment_points = _segment_slice(centerline, int(row['start_idx']), int(row['end_idx']))
    records = segment_points.to_dict(orient='records')
    centerline_points = [
        {
            's_m': round(float(item['s_m']), 3),
            'x': round(float(item['center_x']), 3),
            'y': round(float(item['center_y']), 3),
        }
        for item in records
    ]
    left_boundary_points = [
        [round(float(item['left_x']), 3), round(float(item['left_y']), 3)]
        for item in records
    ]
    right_boundary_points = [
        [round(float(item['right_x']), 3), round(float(item['right_y']), 3)]
        for item in records
    ]
    bbox_points = [
        (float(item['center_x']), float(item['center_y']))
        for item in records
    ] + [
        (float(item['left_x']), float(item['left_y']))
        for item in records
    ] + [
        (float(item['right_x']), float(item['right_y']))
        for item in records
    ]
    anchors = {
        'brake_zone_start': _interp_centerline_point(centerline, _safe_float(row.get('brake_zone_start_m')), lap_length_m),
        'entry_start': _interp_centerline_point(centerline, _safe_float(row.get('entry_start_s_m')), lap_length_m),
        'apex': _interp_centerline_point(centerline, _safe_float(row.get('apex_s_m')), lap_length_m),
        'apex_start': _interp_centerline_point(centerline, _safe_float(row.get('apex_start_s_m')), lap_length_m),
        'apex_end': _interp_centerline_point(centerline, _safe_float(row.get('apex_end_s_m')), lap_length_m),
        'exit_start': _interp_centerline_point(centerline, _safe_float(row.get('exit_start_s_m')), lap_length_m),
    }
    return {
        'segment_id': int(row['segment_id']),
        'segment_name': str(row['segment_name']),
        'corner_or_straight': str(row['segment_type']),
        's_start_m': round(float(row['s_start_m']), 3),
        's_end_m': round(float(row['s_end_m']), 3),
        'segment_length_m': round(float(row['segment_length_m']), 3),
        'wraps_start_finish': bool(float(row['s_end_m']) < float(row['s_start_m'])),
        'bbox': _bbox(bbox_points),
        'centerline_points': centerline_points,
        'left_boundary_points': left_boundary_points,
        'right_boundary_points': right_boundary_points,
        'phase_anchors': anchors,
        'track_width_mean_m': round(float(segment_points['track_width_m'].mean()), 3),
    }


def _build_track_outline(boundary_path: Path) -> dict[str, Any]:
    raw = _load_json(boundary_path)
    boundaries = raw.get('boundaries', {})
    left_pairs = boundaries.get('left_border', [])
    right_pairs = boundaries.get('right_border', [])
    left = [[round(float(x), 3), round(float(y), 3)] for x, y in left_pairs]
    right = [[round(float(x), 3), round(float(y), 3)] for x, y in right_pairs]
    bbox = _bbox([(float(x), float(y)) for x, y in left_pairs + right_pairs])
    return {
        'track_name': raw.get('map_name', 'unknown_track'),
        'left_boundary_points': left,
        'right_boundary_points': right,
        'bbox': bbox,
    }


def _card_segment_map(cards: list[dict[str, Any]], segment_lookup: dict[int, dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for card in cards:
        segment_id = int(card['segment_id'])
        seg = segment_lookup[segment_id]
        rows.append({
            'card_id': str(card['card_id']),
            'segment_id': segment_id,
            'segment_name': str(card['segment_name']),
            'phase': str(card.get('phase', 'unknown')),
            'title': str(card.get('title', '')),
            'segment_bbox': seg['bbox'],
        })
    return rows


def _replay_item_map(
    replay_items: list[dict[str, Any]],
    card_lookup: dict[str, dict[str, Any]],
    segment_lookup: dict[int, dict[str, Any]],
    centerline: pd.DataFrame,
    lap_length_m: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for idx, item in enumerate(replay_items, start=1):
        card_id = str(item.get('card_id', f'replay_{idx:03d}'))
        card = card_lookup.get(card_id, {})
        segment_id_raw = card.get('segment_id')
        segment_id = int(segment_id_raw) if segment_id_raw is not None else None
        trigger_s = _safe_float(item.get('trigger_s_m'))
        event_s = _safe_float(item.get('event_s_m'))
        row = {
            'replay_item_id': f'replay_{idx:03d}',
            'card_id': card_id,
            'segment_id': segment_id,
            'segment_name': card.get('segment_name') or item.get('segment_name'),
            'trigger_s_m': round(trigger_s, 3) if trigger_s is not None else None,
            'event_s_m': round(event_s, 3) if event_s is not None else None,
            'trigger_point': _interp_centerline_point(centerline, trigger_s, lap_length_m),
            'event_point': _interp_centerline_point(centerline, event_s, lap_length_m),
        }
        if segment_id is not None and segment_id in segment_lookup:
            row['segment_bbox'] = segment_lookup[segment_id]['bbox']
        rows.append(row)
    return rows


def _corner_map(corner_brief: list[dict[str, Any]], segment_lookup: dict[int, dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in corner_brief:
        segment_id = int(item['segment_id'])
        seg = segment_lookup[segment_id]
        rows.append({
            'segment_id': segment_id,
            'segment_name': str(item['segment_name']),
            'time_loss_s': round(float(item.get('time_loss_s', 0.0)), 3),
            'top_issues': item.get('top_issues', []),
            'segment_bbox': seg['bbox'],
        })
    return rows


def build_track_map_sidecar(target_session: str, reference_session: str, config: dict[str, Any]) -> dict[str, Any]:
    comparison_name = f'{target_session}_vs_{reference_session}'
    comparison_dir = comparison_output_dir(config, target_session, reference_session)
    reference_dir = session_output_dir(config, reference_session)
    segments_path = reference_dir / 'track_segments.parquet'
    centerline_path = reference_dir / 'centerline.parquet'
    track_meta_path = reference_dir / 'track_metadata.json'
    cards_path = comparison_dir / 'coach_cards.json'
    replay_path = comparison_dir / 'replay_guidance.json'
    corner_brief_path = comparison_dir / 'corner_brief.json'

    required = [segments_path, centerline_path, track_meta_path, cards_path, replay_path, corner_brief_path]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f'Missing required files for track map sidecar: {missing}')

    segments = pd.read_parquet(segments_path)
    centerline = pd.read_parquet(centerline_path)
    track_meta = _load_json(track_meta_path)
    cards = _load_json(cards_path)
    replay_items = _load_json(replay_path)
    corner_brief = _load_json(corner_brief_path)
    lap_length_m = float(track_meta['lap_length_m'])

    segment_records = [_segment_record(row, centerline, lap_length_m) for _, row in segments.iterrows()]
    segment_lookup = {item['segment_id']: item for item in segment_records}
    card_lookup = {str(card['card_id']): card for card in cards}
    track_outline = _build_track_outline(get_boundary_path(config))

    return {
        'track_map_version': '1.0',
        'comparison_name': comparison_name,
        'track_name': track_outline.get('track_name', 'unknown_track'),
        'lap_length_m': round(lap_length_m, 3),
        'track_outline': track_outline,
        'segments': segment_records,
        'card_segment_map': _card_segment_map(cards, segment_lookup),
        'replay_item_map': _replay_item_map(replay_items, card_lookup, segment_lookup, centerline, lap_length_m),
        'corner_map': _corner_map(corner_brief, segment_lookup),
    }


def _write_shared_guide(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    guide = '''# Track Map Sidecar Guide

This file documents `track_map_segments.json`, a read-only UI support sidecar.

## Purpose
Use it to:
- highlight the exact selected segment on the Yas Marina map
- zoom the camera to the selected segment bbox
- place replay trigger and event markers
- link coach cards and corner summaries back to map geometry

## Main sections
- `track_outline`: full left/right track boundaries and global bbox
- `segments`: per-segment geometry slices and phase anchors
- `card_segment_map`: card-to-segment lookup
- `replay_item_map`: replay marker lookup with trigger/event points
- `corner_map`: corner summary to segment lookup

## UI usage
- highlight a segment with `segments[].centerline_points`
- zoom with `segments[].bbox`
- place replay markers with `replay_item_map[].trigger_point` and `event_point`
- resolve a coach card to the map with `card_segment_map`

Backend status: frozen. This sidecar does not change coaching logic.
'''
    path.write_text(guide)


def export_track_map_sidecar(target_session: str, reference_session: str, config: dict[str, Any]) -> dict[str, Any]:
    comparison_name = f'{target_session}_vs_{reference_session}'
    comparison_dir = comparison_output_dir(config, target_session, reference_session)
    payload = build_track_map_sidecar(target_session, reference_session, config)
    output_path = comparison_dir / 'track_map_segments.json'
    output_path.write_text(json.dumps(payload, indent=2))

    published_paths = {'backend_output_path': str(output_path)}

    mapping_cfg = config.get('mapping', {})
    handoff_root_value = mapping_cfg.get('handoff_root')
    app_public_root_value = mapping_cfg.get('app_public_root')

    if handoff_root_value:
        try:
            handoff_root = Path(handoff_root_value)
            handoff_path = handoff_root / comparison_name / 'track_map_segments.json'
            handoff_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(output_path, handoff_path)
            _write_shared_guide(handoff_root / 'shared' / 'TRACK_MAP_UI_GUIDE.md')
            published_paths['lovable_handoff_path'] = str(handoff_path)
        except Exception:
            published_paths['lovable_handoff_path'] = None

    if app_public_root_value:
        try:
            app_public_root = Path(app_public_root_value)
            app_path = app_public_root / comparison_name / 'track_map_segments.json'
            app_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(output_path, app_path)
            _write_shared_guide(app_public_root / 'shared' / 'TRACK_MAP_UI_GUIDE.md')
            published_paths['apex_briefing_path'] = str(app_path)
        except Exception:
            published_paths['apex_briefing_path'] = None

    return {
        'comparison_name': comparison_name,
        'segment_count': len(payload['segments']),
        'card_mapping_count': len(payload['card_segment_map']),
        'replay_mapping_count': len(payload['replay_item_map']),
        'paths': published_paths,
    }
