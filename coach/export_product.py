from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def _records(df: pd.DataFrame) -> list[dict[str, Any]]:
    if df.empty:
        return []
    return json.loads(df.to_json(orient='records'))


def export_comparison(
    output_dir: str | Path,
    summary: dict[str, Any],
    target_laps: pd.DataFrame,
    reference_laps: pd.DataFrame,
    segment_cmp: pd.DataFrame,
    overlay: pd.DataFrame,
    coach_cards: list[dict[str, Any]],
    replay_guidance: list[dict[str, Any]] | None = None,
    extra_payloads: dict[str, Any] | None = None,
) -> dict[str, str]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'run_summary.json').write_text(json.dumps(summary, indent=2))
    lap_summary = {
        'target_laps': _records(target_laps),
        'reference_laps': _records(reference_laps),
    }
    (output_dir / 'lap_summary.json').write_text(json.dumps(lap_summary, indent=2))
    segment_cmp.to_parquet(output_dir / 'segment_metrics.parquet', index=False)
    time_loss_map = overlay[['s_m', 'delta_time_s', 'segment_name', 'segment_type', 'target_speed', 'reference_speed']].rename(columns={'target_speed': 'target_speed_mps', 'reference_speed': 'reference_speed_mps'})
    (output_dir / 'time_loss_map.json').write_text(time_loss_map.to_json(orient='records', indent=2))
    telemetry_overlay = overlay[['s_m', 'target_speed', 'reference_speed', 'target_brake', 'reference_brake', 'target_throttle', 'reference_throttle', 'target_steering', 'reference_steering', 'delta_time_s', 'segment_name', 'segment_id', 'segment_type', 'phase', 'target_elapsed_s', 'reference_elapsed_s']]
    (output_dir / 'telemetry_overlay.json').write_text(telemetry_overlay.to_json(orient='records', indent=2))
    (output_dir / 'coach_cards.json').write_text(json.dumps(coach_cards, indent=2))
    guidance = replay_guidance if replay_guidance is not None else []
    (output_dir / 'replay_guidance.json').write_text(json.dumps(guidance, indent=2))
    if extra_payloads:
        for name, payload in extra_payloads.items():
            (output_dir / f'{name}.json').write_text(json.dumps(payload, indent=2))
    return {
        'run_summary_path': str(output_dir / 'run_summary.json'),
        'lap_summary_path': str(output_dir / 'lap_summary.json'),
        'segment_metrics_path': str(output_dir / 'segment_metrics.parquet'),
        'time_loss_map_path': str(output_dir / 'time_loss_map.json'),
        'telemetry_overlay_path': str(output_dir / 'telemetry_overlay.json'),
        'coach_cards_path': str(output_dir / 'coach_cards.json'),
        'replay_guidance_path': str(output_dir / 'replay_guidance.json'),
    }
