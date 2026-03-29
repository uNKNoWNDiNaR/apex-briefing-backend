from __future__ import annotations

from copy import deepcopy
import os
from pathlib import Path
from typing import Any

import yaml


DEFAULTS = {
    'paths': {
        'dataset_root': '/Volumes/SanDisk/RPE/hackathon_data',
        'output_root': '/Volumes/SanDisk/RPE/backend/outputs',
        'runtime_root': '/Volumes/SanDisk/RPE/backend/runtime',
    },
    'sessions': {
        'good_lap': 'hackathon_good_lap.mcap',
        'fast_laps': 'hackathon_fast_laps.mcap',
        'wheel_to_wheel': 'hackathon_wheel_to_wheel.mcap',
    },
}


def _deep_merge(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
    return base


def _apply_env_overrides(config: dict[str, Any]) -> None:
    env_map = {
        ('paths', 'dataset_root'): os.getenv('COACH_DATASET_ROOT') or os.getenv('RPE_DATASET_ROOT'),
        ('paths', 'output_root'): os.getenv('COACH_OUTPUT_ROOT') or os.getenv('RPE_OUTPUT_ROOT'),
        ('paths', 'runtime_root'): os.getenv('COACH_RUNTIME_ROOT') or os.getenv('RPE_RUNTIME_ROOT'),
        ('mapping', 'handoff_root'): os.getenv('COACH_HANDOFF_ROOT'),
        ('mapping', 'app_public_root'): os.getenv('COACH_APP_PUBLIC_ROOT'),
        ('service', 'cors_allowed_origins'): os.getenv('COACH_CORS_ALLOWED_ORIGINS'),
        ('ai_debrief', 'allow_remote_generation'): os.getenv('COACH_AI_DEBRIEF_ENABLED'),
        ('ai_debrief', 'model'): os.getenv('COACH_AI_MODEL'),
        ('ai_debrief', 'temperature'): os.getenv('COACH_AI_TEMPERATURE'),
        ('ai_selected_detail', 'allow_remote_generation'): os.getenv('COACH_AI_DETAIL_ENABLED'),
        ('ai_selected_detail', 'model'): os.getenv('COACH_AI_DETAIL_MODEL') or os.getenv('COACH_AI_MODEL'),
        ('ai_selected_detail', 'temperature'): os.getenv('COACH_AI_DETAIL_TEMPERATURE') or os.getenv('COACH_AI_TEMPERATURE'),
    }
    for (section, key), value in env_map.items():
        if value:
            if section == 'service' and key == 'cors_allowed_origins':
                config.setdefault(section, {})[key] = [item.strip() for item in value.split(',') if item.strip()]
            elif key == 'allow_remote_generation':
                config.setdefault(section, {})[key] = str(value).strip().lower() in {'1', 'true', 'yes', 'on'}
            elif key == 'temperature':
                try:
                    config.setdefault(section, {})[key] = float(value)
                except Exception:
                    config.setdefault(section, {})[key] = value
            else:
                config.setdefault(section, {})[key] = value


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    config = deepcopy(DEFAULTS)
    config_path = Path(path) if path else Path(__file__).resolve().parents[1] / 'configs' / 'default.yaml'
    if config_path.exists():
        loaded = yaml.safe_load(config_path.read_text()) or {}
        _deep_merge(config, loaded)
    _apply_env_overrides(config)
    config['paths']['dataset_root'] = str(Path(config['paths']['dataset_root']).expanduser())
    config['paths']['output_root'] = str(Path(config['paths']['output_root']).expanduser())
    config['paths']['runtime_root'] = str(Path(config['paths'].get('runtime_root', Path(config['paths']['output_root']) / 'runtime')).expanduser())
    if 'mapping' in config:
        for key in ('handoff_root', 'app_public_root'):
            if key in config['mapping'] and config['mapping'][key]:
                config['mapping'][key] = str(Path(config['mapping'][key]).expanduser())
    return config


def get_dataset_root(config: dict[str, Any]) -> Path:
    return Path(config['paths']['dataset_root'])


def get_output_root(config: dict[str, Any]) -> Path:
    path = Path(config['paths']['output_root'])
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_session_path(config: dict[str, Any], session_name: str) -> Path:
    dataset_root = get_dataset_root(config)
    rel = Path(config['sessions'][session_name]).expanduser()
    if rel.is_absolute():
        return rel
    return dataset_root / rel


def get_boundary_path(config: dict[str, Any]) -> Path:
    return get_dataset_root(config) / config['track']['boundary_json']


def session_output_dir(config: dict[str, Any], session_name: str) -> Path:
    path = get_output_root(config) / 'sessions' / session_name
    path.mkdir(parents=True, exist_ok=True)
    return path


def comparison_output_dir(config: dict[str, Any], target_session: str, reference_session: str) -> Path:
    path = get_output_root(config) / 'comparisons' / f'{target_session}_vs_{reference_session}'
    path.mkdir(parents=True, exist_ok=True)
    return path
