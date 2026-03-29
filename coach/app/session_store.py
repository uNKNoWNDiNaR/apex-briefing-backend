from __future__ import annotations

import json
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROFILE_FIELDS = ('driver_name', 'driver_type', 'experience_level', 'primary_goal')


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _normalize_profile(profile: dict[str, Any] | None) -> dict[str, Any] | None:
    if not profile:
        return None
    normalized: dict[str, Any] = {}
    for key in PROFILE_FIELDS:
        value = profile.get(key)
        if value is None:
            continue
        value = str(value).strip()
        if value:
            normalized[key] = value
    return normalized or None


def get_user_profile(config: dict[str, Any], user_id: str) -> dict[str, Any]:
    user_dir = ensure_user(config, user_id)
    meta = _read_json(user_dir / 'user.json')
    return {
        'user_id': user_id,
        'profile': meta.get('profile'),
        'created_at': meta.get('created_at'),
        'updated_at': meta.get('updated_at'),
    }


def update_user_profile(config: dict[str, Any], user_id: str, profile: dict[str, Any] | None) -> dict[str, Any]:
    user_dir = ensure_user(config, user_id)
    path = user_dir / 'user.json'
    meta = _read_json(path)
    meta['profile'] = _normalize_profile(profile)
    meta['updated_at'] = utc_now_iso()
    _write_json(path, meta)
    return {
        'user_id': user_id,
        'profile': meta.get('profile'),
        'created_at': meta.get('created_at'),
        'updated_at': meta.get('updated_at'),
    }


def resolve_user_profile(config: dict[str, Any], user_id: str, profile_override: dict[str, Any] | None = None) -> dict[str, Any] | None:
    normalized_override = _normalize_profile(profile_override)
    if normalized_override is not None:
        update_user_profile(config, user_id, normalized_override)
        return normalized_override
    return get_user_profile(config, user_id).get('profile')


def get_runtime_root(config: dict[str, Any]) -> Path:
    root = Path(config['paths'].get('runtime_root', Path(config['paths']['output_root']) / 'runtime'))
    root.mkdir(parents=True, exist_ok=True)
    return root


def get_users_root(config: dict[str, Any]) -> Path:
    root = get_runtime_root(config) / 'users'
    root.mkdir(parents=True, exist_ok=True)
    return root


def get_jobs_root(config: dict[str, Any]) -> Path:
    root = get_runtime_root(config) / 'jobs'
    root.mkdir(parents=True, exist_ok=True)
    return root


def ensure_user(config: dict[str, Any], user_id: str) -> Path:
    user_dir = get_users_root(config) / user_id
    (user_dir / 'sessions').mkdir(parents=True, exist_ok=True)
    (user_dir / 'uploads').mkdir(parents=True, exist_ok=True)
    meta = user_dir / 'user.json'
    if not meta.exists():
        _write_json(meta, {'user_id': user_id, 'created_at': utc_now_iso(), 'updated_at': utc_now_iso(), 'profile': None})
    return user_dir


def create_user_session(config: dict[str, Any], user_id: str, original_name: str, source_path: str | Path, reference_session: str, analysis_mode: str, user_profile: dict[str, Any] | None = None) -> dict[str, Any]:
    user_dir = ensure_user(config, user_id)
    session_id = uuid.uuid4().hex[:12]
    upload_dir = user_dir / 'uploads' / session_id
    upload_dir.mkdir(parents=True, exist_ok=True)
    src = Path(source_path)
    stored_path = upload_dir / src.name
    shutil.copy2(src, stored_path)
    session_dir = user_dir / 'sessions' / session_id
    session_dir.mkdir(parents=True, exist_ok=True)
    profile_snapshot = resolve_user_profile(config, user_id, user_profile)
    meta = {
        'session_id': session_id,
        'user_id': user_id,
        'original_name': original_name,
        'uploaded_name': src.name,
        'uploaded_path': str(stored_path),
        'reference_session': reference_session,
        'analysis_mode': analysis_mode,
        'created_at': utc_now_iso(),
        'status': 'uploaded',
        'result_dir': str(session_dir / 'result'),
        'user_profile': profile_snapshot,
    }
    _write_json(session_dir / 'session.json', meta)
    return meta


def get_session_metadata(config: dict[str, Any], user_id: str, session_id: str) -> dict[str, Any]:
    return _read_json(get_users_root(config) / user_id / 'sessions' / session_id / 'session.json')


def update_session_metadata(config: dict[str, Any], user_id: str, session_id: str, updates: dict[str, Any]) -> dict[str, Any]:
    path = get_users_root(config) / user_id / 'sessions' / session_id / 'session.json'
    meta = _read_json(path)
    meta.update(updates)
    _write_json(path, meta)
    return meta


def list_user_sessions(config: dict[str, Any], user_id: str) -> list[dict[str, Any]]:
    sessions_root = get_users_root(config) / user_id / 'sessions'
    if not sessions_root.exists():
        return []
    sessions = []
    for path in sorted(sessions_root.glob('*/session.json')):
        sessions.append(_read_json(path))
    return sorted(sessions, key=lambda item: item.get('created_at', ''), reverse=True)


def create_job(config: dict[str, Any], user_id: str, session_id: str, reference_session: str, analysis_mode: str) -> dict[str, Any]:
    job_id = uuid.uuid4().hex[:12]
    job = {
        'job_id': job_id,
        'user_id': user_id,
        'session_id': session_id,
        'reference_session': reference_session,
        'analysis_mode': analysis_mode,
        'status': 'queued',
        'created_at': utc_now_iso(),
        'updated_at': utc_now_iso(),
        'result_dir': str(get_users_root(config) / user_id / 'sessions' / session_id / 'result'),
        'error': None,
        'result_metadata_path': None,
    }
    _write_json(get_jobs_root(config) / f'{job_id}.json', job)
    update_session_metadata(config, user_id, session_id, {'status': 'queued', 'job_id': job_id})
    return job


def get_job(config: dict[str, Any], job_id: str) -> dict[str, Any]:
    return _read_json(get_jobs_root(config) / f'{job_id}.json')


def update_job(config: dict[str, Any], job_id: str, updates: dict[str, Any]) -> dict[str, Any]:
    path = get_jobs_root(config) / f'{job_id}.json'
    job = _read_json(path)
    job.update(updates)
    job['updated_at'] = utc_now_iso()
    _write_json(path, job)
    return job


def write_result_metadata(config: dict[str, Any], user_id: str, session_id: str, payload: dict[str, Any]) -> Path:
    result_dir = get_users_root(config) / user_id / 'sessions' / session_id / 'result'
    result_dir.mkdir(parents=True, exist_ok=True)
    path = result_dir / 'result_metadata.json'
    _write_json(path, payload)
    return path


def get_result_metadata(config: dict[str, Any], user_id: str, session_id: str) -> dict[str, Any]:
    return _read_json(get_users_root(config) / user_id / 'sessions' / session_id / 'result' / 'result_metadata.json')
