from __future__ import annotations

import json
import os
from typing import Any

try:
    from openai import OpenAI  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    OpenAI = None


def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def generation_enabled(settings: dict[str, Any], env_flag: str) -> bool:
    if env_flag in os.environ:
        return _as_bool(os.getenv(env_flag), False)
    return _as_bool(settings.get('allow_remote_generation'), False)


def _coerce_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return default


def build_client(api_key_env: str = 'OPENAI_API_KEY', base_url_env: str = 'OPENAI_BASE_URL') -> OpenAI | None:
    if OpenAI is None:
        return None
    api_key = os.getenv(api_key_env)
    if not api_key:
        return None
    kwargs: dict[str, Any] = {'api_key': api_key}
    base_url = os.getenv(base_url_env)
    if base_url:
        kwargs['base_url'] = base_url
    return OpenAI(**kwargs)


def call_json_completion(*, system: str, user: str, settings: dict[str, Any], default_model: str, response_schema_hint: dict[str, Any]) -> dict[str, Any] | None:
    client = build_client(
        api_key_env=str(settings.get('api_key_env', 'OPENAI_API_KEY')),
        base_url_env=str(settings.get('base_url_env', 'OPENAI_BASE_URL')),
    )
    if client is None:
        return None
    model = str(settings.get('model', default_model))
    temperature = _coerce_float(settings.get('temperature'), 0.1)
    seed = settings.get('seed')
    messages = [
        {'role': 'system', 'content': system},
        {'role': 'user', 'content': user},
    ]
    request: dict[str, Any] = {
        'model': model,
        'temperature': temperature,
        'response_format': {'type': 'json_object'},
        'messages': messages,
    }
    if seed is not None:
        request['seed'] = seed
    try:
        response = client.chat.completions.create(**request)
        content = response.choices[0].message.content if response.choices else None
        if not content:
            return None
        parsed = json.loads(content)
        if not isinstance(parsed, dict):
            return None
        return parsed
    except Exception:
        return None
