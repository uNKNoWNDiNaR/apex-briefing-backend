from __future__ import annotations

from pathlib import Path
from typing import Any


ALLOWED_EXTENSIONS = {'.mcap'}


class UploadValidationError(ValueError):
    pass


def validate_upload(source_path: str | Path, reference_session: str, analysis_mode: str, config: dict[str, Any]) -> dict[str, Any]:
    path = Path(source_path)
    if not path.exists():
        raise UploadValidationError(f'Uploaded file not found: {path}')
    if not path.is_file():
        raise UploadValidationError(f'Upload path is not a file: {path}')
    if path.suffix.lower() not in ALLOWED_EXTENSIONS:
        raise UploadValidationError(f'Unsupported upload format: {path.suffix}. Supported formats: {sorted(ALLOWED_EXTENSIONS)}')
    if reference_session not in config.get('sessions', {}):
        raise UploadValidationError(f'Unknown reference session: {reference_session}')
    if analysis_mode not in {'pace', 'racecraft'}:
        raise UploadValidationError(f'Unsupported analysis mode: {analysis_mode}')
    if path.stat().st_size <= 0:
        raise UploadValidationError('Uploaded file is empty')
    return {
        'source_path': str(path),
        'file_name': path.name,
        'file_size_bytes': int(path.stat().st_size),
        'reference_session': reference_session,
        'analysis_mode': analysis_mode,
        'accepted_formats': sorted(ALLOWED_EXTENSIONS),
    }
