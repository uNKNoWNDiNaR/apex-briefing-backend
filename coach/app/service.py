from __future__ import annotations

import cgi
import json
import logging
import shutil
import threading
import traceback
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from ..config import load_config
from .job_runner import process_job
from .session_store import create_job, create_user_session, get_job, get_result_metadata, get_session_metadata, list_user_sessions
from .uploads import UploadValidationError, validate_upload


LOGGER = logging.getLogger('rpe_coach.service')


class CoachingServiceHandler(BaseHTTPRequestHandler):
    server_version = 'RPECoachService/1.0'

    def _config(self) -> dict[str, Any]:
        return self.server.config  # type: ignore[attr-defined]

    def _origin_header(self) -> str | None:
        request_origin = self.headers.get('Origin')
        allowed = self._config().get('service', {}).get('cors_allowed_origins', ['*'])
        if not allowed:
            return None
        if '*' in allowed:
            return '*'
        if request_origin and request_origin in allowed:
            return request_origin
        return None

    def _write_common_headers(self, content_type: str, content_length: int) -> None:
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(content_length))
        origin = self._origin_header()
        if origin:
            self.send_header('Access-Control-Allow-Origin', origin)
            self.send_header('Vary', 'Origin')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type, Authorization')

    def _json(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, indent=2).encode('utf-8')
        self.send_response(status)
        self._write_common_headers('application/json', len(body))
        self.end_headers()
        self.wfile.write(body)

    def _error(self, status: int, message: str, **extra: Any) -> None:
        payload = {'status': 'error', 'message': message}
        payload.update(extra)
        self._json(status, payload)

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self._write_common_headers('application/json', 0)
        self.end_headers()

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path.rstrip('/')
        if path in {'', '/'}:
            self._json(200, {'service': 'rpe-coach', 'status': 'ok', 'hint': 'Use /health or /api/... endpoints'})
            return
        if path == '/health':
            self._json(200, {'status': 'ok'})
            return
        if path.startswith('/api/jobs/'):
            job_id = path.split('/')[-1]
            try:
                self._json(200, get_job(self._config(), job_id))
            except Exception as exc:
                self._error(404, str(exc))
            return
        if path.startswith('/api/users/') and path.endswith('/sessions'):
            parts = path.split('/')
            user_id = parts[3]
            self._json(200, {'user_id': user_id, 'sessions': list_user_sessions(self._config(), user_id)})
            return
        if path.startswith('/api/users/') and path.endswith('/result'):
            parts = path.split('/')
            user_id = parts[3]
            session_id = parts[5]
            try:
                self._json(200, get_result_metadata(self._config(), user_id, session_id))
            except Exception as exc:
                self._error(404, str(exc))
            return
        if path.startswith('/api/users/') and '/files/' in path:
            parts = path.split('/')
            user_id = parts[3]
            session_id = parts[5]
            file_name = unquote(parts[7])
            try:
                meta = get_result_metadata(self._config(), user_id, session_id)
                file_path = Path(meta['result_files'][file_name])
                if not file_path.exists():
                    raise FileNotFoundError(file_path)
                body = file_path.read_bytes()
                self.send_response(200)
                if file_path.suffix == '.json':
                    content_type = 'application/json'
                elif file_path.suffix == '.parquet':
                    content_type = 'application/octet-stream'
                else:
                    content_type = 'application/octet-stream'
                self._write_common_headers(content_type, len(body))
                self.end_headers()
                self.wfile.write(body)
            except Exception as exc:
                self._error(404, str(exc))
            return
        if path.startswith('/api/users/') and '/sessions/' in path:
            parts = path.split('/')
            user_id = parts[3]
            session_id = parts[5]
            try:
                session = get_session_metadata(self._config(), user_id, session_id)
                self._json(200, session)
            except Exception as exc:
                self._error(404, str(exc))
            return
        self._error(404, 'Unknown endpoint')

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path.rstrip('/')
        if path == '/api/uploads':
            self._handle_upload()
            return
        self._error(404, 'Unknown endpoint')

    def _handle_upload(self) -> None:
        content_type = self.headers.get('Content-Type', '')
        content_length = self.headers.get('Content-Length', '')
        LOGGER.info('upload_request_started content_type=%s content_length=%s remote=%s', content_type, content_length, self.client_address[0] if self.client_address else 'unknown')
        stage = 'parse_content_type'
        try:
            ctype, pdict = cgi.parse_header(content_type)
            if ctype != 'multipart/form-data':
                self._error(400, 'Expected multipart/form-data upload', stage=stage)
                return
            boundary = pdict.get('boundary')
            if not boundary:
                self._error(400, 'Missing multipart boundary', stage=stage)
                return
            stage = 'parse_form'
            form = cgi.FieldStorage(
                fp=self.rfile,
                headers=self.headers,
                environ={
                    'REQUEST_METHOD': 'POST',
                    'CONTENT_TYPE': content_type,
                    'CONTENT_LENGTH': content_length,
                },
            )
            user_id = form.getvalue('user_id') or 'demo_user'
            reference_session = form.getvalue('reference_session') or 'fast_laps'
            analysis_mode = form.getvalue('analysis_mode') or 'pace'
            LOGGER.info('upload_form_parsed user_id=%s reference_session=%s analysis_mode=%s', user_id, reference_session, analysis_mode)

            stage = 'extract_file_field'
            if 'file' not in form:
                raise UploadValidationError('Missing upload file field')
            file_item = form['file']
            if isinstance(file_item, list) or not getattr(file_item, 'filename', None):
                raise UploadValidationError('Missing upload file field')

            stage = 'save_upload'
            tmp_root = Path(self._config()['paths']['runtime_root']) / 'incoming'
            tmp_root.mkdir(parents=True, exist_ok=True)
            temp_path = tmp_root / f"{uuid.uuid4().hex[:12]}_{Path(file_item.filename).name}"
            with temp_path.open('wb') as fh:
                shutil.copyfileobj(file_item.file, fh, length=8 * 1024 * 1024)
            LOGGER.info('upload_saved temp_path=%s file_name=%s size_bytes=%s', temp_path, Path(file_item.filename).name, temp_path.stat().st_size if temp_path.exists() else 'missing')

            stage = 'validate_upload'
            validation = validate_upload(temp_path, reference_session, analysis_mode, self._config())
            LOGGER.info('upload_validated source_path=%s reference_session=%s analysis_mode=%s', validation.get('source_path'), reference_session, analysis_mode)

            stage = 'create_session'
            session = create_user_session(self._config(), user_id, Path(file_item.filename).stem, temp_path, reference_session, analysis_mode)
            LOGGER.info('upload_session_created session_id=%s uploaded_path=%s', session.get('session_id'), session.get('uploaded_path'))

            stage = 'create_job'
            job = create_job(self._config(), user_id, session['session_id'], reference_session, analysis_mode)
            LOGGER.info('upload_job_created job_id=%s session_id=%s', job.get('job_id'), session.get('session_id'))

            stage = 'start_processing'
            thread = threading.Thread(target=process_job, args=(job['job_id'], getattr(self.server, 'config_path', None)), daemon=True)
            thread.start()
            LOGGER.info('upload_processing_started job_id=%s', job.get('job_id'))
            self._json(202, {'status': 'accepted', 'validation': validation, 'session': session, 'job': job})
        except UploadValidationError as exc:
            LOGGER.warning('upload_validation_failed stage=%s error=%s', stage, exc)
            self._error(400, str(exc), stage=stage)
        except Exception as exc:
            LOGGER.exception('upload_failed stage=%s error=%s', stage, exc)
            self._error(500, str(exc), stage=stage, error_type=type(exc).__name__)


def serve(config_path: str | None = None, host: str = '127.0.0.1', port: int = 8080) -> None:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(name)s %(message)s')
    config = load_config(config_path)
    server = ThreadingHTTPServer((host, port), CoachingServiceHandler)
    server.config = config  # type: ignore[attr-defined]
    server.config_path = config_path  # type: ignore[attr-defined]
    try:
        server.serve_forever()
    finally:
        server.server_close()
