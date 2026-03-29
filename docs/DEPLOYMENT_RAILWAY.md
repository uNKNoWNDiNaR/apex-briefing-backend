# Railway Deployment

## Required environment variables
- `COACH_DATASET_ROOT`: path or mounted volume containing the hackathon MCAP files and `yas_marina_bnd.json`.
- `COACH_OUTPUT_ROOT`: writable directory for generated comparison outputs.
- `COACH_RUNTIME_ROOT`: writable directory for uploads, jobs, and user session state.
- `PORT`: provided by Railway.
- `COACH_CORS_ALLOWED_ORIGINS`: comma-separated list of allowed frontend origins.

## Entrypoint
Railway can use the included `Procfile`:

```
web: python -m coach.cli serve --host 0.0.0.0 --port ${PORT:-8080}
```

## Notes
- `.venv/` is local development state and is intentionally not committed.
- `outputs/` and `runtime/` are writable data directories, not source code.
- The service is API-only; it does not serve a frontend at `/`.

## Quick deploy checklist
1. Mount or provision storage containing the reference dataset and `yas_marina_bnd.json`.
2. Set the four environment variables from `.env.example`.
3. Deploy with `railway.toml` or `Procfile`.
4. Confirm `GET /health` returns `{"status": "ok"}`.
5. Confirm the frontend origin is included in `COACH_CORS_ALLOWED_ORIGINS`.
6. Run one upload against `/api/uploads` before connecting the app.
