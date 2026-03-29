# RPE Coaching Backend

Driver coaching backend for the Yas Marina hackathon dataset. This repository owns session ingestion, lap comparison, coaching generation, racecraft analysis, upload/job handling, and result retrieval for the app.

## What belongs here
- `coach/pipeline/`: telemetry ingestion and track-relative processing.
- `coach/analysis/`: coaching, racecraft, personalization, AI debrief, and map sidecars.
- `coach/app/`: uploads, jobs, result storage, and HTTP API.
- `configs/`: runtime configuration.
- `docs/`: architecture and deployment notes.
- `scripts/`: local bootstrap and run helpers.

## What does not belong here
- UI code
- old racestack/autonomy work
- committed virtual environments
- generated runtime data in source control

## Local setup
```bash
cd /Volumes/SanDisk/RPE/backend
./scripts/bootstrap.sh
./scripts/run_api.sh
```

`.venv/` is intentionally local-only and ignored by git. A senior repo does not commit virtual environments.

## Main commands
```bash
.venv/bin/python -m coach.cli serve --host 0.0.0.0 --port 8080
.venv/bin/python -m coach.cli compare --target good_lap --reference fast_laps
.venv/bin/python -m coach.cli wheel-to-wheel --reference fast_laps
.venv/bin/python -m coach.cli upload-session --user-id demo_user --file /path/to/session.mcap --reference fast_laps --analysis-mode pace --run-now
```

## Directory intent
- `outputs/`: deterministic comparison artifacts and demo outputs.
- `runtime/`: mutable uploads, jobs, user sessions, and result manifests.

## Deployment
See `docs/DEPLOYMENT_RENDER.md`.

## Architecture
See `docs/ARCHITECTURE.md`.

## Render deploy
Set these environment variables in Render:
- `COACH_DATASET_ROOT`
- `COACH_OUTPUT_ROOT`
- `COACH_RUNTIME_ROOT`
- `COACH_CORS_ALLOWED_ORIGINS`

Use `.env.example` as the source of truth for variable names.
Use `render.yaml` as the primary deployment config.
