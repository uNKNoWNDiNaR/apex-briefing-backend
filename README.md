# RPE Coaching Backend

Backend service for the Yas Marina coaching app. This repo is the source of truth for:
- session upload and validation
- job creation and processing
- telemetry ingestion and lap comparison
- coaching outputs and racecraft outputs
- user/session history
- AI sidecars and result retrieval for the app

## Repo layout
- `coach/pipeline/`: ingestion, signal alignment, track progress, segmentation, pipeline runner
- `coach/analysis/`: comparison, coaching, racecraft, personalization, AI sidecars, track map sidecars
- `coach/app/`: upload handling, user/session storage, jobs, HTTP service
- `configs/`: default runtime config
- `docs/`: deployment and architecture notes
- `scripts/`: bootstrap and run helpers
- `outputs/`: deterministic comparison/demo artifacts
- `runtime/`: uploaded sessions, jobs, user state, processed results

## What does not belong here
- frontend UI code
- committed virtual environments
- generated runtime artifacts in source control
- old autonomy/racestack experiments

## Prerequisites
- Python 3.12
- local dataset present at `/Volumes/SanDisk/RPE/hackathon_data`
- optional: OpenAI API key for grounded AI outputs

## Local quick start
```bash
cd /Volumes/SanDisk/RPE/backend
./scripts/bootstrap.sh
./scripts/run_api.sh
```

That starts the backend API on `http://127.0.0.1:8091`.

`.venv/` is intentionally local-only and ignored by git.

## Run the full app locally
The app is backend-driven, but the frontend is a separate repo.

Terminal 1:
```bash
cd /Volumes/SanDisk/RPE/backend
./scripts/run_api.sh
```

Terminal 2:
```bash
cd /Volumes/SanDisk/RPE/apex-briefing
npm install
npm run dev
```

Open:
- `http://127.0.0.1:8080`

The frontend will call this backend for uploads, session history, processed results, coaching outputs, and AI sidecars.

## Main backend commands
Serve API:
```bash
.venv/bin/python -m coach.cli serve --host 127.0.0.1 --port 8091
```

Generate comparison outputs:
```bash
.venv/bin/python -m coach.cli compare --target good_lap --reference fast_laps
.venv/bin/python -m coach.cli wheel-to-wheel --reference fast_laps
```

Run upload flow from CLI:
```bash
.venv/bin/python -m coach.cli upload-session \
  --user-id demo_user \
  --file /Volumes/SanDisk/RPE/hackathon_data/hackathon_good_lap.mcap \
  --reference fast_laps \
  --analysis-mode pace \
  --run-now
```

Generate bounded AI sidecars:
```bash
.venv/bin/python -m coach.cli ai-debrief --target good_lap --reference fast_laps
.venv/bin/python -m coach.cli ai-detail --target good_lap --reference fast_laps
```

## Main HTTP endpoints
Health:
- `GET /health`

Upload and process:
- `POST /api/uploads`

User profile:
- `GET /api/users/<user_id>/profile`
- `POST /api/users/<user_id>/profile`

User sessions:
- `GET /api/users/<user_id>/sessions`
- `GET /api/users/<user_id>/sessions/<session_id>`
- `GET /api/users/<user_id>/sessions/<session_id>/result`
- `GET /api/users/<user_id>/sessions/<session_id>/files/<file_name>`

## Upload flow
1. client posts `.mcap` file to `POST /api/uploads`
2. backend validates file, user, reference session, and analysis mode
3. backend creates a processing job and a user session record
4. pipeline ingests telemetry, computes progress, segments the lap, compares to reference, and writes coaching outputs
5. app retrieves result metadata and result files from the session endpoints

## Files the app depends on
Processed session results can include:
- `run_summary.json`
- `lap_summary.json`
- `coach_cards.json`
- `replay_guidance.json`
- `coach_evidence.json`
- `corner_brief.json`
- `session_takeaways.json`
- `time_loss_map.json`
- `telemetry_overlay.json`
- `driver_profile.json`
- `next_session_plan.json`
- `ai_session_debrief.json`
- `ai_selected_detail.json`
- `track_map_segments.json`
- `racecraft_summary.json` when applicable

## Runtime storage
- `outputs/`: static/generated comparison outputs used for demo and inspection
- `runtime/jobs/`: job status records
- `runtime/users/<user_id>/user.json`: stored user profile
- `runtime/users/<user_id>/sessions/<session_id>/session.json`: uploaded session metadata
- `runtime/users/<user_id>/sessions/<session_id>/result/result_metadata.json`: app-facing result manifest

## Render deployment
Primary deployment files:
- `render.yaml`
- `.env.example`
- `docs/DEPLOYMENT_RENDER.md`

Required env vars:
- `COACH_DATASET_ROOT`
- `COACH_OUTPUT_ROOT`
- `COACH_RUNTIME_ROOT`
- `COACH_CORS_ALLOWED_ORIGINS`

Recommended CORS value if using Lovable plus local dev:
```env
COACH_CORS_ALLOWED_ORIGINS=https://race-vision.lovable.app,https://*.lovableproject.com,http://127.0.0.1:8080,http://localhost:8080
```

Optional cache tuning:
- `COACH_READ_CACHE_TTL_S`
- `COACH_READ_CACHE_MAX_FILE_BYTES`

Grounded AI env vars:
```env
OPENAI_API_KEY=your_key
COACH_AI_DEBRIEF_ENABLED=true
COACH_AI_DETAIL_ENABLED=true
COACH_AI_MODEL=gpt-5
COACH_AI_TEMPERATURE=0.1
```

## Troubleshooting
Backend starts but upload fails:
- check `COACH_DATASET_ROOT`, `COACH_OUTPUT_ROOT`, and `COACH_RUNTIME_ROOT`
- check writable disk space
- check Render logs for upload stage logging

Frontend says `Failed to fetch`:
- check `COACH_CORS_ALLOWED_ORIGINS`
- check that frontend is calling the correct backend URL
- confirm `/health` returns `200`

AI sidecars fall back to template mode:
- verify `OPENAI_API_KEY`
- verify `COACH_AI_DEBRIEF_ENABLED=true`
- verify `COACH_AI_DETAIL_ENABLED=true`
- inspect Render logs for `grounded_llm_*` messages

## Additional docs
- `docs/ARCHITECTURE.md`
- `docs/DEPLOYMENT_RENDER.md`
