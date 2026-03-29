# Render Deployment

## Required environment variables
- `COACH_DATASET_ROOT`: path to the mounted dataset directory.
- `COACH_OUTPUT_ROOT`: writable directory for generated comparison outputs.
- `COACH_RUNTIME_ROOT`: writable directory for uploads, jobs, and user session state.
- `COACH_CORS_ALLOWED_ORIGINS`: comma-separated list of allowed frontend origins.
- `PORT`: provided by Render.

## Recommended persistent disk layout
Mount a Render disk at `/data` and create:

```text
/data
  /hackathon_data
    hackathon_good_lap.mcap
    hackathon_fast_laps.mcap
    hackathon_wheel_to_wheel.mcap
    yas_marina_bnd.json
  /outputs
  /runtime
```

You can create those folders with:

```bash
./scripts/prepare_render_disk.sh /data
```

## Render blueprint
The repository includes `render.yaml` with:
- Python web service
- `/health` health check
- disk mounted at `/data`
- default path environment variables pointing to `/data/...`

## Manual setup on Render
1. Create a new Web Service from this repo.
2. Attach a persistent disk mounted at `/data`.
3. Open a shell on the running service and run:
   ```bash
   ./scripts/prepare_render_disk.sh /data
   ```
4. Copy these files into `/data/hackathon_data`:
   - `hackathon_good_lap.mcap`
   - `hackathon_fast_laps.mcap`
   - `hackathon_wheel_to_wheel.mcap`
   - `yas_marina_bnd.json`
5. Set environment variables:
   - `COACH_DATASET_ROOT=/data/hackathon_data`
   - `COACH_OUTPUT_ROOT=/data/outputs`
   - `COACH_RUNTIME_ROOT=/data/runtime`
   - `COACH_CORS_ALLOWED_ORIGINS=https://race-vision.lovable.app`
6. Deploy the service.
7. Verify `GET /health` returns `{"status": "ok"}`.
8. Run one real upload through `/api/uploads` before wiring the app.

## Notes
- `.venv/` is local development state and is intentionally not committed.
- `outputs/` and `runtime/` in the repo are local/dev directories; on Render, the real writable paths should point to the mounted disk.
- The service is API-only; it does not render a frontend page.
