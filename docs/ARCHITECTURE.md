# Backend Architecture

## Package layout
- `coach/pipeline/`: MCAP ingestion, signal alignment, track progress, segmentation, and orchestration.
- `coach/analysis/`: lap comparison, coaching logic, racecraft, driver profile, AI debrief, and map sidecars.
- `coach/app/`: uploads, jobs, runtime storage, and HTTP service.
- `coach/config.py`: configuration loading and path resolution.
- `coach/cli.py`: stable command entrypoint.

## Runtime flow
1. Upload arrives through `coach/app/service.py` or `coach.cli upload-session`.
2. A session record and job record are created in `runtime/`.
3. `coach/app/job_runner.py` executes `coach/pipeline/pipeline_runner.py`.
4. The comparison package is written under `outputs/` or a user session result directory.
5. The app reads result metadata and per-file JSON/parquet through the service.
