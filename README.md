# Stonkmarket

Track stock drawdowns, visualize dip history, and keep rankings fresh with an in-process scheduler.

## Features

- FastAPI backend + Jinja front-end for dip rankings and charts.
- Auth with admin-only cron management and symbol edits.
- In-process scheduler (cron expressions) that runs jobs and logs executions.
- Caching for ranking/chart data, yfinance-based quotes.

## Quickstart (local)

1. Create a virtualenv and install deps:

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. Copy env and adjust secrets:

   ```bash
   cp .env.example .env
   # edit AUTH_SECRET, ADMIN_USER, ADMIN_PASS, DOMAIN, HTTPS, etc.
   ```

3. Run the app:

   ```bash
   uvicorn main:app --reload
   ```

4. Visit [http://localhost:8000](http://localhost:8000) (dashboard). Auth-only pages: /settings, /symbols, /cronjobs (admin).

## Docker

Build and run with gunicorn + uvicorn workers:

```bash
docker compose up --build -d
```

Environment comes from `.env` (see `.env.example`). SQLite lives at `./data/dips.sqlite` on the host; WAL/SHM persist there.

## Scheduler

- Enabled by default (`SCHEDULER_ENABLED=true`).
- Uses cron expressions (UTC) from the `cronjobs` table.
- Logs to `cronjob_logs`; manual runs available via admin UI or `POST /api/cronjobs/{name}/run`.
- If scaling gunicorn workers, keep `GUNICORN_WORKERS=1` or disable the scheduler in extra instances to avoid duplicate runs.

## Deployment Notes

- Set strong `AUTH_SECRET`, `ADMIN_USER`, `ADMIN_PASS`, and `DOMAIN`; set `HTTPS=true` behind TLS so cookies are secure.
- Run behind a reverse proxy that forwards `X-Forwarded-*` headers.
- For higher concurrency or multi-instance, consider moving to Postgres instead of SQLite.

## Default Admin

- `admin` / `admin` (from `.env.example`) — change in `.env` before deploying.
