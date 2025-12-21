# Stonkmarket API

Production-hardened stock dip analyzer API with Valkey caching, distributed job scheduling, and JWT authentication.

## Features

- **Pure REST API** - Strict CRUD endpoints, OpenAPI documentation
- **JWT Authentication** - Secure token-based auth with bcrypt password hashing
- **Valkey/Redis Cache** - High-performance caching with stampede protection
- **Distributed Scheduling** - APScheduler with distributed locks for safe multi-instance deployment
- **Rate Limiting** - Token bucket algorithm with Valkey backend
- **Security Hardened** - Security headers, strict CORS, input validation
- **Docker Ready** - Multi-stage build, non-root user, healthchecks, resource limits

## Architecture

```
app/
├── api/           # FastAPI routes and dependencies
│   ├── routes/    # REST endpoints (auth, symbols, dips, cronjobs, health)
│   └── app.py     # API factory with middleware
├── cache/         # Valkey integration
│   ├── client.py  # Connection management
│   ├── cache.py   # Cache-aside patterns
│   ├── distributed_lock.py
│   └── rate_limit.py
├── core/          # Configuration and security
│   ├── config.py  # Pydantic settings
│   ├── security.py # JWT + bcrypt
│   ├── logging.py
│   └── exceptions.py
├── database/      # SQLite with connection pooling
├── jobs/          # Background job scheduler
├── repositories/  # Data access layer
├── schemas/       # Pydantic request/response models
├── services/      # Business logic
└── main.py        # Application entry point
```

## Quick Start

### Local Development

1. Create virtualenv and install dependencies:

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. Start Valkey (optional for local dev):

   ```bash
   docker run -d -p 6379:6379 valkey/valkey:8-alpine
   ```

3. Configure environment:

   ```bash
   cp .env.example .env
   # Edit JWT_SECRET, ADMIN_USER, ADMIN_PASS (REQUIRED!)
   ```

4. Run the API:

   ```bash
   uvicorn app.main:app --reload
   ```

5. API available at:
   - Root: http://localhost:8000/
   - API Docs: http://localhost:8000/api/docs (debug mode only)
   - Health: http://localhost:8000/api/health

### Docker Compose (Production)

```bash
# Build and start services
docker compose up --build -d

# View logs
docker compose logs -f stonkmarket

# Stop services
docker compose down
```

## API Endpoints

### Authentication
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/auth/login` | Login with credentials |
| POST | `/api/auth/logout` | Logout (invalidate token) |
| GET | `/api/auth/me` | Get current user info |
| PUT | `/api/auth/credentials` | Update password |

### Symbols
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/symbols` | List all symbols |
| GET | `/api/symbols/{symbol}` | Get symbol config |
| POST | `/api/symbols` | Add new symbol |
| PUT | `/api/symbols/{symbol}` | Update symbol |
| DELETE | `/api/symbols/{symbol}` | Remove symbol |

### Dips
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/dips/ranking` | Get dip ranking |
| GET | `/api/dips/states` | Get all dip states |
| GET | `/api/dips/{symbol}/chart` | Get chart data |
| GET | `/api/dips/{symbol}/info` | Get stock info |

### Cron Jobs
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/cronjobs` | List all jobs |
| GET | `/api/cronjobs/{name}` | Get job details |
| PUT | `/api/cronjobs/{name}` | Update job config |
| POST | `/api/cronjobs/{name}/run` | Run job manually |
| GET | `/api/cronjobs/{name}/logs` | Get job logs |

### Health
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/health` | Full health check |
| GET | `/api/health/ready` | Readiness probe |
| GET | `/api/health/live` | Liveness probe |

## Configuration

All configuration via environment variables (see `.env.example`):

| Variable | Description | Default |
|----------|-------------|---------|
| `JWT_SECRET` | Secret for JWT signing | **Required** |
| `ADMIN_USER` | Admin username | admin |
| `ADMIN_PASS` | Admin password | **Required** |
| `ENVIRONMENT` | development/staging/production | development |
| `DEBUG` | Enable debug mode | false |
| `VALKEY_URL` | Valkey connection URL | redis://localhost:6379/0 |
| `DB_PATH` | SQLite database path | /data/dips.sqlite |
| `CORS_ALLOWED_ORIGINS` | Allowed origins (comma-separated) | |
| `RATE_LIMIT_REQUESTS` | Requests per window | 100 |
| `RATE_LIMIT_WINDOW` | Window in seconds | 60 |
| `SCHEDULER_ENABLED` | Enable job scheduler | true |
| `LOG_LEVEL` | Logging level | INFO |

## Security Notes

- **Always set strong `JWT_SECRET` and `ADMIN_PASS`** - use `python -c "import secrets; print(secrets.token_urlsafe(32))"`
- Run behind a reverse proxy (nginx, Traefik) with TLS termination
- Configure `CORS_ALLOWED_ORIGINS` for production
- The API enforces rate limiting, security headers, and input validation
- Distributed locks prevent duplicate cron job runs across instances

## Frontend

The React frontend is in the `new/` directory. See [new/README.md](new/README.md) for frontend setup.

## License

MIT


- `admin` / `admin` (from `.env.example`) — change in `.env` before deploying.
