from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from .routes import auth as auth_routes
from .routes import cronjobs, dips, symbols
from ..config import settings
from ..scheduler import start_scheduler


def create_app() -> FastAPI:
    app = FastAPI(title="Stonkmarket", version="0.1.0", root_path=settings.root_path)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(symbols.router)
    app.include_router(dips.router)
    app.include_router(cronjobs.router)
    app.include_router(auth_routes.router)

    app.mount("/static", StaticFiles(directory="app/static"), name="static")
    templates = Jinja2Templates(directory="app/templates")
    app.state.templates = templates
    if settings.scheduler_enabled:
        start_scheduler(app)

    return app


app = create_app()
