from __future__ import annotations

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.api.main import app as api_app
from app.config import settings
from app.db import init_db
from app.auth import parse_session

app = FastAPI(title="Stonkmarket UI", version="0.1.0")
app.mount("/api", api_app)
app.mount("/static", StaticFiles(directory="app/static"), name="static")
app.state.templates = Jinja2Templates(directory="app/templates")


@app.on_event("startup")
def startup():
    init_db(settings.db_path)


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return app.state.templates.TemplateResponse("index.html", {"request": request})


@app.get("/symbols", response_class=HTMLResponse)
async def symbols_page(request: Request):
    session = request.cookies.get("session")
    if not parse_session(session):
        return RedirectResponse(url="/login")
    return app.state.templates.TemplateResponse("symbols.html", {"request": request})


@app.get("/cronjobs", response_class=HTMLResponse)
async def cronjobs_page(request: Request):
    session = request.cookies.get("session")
    parsed = parse_session(session)
    if not parsed:
        return RedirectResponse(url="/login")
    username, _, _ = parsed
    if username != settings.default_admin_user:
        return RedirectResponse(url="/")
    return app.state.templates.TemplateResponse("cronjobs.html", {"request": request})


@app.get("/settings", response_class=HTMLResponse)
async def settings_page(request: Request):
    session = request.cookies.get("session")
    if not parse_session(session):
        return RedirectResponse(url="/login")
    return app.state.templates.TemplateResponse("settings.html", {"request": request})


@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    return app.state.templates.TemplateResponse("login.html", {"request": request})


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
