from __future__ import annotations

from fastapi import HTTPException, status

from ..services import dip_service


def run_job(conn, name: str) -> str:
    if name == "data_grab":
        dip_service.refresh_states(conn)
        return "Fetched latest quotes and refreshed dip states"
    if name == "analysis":
        dip_service.compute_ranking_details(conn, force_refresh=True)
        return "Recomputed dip ranking"
    raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Unsupported cron job")
