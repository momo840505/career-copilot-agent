from __future__ import annotations

import secrets

from fastapi import Depends, Header, HTTPException

from career_copilot.config import Settings, get_settings


def require_access_code(
    x_access_code: str | None = Header(default=None),
    settings: Settings = Depends(get_settings),
) -> None:
    if not settings.access_code:
        return
    if not x_access_code or not secrets.compare_digest(x_access_code, settings.access_code):
        raise HTTPException(status_code=401, detail="Missing or incorrect access code.")


def get_client_id(x_client_id: str | None = Header(default=None)) -> str:
    if not x_client_id:
        raise HTTPException(status_code=400, detail="X-Client-Id header is required.")
    if len(x_client_id) > 128:
        raise HTTPException(status_code=400, detail="X-Client-Id is too long.")
    return x_client_id
