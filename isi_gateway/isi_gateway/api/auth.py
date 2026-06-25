"""Optional bearer-token auth dependency.

When ``settings.api_token`` is None, every request is allowed.
When set, requests must carry ``Authorization: Bearer <token>``
or receive HTTP 401.
"""

from __future__ import annotations

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

_bearer = HTTPBearer(auto_error=False)


async def require_token(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),  # noqa: B008
) -> None:
    """FastAPI dependency — raises 401 when a token is configured and absent/wrong."""
    expected: str | None = request.app.state.settings.api_token
    if expected is None:
        return
    if credentials is None or credentials.credentials != expected:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing bearer token",
            headers={"WWW-Authenticate": "Bearer"},
        )
