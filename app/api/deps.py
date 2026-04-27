import hmac

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.core.config import Settings, get_settings

_bearer = HTTPBearer(auto_error=True)


def require_bearer_token(
    credentials: HTTPAuthorizationCredentials = Depends(_bearer),
    settings: Settings = Depends(get_settings),
) -> None:
    expected = settings.api_token.get_secret_value()
    presented = credentials.credentials or ""
    if not hmac.compare_digest(presented, expected):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="invalid token")


def get_retriever(request: Request):
    return request.app.state.retriever


def get_request_id(request: Request) -> str:
    return getattr(request.state, "request_id", "-")
