"""Core infrastructure: settings, security, logging, exceptions."""

from .client_identity import (
    get_client_ip,
    get_request_fingerprint,
    get_suggestion_identifier,
    get_vote_identifier,
)
from .client_identity import (
    get_server_fingerprint as get_browser_fingerprint,
)
from .config import settings
from .exceptions import (
    AppException,
    AuthenticationError,
    AuthorizationError,
    ExternalServiceError,
    NotFoundError,
    RateLimitError,
    ValidationError,
)
from .security import (
    TokenData,
    create_access_token,
    decode_access_token,
    hash_password,
    verify_password,
)


__all__ = [
    "AppException",
    "AuthenticationError",
    "AuthorizationError",
    "ExternalServiceError",
    "NotFoundError",
    "RateLimitError",
    "TokenData",
    "ValidationError",
    "create_access_token",
    "decode_access_token",
    "get_browser_fingerprint",
    "get_client_ip",
    "get_request_fingerprint",
    "get_suggestion_identifier",
    "get_vote_identifier",
    "hash_password",
    "settings",
    "verify_password",
]
