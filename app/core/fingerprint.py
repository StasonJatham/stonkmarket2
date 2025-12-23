"""
DEPRECATED: Use app.core.client_identity instead.

This module remains for backwards compatibility only.
All functionality has been moved to client_identity.py.
"""

from app.core.client_identity import (
    get_client_ip,
    get_server_fingerprint as get_browser_fingerprint,
    get_request_fingerprint,
    get_vote_identifier,
    get_suggestion_identifier,
)

__all__ = [
    "get_client_ip",
    "get_browser_fingerprint",
    "get_request_fingerprint",
    "get_vote_identifier",
    "get_suggestion_identifier",
]
