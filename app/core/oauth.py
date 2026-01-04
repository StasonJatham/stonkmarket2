"""OAuth configuration for social authentication.

Supports Google and GitHub OAuth providers using Authlib.

Usage:
    from app.core.oauth import oauth
    
    # In routes:
    @router.get("/oauth/{provider}")
    async def oauth_login(provider: str, request: Request):
        client = oauth.create_client(provider)
        redirect_uri = request.url_for("oauth_callback", provider=provider)
        return await client.authorize_redirect(request, redirect_uri)
"""

from __future__ import annotations

from authlib.integrations.starlette_client import OAuth

from app.core.config import settings


oauth = OAuth()


def configure_oauth() -> None:
    """
    Configure OAuth providers from environment settings.
    
    Call this during app startup after settings are loaded.
    """
    # Google OAuth
    if settings.google_client_id and settings.google_client_secret:
        oauth.register(
            name="google",
            client_id=settings.google_client_id,
            client_secret=settings.google_client_secret,
            server_metadata_url="https://accounts.google.com/.well-known/openid-configuration",
            client_kwargs={"scope": "openid email profile"},
        )
    
    # GitHub OAuth
    if settings.github_client_id and settings.github_client_secret:
        oauth.register(
            name="github",
            client_id=settings.github_client_id,
            client_secret=settings.github_client_secret,
            authorize_url="https://github.com/login/oauth/authorize",
            access_token_url="https://github.com/login/oauth/access_token",
            api_base_url="https://api.github.com/",
            client_kwargs={"scope": "user:email"},
        )


def get_enabled_providers() -> list[str]:
    """Get list of enabled OAuth providers."""
    providers = []
    if settings.google_client_id and settings.google_client_secret:
        providers.append("google")
    if settings.github_client_id and settings.github_client_secret:
        providers.append("github")
    return providers
