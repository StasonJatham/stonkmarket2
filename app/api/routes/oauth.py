"""OAuth authentication routes for social login.

Supports Google and GitHub OAuth2 providers.

Usage:
    Frontend redirects to: /api/oauth/{provider}
    After OAuth: User is redirected back to frontend with session cookie set
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request, Response
from fastapi.responses import RedirectResponse

from app.core.config import settings
from app.core.logging import get_logger
from app.core.oauth import get_enabled_providers, oauth
from app.core.security import create_access_token
from app.repositories import auth_user_orm as auth_repo


logger = get_logger("api.oauth")

router = APIRouter(prefix="/oauth", tags=["OAuth"])


@router.get("/providers")
async def list_providers() -> dict:
    """List enabled OAuth providers."""
    return {"providers": get_enabled_providers()}


@router.get("/{provider}")
async def oauth_login(
    provider: str,
    request: Request,
):
    """
    Initiate OAuth login flow.
    
    Redirects to the provider's authorization page.
    """
    enabled = get_enabled_providers()
    if provider not in enabled:
        raise HTTPException(
            status_code=400,
            detail=f"OAuth provider '{provider}' is not configured. Available: {enabled}",
        )
    
    client = oauth.create_client(provider)
    redirect_uri = str(request.url_for("oauth_callback", provider=provider))
    
    logger.info(f"Starting OAuth flow for {provider}, redirect_uri={redirect_uri}")
    return await client.authorize_redirect(request, redirect_uri)


@router.get("/{provider}/callback", name="oauth_callback")
async def oauth_callback(
    provider: str,
    request: Request,
    response: Response,
):
    """
    Handle OAuth callback from provider.
    
    Creates or links user account and sets session cookie.
    """
    enabled = get_enabled_providers()
    if provider not in enabled:
        raise HTTPException(status_code=400, detail=f"Invalid provider: {provider}")
    
    try:
        client = oauth.create_client(provider)
        token = await client.authorize_access_token(request)
    except Exception as e:
        logger.error(f"OAuth callback error for {provider}: {e}")
        # Redirect to frontend with error
        return RedirectResponse(
            url=f"{settings.oauth_redirect_url}/login?error=oauth_failed",
        )
    
    # Extract user info from provider
    try:
        if provider == "google":
            user_info = token.get("userinfo")
            if not user_info:
                # Fallback: fetch from userinfo endpoint
                user_info = await client.userinfo()
            email = user_info.get("email")
            name = user_info.get("name") or (email.split("@")[0] if email else "user")
            avatar = user_info.get("picture")
            provider_id = user_info.get("sub")
            
        elif provider == "github":
            resp = await client.get("user")
            resp.raise_for_status()
            user_info = resp.json()
            
            # GitHub may have private email, need separate request
            email = user_info.get("email")
            if not email:
                # Try to get primary email from emails endpoint
                try:
                    emails_resp = await client.get("user/emails")
                    emails = emails_resp.json()
                    primary = next((e for e in emails if e.get("primary")), None)
                    email = primary.get("email") if primary else None
                except Exception:
                    pass
            
            name = user_info.get("login")
            avatar = user_info.get("avatar_url")
            provider_id = str(user_info.get("id"))
        else:
            raise HTTPException(status_code=400, detail=f"Unknown provider: {provider}")
        
        if not provider_id:
            raise HTTPException(status_code=400, detail="Could not get user ID from provider")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to extract user info from {provider}: {e}")
        return RedirectResponse(
            url=f"{settings.oauth_redirect_url}/login?error=oauth_failed",
        )
    
    # Find or create user
    user = await auth_repo.get_user_by_provider(provider, provider_id)
    
    if not user:
        # Check if email exists (link accounts)
        if email:
            existing = await auth_repo.get_user_by_email(email)
            if existing:
                # Link this provider to existing account
                await auth_repo.link_provider(existing.id, provider, provider_id)
                user = existing
                logger.info(f"Linked {provider} to existing user {existing.username}")
        
        if not user:
            # Create new user
            user = await auth_repo.create_oauth_user(
                username=name,
                email=email,
                avatar_url=avatar,
                auth_provider=provider,
                provider_id=provider_id,
            )
            logger.info(f"Created new user {user.username} via {provider}")
    
    # Update last login
    await auth_repo.update_last_login(user.id)
    
    # Generate JWT
    access_token = create_access_token(user.username, user.is_admin)
    
    # Create redirect response with cookie
    redirect_url = f"{settings.oauth_redirect_url}/dashboard"
    redirect_response = RedirectResponse(url=redirect_url, status_code=302)
    
    # Set session cookie
    redirect_response.set_cookie(
        key="session",
        value=access_token,
        httponly=True,
        secure=settings.https_enabled,
        samesite="lax",
        max_age=settings.access_token_expire_minutes * 60,
        domain=settings.domain,
    )
    
    logger.info(f"OAuth login successful for user {user.username} via {provider}")
    return redirect_response
