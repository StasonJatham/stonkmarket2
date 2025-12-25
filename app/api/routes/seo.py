"""
SEO Routes - Sitemap and robots.txt generation

Provides dynamic sitemap.xml generation for the SPA,
including all indexed routes and stock-specific pages.
"""

from datetime import datetime, timezone
from typing import Optional
from fastapi import APIRouter, Response
from fastapi.responses import PlainTextResponse

from app.core.logging import get_logger

logger = get_logger("api.routes.seo")

router = APIRouter(tags=["SEO"])

SITE_URL = "https://stonkmarket.app"

# Static routes that should always be in sitemap
STATIC_ROUTES = [
    {"loc": "/", "priority": "1.0", "changefreq": "hourly"},
    {"loc": "/swipe", "priority": "0.9", "changefreq": "hourly"},
    {"loc": "/suggest", "priority": "0.7", "changefreq": "daily"},
    {"loc": "/about", "priority": "0.8", "changefreq": "weekly"},
    {"loc": "/privacy", "priority": "0.3", "changefreq": "monthly"},
    {"loc": "/imprint", "priority": "0.3", "changefreq": "monthly"},
    {"loc": "/contact", "priority": "0.4", "changefreq": "monthly"},
]


def format_date(dt: Optional[datetime]) -> str:
    """Format datetime as W3C date for sitemap."""
    if dt is None:
        dt = datetime.now(timezone.utc)
    return dt.strftime("%Y-%m-%d")


@router.get(
    "/sitemap.xml",
    response_class=PlainTextResponse,
    summary="Generate sitemap.xml",
    description="Dynamically generates sitemap.xml with all indexed routes and stock pages.",
    include_in_schema=False,  # Hide from OpenAPI since it's for crawlers
)
async def get_sitemap() -> Response:
    """
    Generate sitemap.xml dynamically.
    
    Includes:
    - All static SPA routes (dashboard, swipe, privacy, etc.)
    - Dynamic stock pages if we add /stocks/:symbol routes in future
    """
    now = format_date(datetime.now(timezone.utc))
    
    # Start XML
    xml_parts = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">',
    ]
    
    # Add static routes
    for route in STATIC_ROUTES:
        xml_parts.append(f"""  <url>
    <loc>{SITE_URL}{route['loc']}</loc>
    <lastmod>{now}</lastmod>
    <changefreq>{route['changefreq']}</changefreq>
    <priority>{route['priority']}</priority>
  </url>""")
    
    # In future, add dynamic stock pages here when we create /stocks/:symbol routes
    # Example:
    # try:
    #     symbols = await fetch_all("SELECT symbol, updated_at FROM symbols WHERE active = true")
    #     for sym in symbols:
    #         xml_parts.append(f"""  <url>
    #     <loc>{SITE_URL}/stocks/{sym['symbol'].lower()}</loc>
    #     <lastmod>{format_date(sym.get('updated_at'))}</lastmod>
    #     <changefreq>daily</changefreq>
    #     <priority>0.6</priority>
    #   </url>""")
    # except Exception as e:
    #     logger.warning(f"Failed to fetch symbols for sitemap: {e}")
    
    xml_parts.append("</urlset>")
    
    xml_content = "\n".join(xml_parts)
    
    return Response(
        content=xml_content,
        media_type="application/xml",
        headers={
            "Cache-Control": "public, max-age=3600",  # Cache for 1 hour
        }
    )


@router.get(
    "/robots.txt",
    response_class=PlainTextResponse,
    summary="Generate robots.txt",
    description="Dynamic robots.txt with sitemap reference.",
    include_in_schema=False,
)
async def get_robots() -> Response:
    """
    Generate robots.txt dynamically.
    
    This supplements the static robots.txt in /public for when
    the API serves the frontend (e.g., in production with reverse proxy).
    """
    robots_content = f"""# Robots.txt for StonkMarket
User-agent: *
Allow: /

# Sitemap
Sitemap: {SITE_URL}/sitemap.xml

# Disallow admin and API routes
Disallow: /admin
Disallow: /api/
Disallow: /login

# Allow specific API routes for structured data
Allow: /api/sitemap.xml

# Crawl-delay (be nice to the server)
Crawl-delay: 1
"""
    
    return Response(
        content=robots_content,
        media_type="text/plain",
        headers={
            "Cache-Control": "public, max-age=86400",  # Cache for 24 hours
        }
    )
