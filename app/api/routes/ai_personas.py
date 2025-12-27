"""AI Persona management routes.

Provides endpoints for managing AI investor personas (Warren Buffett, Peter Lynch, etc.)
including avatar image upload and configuration.
"""

from __future__ import annotations

import base64
from typing import Annotated

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile
from pydantic import BaseModel, ConfigDict
from sqlalchemy import select, update

from app.api.dependencies import require_admin
from app.core.logging import get_logger
from app.database.connection import get_session
from app.database.orm import AIPersona
from app.services.image_optimizer import optimize_for_avatar


logger = get_logger("api.routes.ai_personas")

router = APIRouter(prefix="/ai-personas", tags=["AI Personas"])


# =============================================================================
# SCHEMAS
# =============================================================================


class PersonaResponse(BaseModel):
    """AI persona response model."""
    model_config = ConfigDict(from_attributes=True)
    
    id: int
    key: str
    name: str
    description: str | None
    philosophy: str | None
    has_avatar: bool
    avatar_url: str | None
    is_active: bool
    display_order: int


class PersonaUpdateRequest(BaseModel):
    """Request to update persona details."""
    name: str | None = None
    description: str | None = None
    philosophy: str | None = None
    is_active: bool | None = None
    display_order: int | None = None


# =============================================================================
# PUBLIC ENDPOINTS
# =============================================================================


@router.get("", response_model=list[PersonaResponse])
async def list_personas(
    active_only: bool = Query(True, description="Only return active personas"),
):
    """
    List all AI personas.
    
    Public endpoint - returns persona info for display in UI.
    """
    async with get_session() as session:
        query = select(AIPersona)
        if active_only:
            query = query.where(AIPersona.is_active == True)
        query = query.order_by(AIPersona.display_order, AIPersona.name)
        
        result = await session.execute(query)
        personas = result.scalars().all()
        
        return [
            PersonaResponse(
                id=p.id,
                key=p.key,
                name=p.name,
                description=p.description,
                philosophy=p.philosophy,
                has_avatar=p.avatar_data is not None,
                avatar_url=f"/api/ai-personas/{p.key}/avatar" if p.avatar_data else None,
                is_active=p.is_active,
                display_order=p.display_order,
            )
            for p in personas
        ]


@router.get("/{persona_key}")
async def get_persona(persona_key: str):
    """
    Get a specific AI persona by key.
    
    Public endpoint.
    """
    async with get_session() as session:
        result = await session.execute(
            select(AIPersona).where(AIPersona.key == persona_key)
        )
        persona = result.scalar_one_or_none()
        
        if not persona:
            raise HTTPException(status_code=404, detail="Persona not found")
        
        return PersonaResponse(
            id=persona.id,
            key=persona.key,
            name=persona.name,
            description=persona.description,
            philosophy=persona.philosophy,
            has_avatar=persona.avatar_data is not None,
            avatar_url=f"/api/ai-personas/{persona.key}/avatar" if persona.avatar_data else None,
            is_active=persona.is_active,
            display_order=persona.display_order,
        )


@router.get("/{persona_key}/avatar")
async def get_persona_avatar(persona_key: str):
    """
    Get the avatar image for a persona.
    
    Returns the optimized WebP image directly.
    """
    from fastapi.responses import Response
    
    async with get_session() as session:
        result = await session.execute(
            select(AIPersona.avatar_data, AIPersona.avatar_mime_type)
            .where(AIPersona.key == persona_key)
        )
        row = result.one_or_none()
        
        if not row or not row.avatar_data:
            raise HTTPException(status_code=404, detail="Avatar not found")
        
        return Response(
            content=row.avatar_data,
            media_type=row.avatar_mime_type or "image/webp",
            headers={
                "Cache-Control": "public, max-age=86400",  # Cache for 24 hours
            }
        )


# =============================================================================
# ADMIN ENDPOINTS
# =============================================================================


@router.put(
    "/{persona_key}",
    response_model=PersonaResponse,
    dependencies=[Depends(require_admin)],
)
async def update_persona(
    persona_key: str,
    request: PersonaUpdateRequest,
):
    """
    Update AI persona details.
    
    Admin only.
    """
    async with get_session() as session:
        result = await session.execute(
            select(AIPersona).where(AIPersona.key == persona_key)
        )
        persona = result.scalar_one_or_none()
        
        if not persona:
            raise HTTPException(status_code=404, detail="Persona not found")
        
        # Update fields
        update_data = {}
        if request.name is not None:
            update_data["name"] = request.name
        if request.description is not None:
            update_data["description"] = request.description
        if request.philosophy is not None:
            update_data["philosophy"] = request.philosophy
        if request.is_active is not None:
            update_data["is_active"] = request.is_active
        if request.display_order is not None:
            update_data["display_order"] = request.display_order
        
        if update_data:
            await session.execute(
                update(AIPersona)
                .where(AIPersona.key == persona_key)
                .values(**update_data)
            )
            await session.commit()
            
            # Refresh
            result = await session.execute(
                select(AIPersona).where(AIPersona.key == persona_key)
            )
            persona = result.scalar_one()
        
        return PersonaResponse(
            id=persona.id,
            key=persona.key,
            name=persona.name,
            description=persona.description,
            philosophy=persona.philosophy,
            has_avatar=persona.avatar_data is not None,
            avatar_url=f"/api/ai-personas/{persona.key}/avatar" if persona.avatar_data else None,
            is_active=persona.is_active,
            display_order=persona.display_order,
        )


@router.post(
    "/{persona_key}/avatar",
    response_model=dict,
    dependencies=[Depends(require_admin)],
)
async def upload_persona_avatar(
    persona_key: str,
    file: Annotated[UploadFile, File(description="Avatar image file")],
    size: int = Query(128, ge=32, le=256, description="Avatar size in pixels"),
):
    """
    Upload an avatar image for an AI persona.
    
    The image will be:
    - Cropped to square (center crop)
    - Resized to the specified size
    - Converted to WebP for optimal compression
    
    Admin only.
    """
    async with get_session() as session:
        # Check persona exists
        result = await session.execute(
            select(AIPersona).where(AIPersona.key == persona_key)
        )
        persona = result.scalar_one_or_none()
        
        if not persona:
            raise HTTPException(status_code=404, detail="Persona not found")
        
        # Read and optimize image
        try:
            image_data = await file.read()
            avatar_result = optimize_for_avatar(image_data, size=size)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            logger.error(f"Failed to process avatar: {e}")
            raise HTTPException(status_code=400, detail="Failed to process image")
        
        # Save to database
        await session.execute(
            update(AIPersona)
            .where(AIPersona.key == persona_key)
            .values(
                avatar_data=avatar_result.data,
                avatar_mime_type=avatar_result.mime_type,
            )
        )
        await session.commit()
        
        logger.info(
            f"Avatar uploaded for {persona_key}: "
            f"{avatar_result.optimized_size / 1024:.1f}KB, {avatar_result.size}x{avatar_result.size}"
        )
        
        return {
            "message": "Avatar uploaded successfully",
            "persona_key": persona_key,
            "size": avatar_result.size,
            "file_size_kb": round(avatar_result.optimized_size / 1024, 1),
            "savings_percent": round(avatar_result.savings_percent, 0),
            "avatar_url": f"/api/ai-personas/{persona_key}/avatar",
        }


@router.delete(
    "/{persona_key}/avatar",
    response_model=dict,
    dependencies=[Depends(require_admin)],
)
async def delete_persona_avatar(persona_key: str):
    """
    Delete the avatar image for an AI persona.
    
    Admin only.
    """
    async with get_session() as session:
        result = await session.execute(
            select(AIPersona).where(AIPersona.key == persona_key)
        )
        persona = result.scalar_one_or_none()
        
        if not persona:
            raise HTTPException(status_code=404, detail="Persona not found")
        
        if not persona.avatar_data:
            raise HTTPException(status_code=404, detail="No avatar to delete")
        
        await session.execute(
            update(AIPersona)
            .where(AIPersona.key == persona_key)
            .values(avatar_data=None, avatar_mime_type=None)
        )
        await session.commit()
        
        logger.info(f"Avatar deleted for {persona_key}")
        
        return {"message": "Avatar deleted successfully", "persona_key": persona_key}
