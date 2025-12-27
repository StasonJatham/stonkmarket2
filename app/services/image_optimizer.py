"""Image optimization service for portfolio screenshot processing.

Converts and optimizes images before AI analysis to:
1. Reduce OpenAI API costs (smaller images = fewer tokens)
2. Support all formats including HEIC (iPhone photos)
3. Maintain consistent quality across all inputs

Supported input formats:
- JPEG, PNG, WebP, GIF, BMP, TIFF
- HEIC/HEIF (iPhone photos) via pillow-heif
- PDF (first page only)

Output: Optimized JPEG with configurable quality
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from typing import BinaryIO

from PIL import Image, ExifTags

from app.core.logging import get_logger


logger = get_logger("services.image_optimizer")


# =============================================================================
# Configuration
# =============================================================================

# Target max dimension (width or height)
# GPT-4V works well with images up to 2000px on the long side
MAX_DIMENSION = 1500

# JPEG quality (1-100, higher = better quality but larger file)
# 85 is a good balance for text/screenshot readability
JPEG_QUALITY = 85

# Max file size in bytes (target, may exceed slightly)
MAX_FILE_SIZE = 1 * 1024 * 1024  # 1MB target

# Min quality to try before giving up on size target
MIN_QUALITY = 60


# =============================================================================
# HEIC Support
# =============================================================================

def _register_heif_opener() -> bool:
    """Register HEIF/HEIC opener with Pillow if available."""
    try:
        import pillow_heif
        pillow_heif.register_heif_opener()
        return True
    except ImportError:
        logger.warning("pillow-heif not installed, HEIC support disabled")
        return False


# Register on module load
_HEIF_AVAILABLE = _register_heif_opener()


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class OptimizationResult:
    """Result of image optimization."""
    
    data: bytes
    mime_type: str
    original_size: int
    optimized_size: int
    original_format: str
    width: int
    height: int
    
    @property
    def compression_ratio(self) -> float:
        """How much smaller the optimized image is (0-1)."""
        if self.original_size == 0:
            return 0
        return 1 - (self.optimized_size / self.original_size)
    
    @property
    def savings_percent(self) -> float:
        """Percentage of size saved."""
        return self.compression_ratio * 100


# =============================================================================
# Image Processing
# =============================================================================

def _fix_orientation(img: Image.Image) -> Image.Image:
    """Fix image orientation based on EXIF data.
    
    iPhone photos often have rotation stored in EXIF rather than
    the actual pixel data. This corrects for that.
    """
    try:
        # Get EXIF data
        exif = img.getexif()
        if not exif:
            return img
        
        # Find orientation tag
        orientation_key = None
        for key, val in ExifTags.TAGS.items():
            if val == 'Orientation':
                orientation_key = key
                break
        
        if orientation_key is None or orientation_key not in exif:
            return img
        
        orientation = exif[orientation_key]
        
        # Apply rotation based on orientation
        rotations = {
            3: 180,
            6: 270,
            8: 90,
        }
        
        if orientation in rotations:
            img = img.rotate(rotations[orientation], expand=True)
        
        # Handle mirroring
        if orientation in (2, 4):
            img = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        if orientation in (5, 7):
            img = img.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
            
    except Exception as e:
        logger.debug(f"Could not fix orientation: {e}")
    
    return img


def _resize_image(img: Image.Image, max_dimension: int) -> Image.Image:
    """Resize image to fit within max dimension while preserving aspect ratio."""
    width, height = img.size
    
    # Check if resize needed
    if width <= max_dimension and height <= max_dimension:
        return img
    
    # Calculate new size
    if width > height:
        new_width = max_dimension
        new_height = int(height * (max_dimension / width))
    else:
        new_height = max_dimension
        new_width = int(width * (max_dimension / height))
    
    # Use LANCZOS for high-quality downscaling
    return img.resize((new_width, new_height), Image.Resampling.LANCZOS)


def _convert_to_rgb(img: Image.Image) -> Image.Image:
    """Convert image to RGB mode for JPEG compatibility."""
    if img.mode == 'RGBA':
        # Create white background
        background = Image.new('RGB', img.size, (255, 255, 255))
        background.paste(img, mask=img.split()[3])  # Use alpha as mask
        return background
    elif img.mode == 'P':
        # Palette mode - convert to RGBA first if has transparency
        if 'transparency' in img.info:
            img = img.convert('RGBA')
            return _convert_to_rgb(img)
        return img.convert('RGB')
    elif img.mode == 'L':
        # Grayscale
        return img.convert('RGB')
    elif img.mode != 'RGB':
        return img.convert('RGB')
    return img


def _compress_to_target_size(
    img: Image.Image,
    target_size: int,
    initial_quality: int = JPEG_QUALITY,
    min_quality: int = MIN_QUALITY,
) -> tuple[bytes, int]:
    """Compress image to target size, adjusting quality as needed.
    
    Returns (jpeg_bytes, final_quality).
    """
    quality = initial_quality
    
    while quality >= min_quality:
        buffer = io.BytesIO()
        img.save(
            buffer,
            format='JPEG',
            quality=quality,
            optimize=True,
            progressive=True,
        )
        data = buffer.getvalue()
        
        if len(data) <= target_size or quality == min_quality:
            return data, quality
        
        # Reduce quality and try again
        quality -= 5
    
    # Return best effort
    buffer = io.BytesIO()
    img.save(buffer, format='JPEG', quality=min_quality, optimize=True)
    return buffer.getvalue(), min_quality


# =============================================================================
# Main Function
# =============================================================================

def optimize_image(
    data: bytes | BinaryIO,
    max_dimension: int = MAX_DIMENSION,
    jpeg_quality: int = JPEG_QUALITY,
    max_file_size: int = MAX_FILE_SIZE,
) -> OptimizationResult:
    """
    Optimize an image for AI analysis.
    
    Args:
        data: Raw image bytes or file-like object
        max_dimension: Maximum width/height in pixels
        jpeg_quality: Initial JPEG quality (1-100)
        max_file_size: Target max file size in bytes
        
    Returns:
        OptimizationResult with optimized JPEG data
        
    Raises:
        ValueError: If image cannot be processed
    """
    # Handle bytes vs file-like object
    if isinstance(data, bytes):
        original_size = len(data)
        buffer = io.BytesIO(data)
    else:
        data.seek(0, 2)  # Seek to end
        original_size = data.tell()
        data.seek(0)
        buffer = data
    
    # Open image
    try:
        img = Image.open(buffer)
        original_format = img.format or "UNKNOWN"
    except Exception as e:
        raise ValueError(f"Cannot open image: {e}") from e
    
    logger.debug(
        f"Processing {original_format} image: {img.size[0]}x{img.size[1]}, "
        f"{original_size / 1024:.1f}KB"
    )
    
    # Fix orientation (important for iPhone photos)
    img = _fix_orientation(img)
    
    # Resize if needed
    img = _resize_image(img, max_dimension)
    
    # Convert to RGB for JPEG
    img = _convert_to_rgb(img)
    
    # Compress to JPEG
    jpeg_data, final_quality = _compress_to_target_size(
        img,
        target_size=max_file_size,
        initial_quality=jpeg_quality,
    )
    
    result = OptimizationResult(
        data=jpeg_data,
        mime_type="image/jpeg",
        original_size=original_size,
        optimized_size=len(jpeg_data),
        original_format=original_format,
        width=img.size[0],
        height=img.size[1],
    )
    
    logger.info(
        f"Optimized {original_format} ({original_size / 1024:.1f}KB) -> "
        f"JPEG ({result.optimized_size / 1024:.1f}KB), "
        f"saved {result.savings_percent:.0f}%"
    )
    
    return result


def optimize_image_for_vision(
    data: bytes | BinaryIO,
) -> tuple[bytes, str]:
    """
    Convenience function that returns just the optimized bytes and mime type.
    
    Returns:
        Tuple of (jpeg_bytes, "image/jpeg")
    """
    result = optimize_image(data)
    return result.data, result.mime_type


# =============================================================================
# Avatar Optimization
# =============================================================================

# Avatar configuration
AVATAR_SIZE = 128  # Target avatar size in pixels (square)
AVATAR_MAX_SIZE = 256  # Maximum avatar size
AVATAR_WEBP_QUALITY = 85


@dataclass
class AvatarResult:
    """Result of avatar optimization."""
    
    data: bytes
    mime_type: str
    original_size: int
    optimized_size: int
    size: int  # Square dimension (width = height)
    
    @property
    def savings_percent(self) -> float:
        """Percentage of size saved."""
        if self.original_size == 0:
            return 0
        return (1 - (self.optimized_size / self.original_size)) * 100


def _crop_to_square(img: Image.Image) -> Image.Image:
    """Crop image to square from center."""
    width, height = img.size
    
    if width == height:
        return img
    
    # Calculate crop box (center crop)
    if width > height:
        left = (width - height) // 2
        top = 0
        right = left + height
        bottom = height
    else:
        left = 0
        top = (height - width) // 2
        right = width
        bottom = top + width
    
    return img.crop((left, top, right, bottom))


def optimize_for_avatar(
    data: bytes | BinaryIO,
    size: int = AVATAR_SIZE,
    max_size: int = AVATAR_MAX_SIZE,
) -> AvatarResult:
    """
    Optimize an image for avatar/profile display.
    
    Creates a small, square WebP image optimized for avatar use cases.
    
    Features:
    - Crops to square (center crop)
    - Resizes to target size
    - Converts to WebP for best compression
    - Fixes EXIF orientation
    - Handles transparency (RGBA)
    
    Args:
        data: Raw image bytes or file-like object
        size: Target avatar size in pixels (square)
        max_size: Maximum allowed size (clips to this if size > max_size)
        
    Returns:
        AvatarResult with optimized WebP data
        
    Raises:
        ValueError: If image cannot be processed
    """
    # Validate size
    target_size = min(size, max_size)
    
    # Handle bytes vs file-like object
    if isinstance(data, bytes):
        original_size = len(data)
        buffer = io.BytesIO(data)
    else:
        data.seek(0, 2)  # Seek to end
        original_size = data.tell()
        data.seek(0)
        buffer = data
    
    # Open image
    try:
        img = Image.open(buffer)
        original_format = img.format or "UNKNOWN"
    except Exception as e:
        raise ValueError(f"Cannot open image: {e}") from e
    
    logger.debug(
        f"Processing avatar from {original_format}: {img.size[0]}x{img.size[1]}, "
        f"{original_size / 1024:.1f}KB"
    )
    
    # Fix orientation (important for iPhone photos)
    img = _fix_orientation(img)
    
    # Crop to square
    img = _crop_to_square(img)
    
    # Resize to target size
    if img.size[0] != target_size:
        img = img.resize((target_size, target_size), Image.Resampling.LANCZOS)
    
    # Convert to RGB if needed (keep RGBA for transparency)
    if img.mode == 'P':
        if 'transparency' in img.info:
            img = img.convert('RGBA')
        else:
            img = img.convert('RGB')
    elif img.mode not in ('RGB', 'RGBA'):
        img = img.convert('RGB')
    
    # Save as WebP
    output_buffer = io.BytesIO()
    img.save(
        output_buffer,
        format='WEBP',
        quality=AVATAR_WEBP_QUALITY,
        method=6,  # Highest compression effort
    )
    webp_data = output_buffer.getvalue()
    
    result = AvatarResult(
        data=webp_data,
        mime_type="image/webp",
        original_size=original_size,
        optimized_size=len(webp_data),
        size=target_size,
    )
    
    logger.info(
        f"Avatar optimized: {original_format} ({original_size / 1024:.1f}KB) -> "
        f"WebP {target_size}x{target_size} ({result.optimized_size / 1024:.1f}KB), "
        f"saved {result.savings_percent:.0f}%"
    )
    
    return result
