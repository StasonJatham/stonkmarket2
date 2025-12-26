"""Tests for settings behavior and validation.

Tests verify:
- Auto-approval gate conditions
- TTL=0 disabling cache
- DB pool size validation
- Auth secret production warning
- Settings wiring (cooldown, session TTL, etc.)
"""

from __future__ import annotations

import warnings
from unittest.mock import AsyncMock, patch, MagicMock

import pytest


class TestAutoApprovalGates:
    """Tests for auto-approval conditions in suggestions."""

    @pytest.mark.asyncio
    async def test_auto_approval_respects_enabled_flag(self):
        """Auto-approval should not trigger when auto_approve_enabled is False."""
        from app.api.routes.suggestions import _check_and_apply_auto_approval
        
        # Mock settings with auto_approve_enabled=False
        mock_settings = MagicMock()
        mock_settings.auto_approve_enabled = False
        mock_settings.auto_approve_votes = 10
        mock_settings.auto_approve_unique_voters = 3
        mock_settings.auto_approve_min_age_hours = 1
        
        with patch("app.core.config.settings", mock_settings):
            result = await _check_and_apply_auto_approval(
                suggestion_id=1,
                symbol="TEST",
                new_score=100,  # Way above threshold
            )
        
        assert result is False

    @pytest.mark.asyncio
    async def test_auto_approval_requires_unique_voters(self):
        """Auto-approval needs minimum unique voters."""
        from app.api.routes.suggestions import _check_and_apply_auto_approval
        
        mock_settings = MagicMock()
        mock_settings.auto_approve_enabled = True
        mock_settings.auto_approve_votes = 10
        mock_settings.auto_approve_unique_voters = 5
        mock_settings.auto_approve_min_age_hours = 1
        
        with patch("app.core.config.settings", mock_settings), \
             patch("app.api.routes.suggestions.suggestions_repo.get_unique_voter_count", return_value=2), \
             patch("app.api.routes.suggestions.get_runtime_setting", return_value=10):
            result = await _check_and_apply_auto_approval(
                suggestion_id=1,
                symbol="TEST",
                new_score=100,  # Above threshold
            )
        
        assert result is False

    @pytest.mark.asyncio
    async def test_auto_approval_requires_min_age(self):
        """Auto-approval needs minimum age in hours."""
        from app.api.routes.suggestions import _check_and_apply_auto_approval
        
        mock_settings = MagicMock()
        mock_settings.auto_approve_enabled = True
        mock_settings.auto_approve_votes = 10
        mock_settings.auto_approve_unique_voters = 3
        mock_settings.auto_approve_min_age_hours = 48
        
        # Mock repository methods: enough voters but too young
        with patch("app.core.config.settings", mock_settings), \
             patch("app.api.routes.suggestions.suggestions_repo.get_unique_voter_count", return_value=10), \
             patch("app.api.routes.suggestions.suggestions_repo.get_suggestion_age_hours", return_value=12.0), \
             patch("app.api.routes.suggestions.get_runtime_setting", return_value=10):
            result = await _check_and_apply_auto_approval(
                suggestion_id=1,
                symbol="TEST",
                new_score=100,
            )
        
        assert result is False


class TestCacheTTLBehavior:
    """Tests for cache TTL=0 behavior."""

    @pytest.mark.asyncio
    async def test_ttl_zero_disables_caching(self):
        """TTL=0 should skip cache storage entirely."""
        from app.cache.cache import Cache
        
        cache = Cache(prefix="test", default_ttl=300)
        
        # Mock the valkey client
        mock_client = AsyncMock()
        
        with patch("app.cache.cache.get_valkey_client", return_value=mock_client):
            # Set with TTL=0 should return True but not call set()
            result = await cache.set("key", "value", ttl=0)
        
        assert result is True
        mock_client.set.assert_not_called()

    @pytest.mark.asyncio
    async def test_ttl_none_uses_default(self):
        """TTL=None should use default_ttl."""
        from app.cache.cache import Cache
        
        cache = Cache(prefix="test", default_ttl=300)
        
        mock_client = AsyncMock()
        
        with patch("app.cache.cache.get_valkey_client", return_value=mock_client):
            await cache.set("key", "value", ttl=None)
        
        # Should be called with default_ttl=300
        mock_client.set.assert_called_once()
        call_kwargs = mock_client.set.call_args
        assert call_kwargs.kwargs.get("ex") == 300

    @pytest.mark.asyncio
    async def test_default_ttl_zero_disables_caching(self):
        """Cache with default_ttl=0 should disable caching by default."""
        from app.cache.cache import Cache
        
        cache = Cache(prefix="test", default_ttl=0)
        
        mock_client = AsyncMock()
        
        with patch("app.cache.cache.get_valkey_client", return_value=mock_client):
            result = await cache.set("key", "value")  # Uses default_ttl=0
        
        assert result is True
        mock_client.set.assert_not_called()


class TestDbPoolValidation:
    """Tests for DB pool size validation."""

    def test_pool_max_less_than_min_raises(self):
        """db_pool_max_size < db_pool_min_size should raise ValueError."""
        from pydantic import ValidationError
        from app.core.config import Settings
        
        with pytest.raises(ValidationError) as exc_info:
            Settings(
                db_pool_min_size=20,
                db_pool_max_size=10,  # Invalid: less than min
            )
        
        assert "db_pool_max_size" in str(exc_info.value)

    def test_pool_equal_sizes_valid(self):
        """db_pool_max_size == db_pool_min_size should be valid."""
        from app.core.config import Settings
        
        settings = Settings(
            db_pool_min_size=10,
            db_pool_max_size=10,
        )
        
        assert settings.db_pool_min_size == 10
        assert settings.db_pool_max_size == 10


class TestAuthSecretGuard:
    """Tests for auth_secret production warning."""

    def test_default_secret_in_production_warns(self):
        """Using default auth_secret in production should warn."""
        from app.core.config import Settings
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            Settings(
                environment="production",
                auth_secret="dev-secret-please-change-in-production-min-32-chars",
            )
        
        # Check that a RuntimeWarning was issued
        assert len(w) >= 1
        assert any("auth_secret" in str(warning.message) for warning in w)
        assert any(issubclass(warning.category, RuntimeWarning) for warning in w)

    def test_custom_secret_in_production_no_warning(self):
        """Custom auth_secret in production should not warn."""
        from app.core.config import Settings
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            Settings(
                environment="production",
                auth_secret="my-super-secure-production-secret-key-123",
            )
        
        # Filter for our specific warning
        auth_warnings = [
            warning for warning in w 
            if "auth_secret" in str(warning.message)
        ]
        assert len(auth_warnings) == 0


class TestSettingsWiring:
    """Tests for settings being properly wired to code."""

    def test_vote_cooldown_reads_from_settings(self):
        """Vote cooldown function should return value from settings."""
        from app.core.config import settings
        
        # The function should return the configured value
        from app.core.client_identity import _get_vote_cooldown_days
        result = _get_vote_cooldown_days()
        
        # Should match the configured default (7 days)
        assert result == settings.vote_cooldown_days

    def test_session_ttl_reads_from_settings(self):
        """Session TTL function should return value from settings."""
        from app.core.config import settings
        
        from app.core.client_identity import _get_session_ttl_seconds
        result = _get_session_ttl_seconds()
        
        # Should match the configured default
        assert result == settings.session_ttl


class TestAutoApproveVotesDefault:
    """Tests for unified auto_approve_votes default."""

    def test_runtime_settings_default_matches_config(self):
        """Runtime settings default should match config default."""
        from app.services.runtime_settings import DEFAULT_SETTINGS
        from app.core.config import settings
        
        # Both should be 50 (unified)
        assert DEFAULT_SETTINGS["auto_approve_votes"] == 50
        assert settings.auto_approve_votes == 50
