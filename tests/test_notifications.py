"""Tests for notification system endpoints."""

from __future__ import annotations

from datetime import UTC, datetime

from fastapi import status
from fastapi.testclient import TestClient

from app.api.dependencies import require_user
from app.core.security import TokenData


def _create_mock_user_override(username: str = "test_user"):
    """Create a dependency override that returns a mock user token."""
    async def override():
        return TokenData(
            sub=username,
            exp=datetime.now(UTC),
            iat=datetime.now(UTC),
            iss="stonkmarket",
            aud="stonkmarket-api",
            jti="test-jti",
            is_admin=False,
        )
    return override


# =============================================================================
# CHANNEL ENDPOINTS - AUTH TESTS
# =============================================================================


class TestChannelEndpointsAuth:
    """Test authentication requirements for channel endpoints."""

    def test_list_channels_without_auth_returns_401(self, client: TestClient):
        """GET /notifications/channels without auth returns 401."""
        response = client.get("/notifications/channels")
        assert response.status_code == status.HTTP_401_UNAUTHORIZED

    def test_create_channel_without_auth_returns_401(self, client: TestClient):
        """POST /notifications/channels without auth returns 401."""
        response = client.post("/notifications/channels", json={
            "name": "Test Channel",
            "channel_type": "discord",
            "apprise_url": "discord://webhook_id/token",
        })
        assert response.status_code == status.HTTP_401_UNAUTHORIZED

    def test_delete_channel_without_auth_returns_401(self, client: TestClient):
        """DELETE /notifications/channels/{id} without auth returns 401."""
        response = client.delete("/notifications/channels/1")
        assert response.status_code == status.HTTP_401_UNAUTHORIZED


class TestCreateChannelValidation:
    """Tests for POST /notifications/channels validation (no DB needed)."""

    def test_create_channel_validates_name(self, client: TestClient):
        """POST /notifications/channels validates name field."""
        client.app.dependency_overrides[require_user] = _create_mock_user_override()
        try:
            response = client.post("/notifications/channels", json={
                "name": "",  # Empty name
                "channel_type": "discord",
                "apprise_url": "discord://webhook_id/token",
            })
            assert response.status_code == status.HTTP_422_UNPROCESSABLE_CONTENT
        finally:
            client.app.dependency_overrides.clear()

    def test_create_channel_validates_channel_type(self, client: TestClient):
        """POST /notifications/channels validates channel_type enum."""
        client.app.dependency_overrides[require_user] = _create_mock_user_override()
        try:
            response = client.post("/notifications/channels", json={
                "name": "Test Channel",
                "channel_type": "invalid_type",
                "apprise_url": "some://url-that-is-long-enough",
            })
            assert response.status_code == status.HTTP_422_UNPROCESSABLE_CONTENT
        finally:
            client.app.dependency_overrides.clear()

    def test_create_channel_validates_apprise_url_length(self, client: TestClient):
        """POST /notifications/channels validates apprise_url minimum length."""
        client.app.dependency_overrides[require_user] = _create_mock_user_override()
        try:
            response = client.post("/notifications/channels", json={
                "name": "Test Channel",
                "channel_type": "discord",
                "apprise_url": "short",  # Too short (min 10 chars)
            })
            assert response.status_code == status.HTTP_422_UNPROCESSABLE_CONTENT
        finally:
            client.app.dependency_overrides.clear()


# =============================================================================
# RULE ENDPOINTS - AUTH TESTS
# =============================================================================


class TestRuleEndpointsAuth:
    """Test authentication requirements for rule endpoints."""

    def test_list_rules_without_auth_returns_401(self, client: TestClient):
        """GET /notifications/rules without auth returns 401."""
        response = client.get("/notifications/rules")
        assert response.status_code == status.HTTP_401_UNAUTHORIZED

    def test_create_rule_without_auth_returns_401(self, client: TestClient):
        """POST /notifications/rules without auth returns 401."""
        response = client.post("/notifications/rules", json={
            "name": "Test Rule",
            "channel_id": 1,
            "trigger_type": "DIP_EXCEEDS_PERCENT",
        })
        assert response.status_code == status.HTTP_401_UNAUTHORIZED

    def test_delete_rule_without_auth_returns_401(self, client: TestClient):
        """DELETE /notifications/rules/{id} without auth returns 401."""
        response = client.delete("/notifications/rules/1")
        assert response.status_code == status.HTTP_401_UNAUTHORIZED


class TestCreateRuleValidation:
    """Tests for POST /notifications/rules validation (no DB needed)."""

    def test_create_rule_validates_trigger_type(self, client: TestClient):
        """POST /notifications/rules validates trigger_type enum."""
        client.app.dependency_overrides[require_user] = _create_mock_user_override()
        try:
            response = client.post("/notifications/rules", json={
                "name": "Test Rule",
                "channel_id": 1,
                "trigger_type": "INVALID_TRIGGER",
            })
            assert response.status_code == status.HTTP_422_UNPROCESSABLE_CONTENT
        finally:
            client.app.dependency_overrides.clear()

    def test_create_rule_validates_cooldown_range(self, client: TestClient):
        """POST /notifications/rules validates cooldown_minutes range."""
        client.app.dependency_overrides[require_user] = _create_mock_user_override()
        try:
            # Too low (min is 5)
            response = client.post("/notifications/rules", json={
                "name": "Test Rule",
                "channel_id": 1,
                "trigger_type": "DIP_EXCEEDS_PERCENT",
                "cooldown_minutes": 1,
            })
            assert response.status_code == status.HTTP_422_UNPROCESSABLE_CONTENT
        finally:
            client.app.dependency_overrides.clear()


# =============================================================================
# HISTORY & SUMMARY - AUTH TESTS
# =============================================================================


class TestHistoryEndpointsAuth:
    """Test authentication for history endpoints."""

    def test_get_history_without_auth_returns_401(self, client: TestClient):
        """GET /notifications/history without auth returns 401."""
        response = client.get("/notifications/history")
        assert response.status_code == status.HTTP_401_UNAUTHORIZED

    def test_get_summary_without_auth_returns_401(self, client: TestClient):
        """GET /notifications/summary without auth returns 401."""
        response = client.get("/notifications/summary")
        assert response.status_code == status.HTTP_401_UNAUTHORIZED


class TestTriggerTypesEndpoint:
    """Tests for the trigger types endpoint."""

    def test_get_trigger_types_returns_list(self, client: TestClient):
        """GET /notifications/trigger-types returns list of trigger types (no auth required)."""
        # This endpoint is public since it only returns metadata
        response = client.get("/notifications/trigger-types")
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert isinstance(data, list)
        assert len(data) > 0
        # Check first item has expected fields
        first_type = data[0]
        assert "value" in first_type
        assert "category" in first_type
        assert "default_operator" in first_type

    def test_trigger_types_include_all_categories(self, client: TestClient):
        """GET /notifications/trigger-types includes triggers from all categories."""
        response = client.get("/notifications/trigger-types")
        data = response.json()
        categories = {t["category"] for t in data}
        expected_categories = {
            "Price & Dip",
            "Signals",
            "Fundamentals",
            "AI Analysis",
            "Portfolio",
            "Watchlist",
        }
        assert categories == expected_categories


# =============================================================================
# SCHEMA VALIDATION
# =============================================================================


class TestSchemaValidation:
    """Tests for schema validation."""

    def test_trigger_type_enum_values(self):
        """Verify TriggerType enum contains expected values."""
        from app.schemas.notifications import TriggerType
        
        # Price & Dip
        assert TriggerType.PRICE_DROPS_BELOW == "PRICE_DROPS_BELOW"
        assert TriggerType.DIP_EXCEEDS_PERCENT == "DIP_EXCEEDS_PERCENT"
        
        # Signals
        assert TriggerType.DIPFINDER_ALERT == "DIPFINDER_ALERT"
        
        # AI
        assert TriggerType.AI_RATING_STRONG_BUY == "AI_RATING_STRONG_BUY"
        
        # Portfolio
        assert TriggerType.PORTFOLIO_DRAWDOWN_EXCEEDS == "PORTFOLIO_DRAWDOWN_EXCEEDS"

    def test_comparison_operator_enum_values(self):
        """Verify ComparisonOperator enum contains expected values."""
        from app.schemas.notifications import ComparisonOperator
        
        assert ComparisonOperator.GT == "GT"
        assert ComparisonOperator.LT == "LT"
        assert ComparisonOperator.GTE == "GTE"
        assert ComparisonOperator.LTE == "LTE"
        assert ComparisonOperator.EQ == "EQ"
        assert ComparisonOperator.NEQ == "NEQ"
        assert ComparisonOperator.CHANGE == "CHANGE"

    def test_channel_type_enum_values(self):
        """Verify ChannelType enum contains expected values."""
        from app.schemas.notifications import ChannelType
        
        assert ChannelType.DISCORD == "discord"
        assert ChannelType.TELEGRAM == "telegram"
        assert ChannelType.EMAIL == "email"
        assert ChannelType.SLACK == "slack"
        assert ChannelType.PUSHOVER == "pushover"
        assert ChannelType.NTFY == "ntfy"
        assert ChannelType.WEBHOOK == "webhook"

    def test_rule_priority_enum_values(self):
        """Verify RulePriority enum contains expected values."""
        from app.schemas.notifications import RulePriority
        
        assert RulePriority.LOW == "low"
        assert RulePriority.NORMAL == "normal"
        assert RulePriority.HIGH == "high"
        assert RulePriority.CRITICAL == "critical"

    def test_trigger_type_info_registry(self):
        """Verify TRIGGER_TYPE_INFO contains metadata for all trigger types."""
        from app.schemas.notifications import TriggerType, TRIGGER_TYPE_INFO
        
        # All trigger types should have info
        for trigger_type in TriggerType:
            assert trigger_type in TRIGGER_TYPE_INFO, f"Missing info for {trigger_type}"
            info = TRIGGER_TYPE_INFO[trigger_type]
            assert info.type == trigger_type
            assert info.name
            assert info.description
            assert info.category

    def test_trigger_type_info_categories(self):
        """Verify all expected categories exist in TRIGGER_TYPE_INFO."""
        from app.schemas.notifications import TRIGGER_TYPE_INFO
        
        categories = {info.category for info in TRIGGER_TYPE_INFO.values()}
        expected_categories = {
            "Price & Dip",
            "Signals",
            "Fundamentals",
            "AI Analysis",
            "Portfolio",
            "Watchlist",
        }
        assert categories == expected_categories


# =============================================================================
# SERVICE LAYER TESTS
# =============================================================================


class TestTriggerEvaluation:
    """Tests for trigger evaluation logic."""

    def test_compare_values_greater_than(self):
        """Test GT comparison operator."""
        from app.services.notifications.triggers import compare_values
        from app.schemas.notifications import ComparisonOperator
        
        assert compare_values(10.0, 5.0, ComparisonOperator.GT) is True
        assert compare_values(5.0, 10.0, ComparisonOperator.GT) is False
        assert compare_values(5.0, 5.0, ComparisonOperator.GT) is False

    def test_compare_values_less_than(self):
        """Test LT comparison operator."""
        from app.services.notifications.triggers import compare_values
        from app.schemas.notifications import ComparisonOperator
        
        assert compare_values(5.0, 10.0, ComparisonOperator.LT) is True
        assert compare_values(10.0, 5.0, ComparisonOperator.LT) is False
        assert compare_values(5.0, 5.0, ComparisonOperator.LT) is False

    def test_compare_values_greater_than_or_equal(self):
        """Test GTE comparison operator."""
        from app.services.notifications.triggers import compare_values
        from app.schemas.notifications import ComparisonOperator
        
        assert compare_values(10.0, 5.0, ComparisonOperator.GTE) is True
        assert compare_values(5.0, 5.0, ComparisonOperator.GTE) is True
        assert compare_values(5.0, 10.0, ComparisonOperator.GTE) is False

    def test_compare_values_less_than_or_equal(self):
        """Test LTE comparison operator."""
        from app.services.notifications.triggers import compare_values
        from app.schemas.notifications import ComparisonOperator
        
        assert compare_values(5.0, 10.0, ComparisonOperator.LTE) is True
        assert compare_values(5.0, 5.0, ComparisonOperator.LTE) is True
        assert compare_values(10.0, 5.0, ComparisonOperator.LTE) is False

    def test_compare_values_equal(self):
        """Test EQ comparison operator."""
        from app.services.notifications.triggers import compare_values
        from app.schemas.notifications import ComparisonOperator
        
        assert compare_values(5.0, 5.0, ComparisonOperator.EQ) is True
        assert compare_values(5.0, 5.01, ComparisonOperator.EQ) is False

    def test_compare_values_not_equal(self):
        """Test NEQ comparison operator."""
        from app.services.notifications.triggers import compare_values
        from app.schemas.notifications import ComparisonOperator
        
        assert compare_values(5.0, 10.0, ComparisonOperator.NEQ) is True
        assert compare_values(5.0, 5.0, ComparisonOperator.NEQ) is False


class TestMessageBuilder:
    """Tests for notification message building."""

    def test_format_trigger_value_percent(self):
        """Test formatting percent values."""
        from app.services.notifications.message_builder import format_trigger_value
        
        result = format_trigger_value(15.5, "%")
        assert "15.5" in result
        assert "%" in result

    def test_format_trigger_value_price(self):
        """Test formatting price values."""
        from app.services.notifications.message_builder import format_trigger_value
        
        result = format_trigger_value(123.45, "$")
        assert "123.45" in result
        assert "$" in result

    def test_format_trigger_value_days(self):
        """Test formatting day values."""
        from app.services.notifications.message_builder import format_trigger_value
        
        result = format_trigger_value(7, "days")
        assert "7" in result
        assert "day" in result.lower()

    def test_format_trigger_value_score(self):
        """Test formatting score values."""
        from app.services.notifications.message_builder import format_trigger_value
        
        result = format_trigger_value(85.0, "score")
        assert "85" in result


class TestSafetyModule:
    """Tests for notification safety features."""

    def test_check_staleness_fresh_data(self):
        """Test staleness check with fresh data."""
        from app.services.notifications.safety import check_staleness
        from datetime import datetime, timedelta, UTC
        
        # Recent data (1 hour old)
        last_updated = datetime.now(UTC) - timedelta(hours=1)
        is_stale, age = check_staleness(last_updated)
        assert is_stale is False  # Not stale
        assert age is not None
        assert age < 2  # Less than 2 hours old

    def test_check_staleness_stale_data(self):
        """Test staleness check with stale data."""
        from app.services.notifications.safety import check_staleness
        from datetime import datetime, timedelta, UTC
        
        # Old data (48 hours old)
        last_updated = datetime.now(UTC) - timedelta(hours=48)
        is_stale, age = check_staleness(last_updated)
        assert is_stale is True  # Is stale
        assert age is not None
        assert age > 24  # More than 24 hours

    def test_compute_content_hash(self):
        """Test content hash computation."""
        from app.services.notifications.safety import compute_content_hash
        
        hash1 = compute_content_hash("Title", "Body")
        hash2 = compute_content_hash("Title", "Body")
        hash3 = compute_content_hash("Title", "Different")
        
        # Same content = same hash
        assert hash1 == hash2
        # Different content = different hash
        assert hash1 != hash3
        # Hash is SHA-256 hex digest (64 chars)
        assert len(hash1) == 64


class TestTriggerInfo:
    """Tests for trigger type information."""

    def test_get_trigger_info_exists(self):
        """Test getting info for existing trigger type."""
        from app.schemas.notifications import TriggerType, TRIGGER_TYPE_INFO
        
        info = TRIGGER_TYPE_INFO.get(TriggerType.DIP_EXCEEDS_PERCENT)
        assert info is not None
        assert info.type == TriggerType.DIP_EXCEEDS_PERCENT
        assert info.category == "Price & Dip"
        # value_unit is "percent" in the schema
        assert info.value_unit == "percent"

    def test_get_required_data_fields(self):
        """Test getting required data fields for trigger types."""
        from app.services.notifications.triggers import get_required_data_fields
        from app.schemas.notifications import TriggerType
        
        # Price trigger needs price field
        fields = get_required_data_fields(TriggerType.PRICE_DROPS_BELOW.value)
        assert "current_price" in fields
        
        # AI trigger needs AI data
        fields = get_required_data_fields(TriggerType.AI_RATING_STRONG_BUY.value)
        assert "ai_rating" in fields

