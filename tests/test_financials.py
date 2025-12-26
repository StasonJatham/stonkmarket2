"""
Tests for financial statement fetching and domain-specific scoring.

Tests:
1. Financial statement fetching from yfinance
2. Helper functions for extracting financial metrics
3. Domain adapters using financial statement data (Bank, REIT, Insurance)
"""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from app.dipfinder.domain import Domain
from app.dipfinder.domain_scoring import (
    BankAdapter,
    REITAdapter,
    InsuranceAdapter,
    _get_financial_metric,
    _get_ffo,
    _get_net_interest_income,
    _get_loss_adjustment_expense,
    _get_total_revenue,
    _get_total_assets,
    _calculate_nim,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def bank_info_with_financials():
    """Mock info dict for a bank with financial statements."""
    return {
        "symbol": "JPM",
        "return_on_equity": 0.16,
        "return_on_assets": 0.013,
        "price_to_book": 1.8,
        "book_value": 125.0,
        "dividend_yield": 0.025,
        "recommendation_mean": 2.0,
        "current_price": 220.0,
        "financials": {
            "quarterly": {
                "income_statement": {
                    "Net Interest Income": 24_000_000_000,
                    "Interest Expense": 25_000_000_000,
                    "Interest Income": 49_000_000_000,
                    "Total Revenue": 46_000_000_000,
                    "Net Income": 14_000_000_000,
                },
                "balance_sheet": {
                    "Total Assets": 4_500_000_000_000,
                    "Total Liabilities": 4_200_000_000_000,
                },
                "cash_flow": {
                    "Operating Cash Flow": 20_000_000_000,
                },
            },
            "annual": {
                "income_statement": {},
                "balance_sheet": {},
                "cash_flow": {},
            },
        },
    }


@pytest.fixture
def bank_info_without_financials():
    """Mock info dict for a bank without financial statements."""
    return {
        "symbol": "JPM",
        "return_on_equity": 0.16,
        "return_on_assets": 0.013,
        "price_to_book": 1.8,
        "book_value": 125.0,
        "dividend_yield": 0.025,
        "recommendation_mean": 2.0,
    }


@pytest.fixture
def reit_info_with_financials():
    """Mock info dict for a REIT with financial statements."""
    return {
        "symbol": "O",
        "dividend_yield": 0.055,
        "price_to_book": 1.2,
        "debt_to_equity": 80,
        "recommendation_mean": 2.5,
        "current_price": 57.0,
        "shares_outstanding": 900_000_000,
        "financials": {
            "quarterly": {
                "income_statement": {
                    "Net Income": 315_000_000,
                    "Total Revenue": 1_200_000_000,
                },
                "balance_sheet": {},
                "cash_flow": {
                    "Depreciation Amortization Depletion": 632_000_000,
                    "Operating Cash Flow": 900_000_000,
                },
            },
            "annual": {
                "income_statement": {},
                "balance_sheet": {},
                "cash_flow": {},
            },
        },
    }


@pytest.fixture
def insurance_info_with_financials():
    """Mock info dict for an insurer with financial statements."""
    return {
        "symbol": "PGR",
        "return_on_equity": 0.25,
        "return_on_assets": 0.05,
        "price_to_book": 4.5,
        "book_value": 35.0,
        "dividend_yield": 0.02,
        "recommendation_mean": 2.2,
        "financials": {
            "quarterly": {
                "income_statement": {
                    "Total Revenue": 22_500_000_000,
                    "Net Policyholder Benefits And Claims": 14_390_000_000,
                    "Net Income": 2_620_000_000,
                },
                "balance_sheet": {},
                "cash_flow": {},
            },
            "annual": {
                "income_statement": {},
                "balance_sheet": {},
                "cash_flow": {},
            },
        },
    }


# =============================================================================
# Helper Function Tests
# =============================================================================

class TestFinancialMetricHelpers:
    """Tests for financial statement helper functions."""
    
    def test_get_financial_metric_quarterly_income(self, bank_info_with_financials):
        """Test extracting metric from quarterly income statement."""
        nii = _get_financial_metric(
            bank_info_with_financials,
            "income_statement",
            "Net Interest Income",
            "quarterly",
        )
        assert nii == 24_000_000_000
    
    def test_get_financial_metric_quarterly_balance(self, bank_info_with_financials):
        """Test extracting metric from quarterly balance sheet."""
        assets = _get_financial_metric(
            bank_info_with_financials,
            "balance_sheet",
            "Total Assets",
            "quarterly",
        )
        assert assets == 4_500_000_000_000
    
    def test_get_financial_metric_missing_financials(self):
        """Test returns None when financials key is missing."""
        info = {"symbol": "TEST"}
        result = _get_financial_metric(info, "income_statement", "Net Income", "quarterly")
        assert result is None
    
    def test_get_financial_metric_missing_metric(self, bank_info_with_financials):
        """Test returns None for non-existent metric."""
        result = _get_financial_metric(
            bank_info_with_financials,
            "income_statement",
            "NonexistentMetric",
            "quarterly",
        )
        assert result is None
    
    def test_get_net_interest_income(self, bank_info_with_financials):
        """Test NII extraction helper."""
        nii = _get_net_interest_income(bank_info_with_financials)
        assert nii == 24_000_000_000
    
    def test_get_total_assets(self, bank_info_with_financials):
        """Test total assets extraction helper."""
        assets = _get_total_assets(bank_info_with_financials)
        assert assets == 4_500_000_000_000
    
    def test_get_total_revenue(self, bank_info_with_financials):
        """Test total revenue extraction helper."""
        revenue = _get_total_revenue(bank_info_with_financials)
        assert revenue == 46_000_000_000
    
    def test_calculate_nim(self, bank_info_with_financials):
        """Test NIM calculation."""
        nim = _calculate_nim(bank_info_with_financials)
        # NIM = (NII * 4) / Total Assets = (24B * 4) / 4.5T ≈ 2.13%
        assert nim is not None
        assert 0.02 < nim < 0.025
    
    def test_calculate_nim_missing_data(self):
        """Test NIM returns None when data is missing."""
        info = {"symbol": "TEST"}
        nim = _calculate_nim(info)
        assert nim is None
    
    def test_get_ffo(self, reit_info_with_financials):
        """Test FFO calculation for REIT."""
        ffo = _get_ffo(reit_info_with_financials)
        # FFO = Net Income + D&A = 315M + 632M = 947M
        assert ffo == 947_000_000
    
    def test_get_ffo_missing_depreciation(self):
        """Test FFO returns None when depreciation is missing."""
        info = {
            "financials": {
                "quarterly": {
                    "income_statement": {"Net Income": 100_000_000},
                    "cash_flow": {},
                },
            },
        }
        ffo = _get_ffo(info)
        assert ffo is None
    
    def test_get_loss_adjustment_expense(self, insurance_info_with_financials):
        """Test loss expense extraction for insurer."""
        loss = _get_loss_adjustment_expense(insurance_info_with_financials)
        assert loss == 14_390_000_000


# =============================================================================
# Bank Adapter Tests
# =============================================================================

class TestBankAdapter:
    """Tests for BankAdapter with financial statements."""
    
    def test_bank_adapter_with_financials(self, bank_info_with_financials):
        """Test bank adapter uses NIM from financial statements."""
        adapter = BankAdapter()
        result = adapter.compute_quality_score(bank_info_with_financials)
        
        assert result.domain == Domain.BANK
        assert result.final_score > 0
        
        # Check that NIM sub-score exists
        nim_scores = [s for s in result.sub_scores if s.name == "net_interest_margin"]
        assert len(nim_scores) == 1
        assert nim_scores[0].available is True
        assert "NIM" in nim_scores[0].reason
        
        # Notes should mention using financial statements
        assert "financial statements" in result.notes
    
    def test_bank_adapter_without_financials(self, bank_info_without_financials):
        """Test bank adapter gracefully handles missing financials."""
        adapter = BankAdapter()
        result = adapter.compute_quality_score(bank_info_without_financials)
        
        assert result.domain == Domain.BANK
        assert result.final_score > 0
        
        # NIM sub-score should exist but be unavailable
        nim_scores = [s for s in result.sub_scores if s.name == "net_interest_margin"]
        assert len(nim_scores) == 1
        assert nim_scores[0].available is False
    
    def test_bank_adapter_good_metrics(self, bank_info_with_financials):
        """Test bank with good metrics gets high score."""
        adapter = BankAdapter()
        result = adapter.compute_quality_score(bank_info_with_financials)
        
        # Good ROE (16%), decent P/B (1.8), positive NIM
        assert result.final_score >= 60


# =============================================================================
# REIT Adapter Tests
# =============================================================================

class TestREITAdapter:
    """Tests for REITAdapter with financial statements."""
    
    def test_reit_adapter_with_financials(self, reit_info_with_financials):
        """Test REIT adapter calculates FFO from financial statements."""
        adapter = REITAdapter()
        result = adapter.compute_quality_score(reit_info_with_financials)
        
        assert result.domain == Domain.REIT
        assert result.final_score > 0
        
        # Check that P/FFO sub-score exists and is calculated
        pffo_scores = [s for s in result.sub_scores if s.name == "p_ffo"]
        assert len(pffo_scores) == 1
        assert pffo_scores[0].available is True
        assert "P/FFO" in pffo_scores[0].reason
        
        # Notes should mention using financial statements
        assert "financial statements" in result.notes
    
    def test_reit_adapter_ffo_calculation(self, reit_info_with_financials):
        """Test FFO and P/FFO calculation."""
        adapter = REITAdapter()
        result = adapter.compute_quality_score(reit_info_with_financials)
        
        pffo_score = next(s for s in result.sub_scores if s.name == "p_ffo")
        
        # Verify P/FFO calculation:
        # FFO = 315M + 632M = 947M quarterly
        # Annual FFO = 947M * 4 = 3.788B
        # FFO/share = 3.788B / 900M shares = $4.21
        # P/FFO = $57 / $4.21 = 13.5x
        # This should result in a good score (12-16x is fair)
        assert pffo_score.score >= 50
    
    def test_reit_adapter_without_financials(self):
        """Test REIT adapter handles missing financials."""
        info = {
            "dividend_yield": 0.055,
            "price_to_book": 1.2,
            "debt_to_equity": 80,
            "recommendation_mean": 2.5,
        }
        
        adapter = REITAdapter()
        result = adapter.compute_quality_score(info)
        
        assert result.domain == Domain.REIT
        assert result.final_score > 0
        
        # P/FFO should be unavailable
        pffo_scores = [s for s in result.sub_scores if s.name == "p_ffo"]
        assert len(pffo_scores) == 1
        assert pffo_scores[0].available is False


# =============================================================================
# Insurance Adapter Tests
# =============================================================================

class TestInsuranceAdapter:
    """Tests for InsuranceAdapter with financial statements."""
    
    def test_insurance_adapter_with_financials(self, insurance_info_with_financials):
        """Test insurance adapter uses loss ratio from financial statements."""
        adapter = InsuranceAdapter()
        result = adapter.compute_quality_score(insurance_info_with_financials)
        
        assert result.domain == Domain.INSURER
        assert result.final_score > 0
        
        # Check that loss_ratio sub-score exists
        loss_scores = [s for s in result.sub_scores if s.name == "loss_ratio"]
        assert len(loss_scores) == 1
        assert loss_scores[0].available is True
        assert "loss_ratio" in loss_scores[0].reason
        
        # Notes should mention using financial statements
        assert "financial statements" in result.notes
    
    def test_insurance_adapter_loss_ratio_calculation(self, insurance_info_with_financials):
        """Test loss ratio calculation."""
        adapter = InsuranceAdapter()
        result = adapter.compute_quality_score(insurance_info_with_financials)
        
        loss_score = next(s for s in result.sub_scores if s.name == "loss_ratio")
        
        # Loss ratio = 14.39B / 22.5B ≈ 64%
        # This is a healthy loss ratio (60-70%)
        assert loss_score.score >= 50
    
    def test_insurance_adapter_without_financials(self):
        """Test insurance adapter handles missing financials."""
        info = {
            "return_on_equity": 0.25,
            "return_on_assets": 0.05,
            "price_to_book": 4.5,
            "dividend_yield": 0.02,
            "recommendation_mean": 2.2,
        }
        
        adapter = InsuranceAdapter()
        result = adapter.compute_quality_score(info)
        
        assert result.domain == Domain.INSURER
        assert result.final_score > 0
        
        # Loss ratio should be unavailable with zero weight
        loss_scores = [s for s in result.sub_scores if s.name == "loss_ratio"]
        assert len(loss_scores) == 1
        assert loss_scores[0].available is False


# =============================================================================
# Integration Tests (require network, skip by default)
# =============================================================================

@pytest.mark.skip(reason="Integration test - requires network and yfinance")
class TestFinancialsIntegration:
    """Integration tests for financial statement fetching."""
    
    @pytest.mark.asyncio
    async def test_fetch_bank_financials(self):
        """Test fetching real financials for a bank."""
        from app.services.data_providers.yfinance_service import YFinanceService
        
        service = YFinanceService()
        financials = await service.get_financials("JPM")
        
        assert financials is not None
        assert "quarterly" in financials
        assert "annual" in financials
        
        qis = financials["quarterly"]["income_statement"]
        assert "Net Interest Income" in qis
        assert qis["Net Interest Income"] > 0
    
    @pytest.mark.asyncio
    async def test_fetch_reit_financials(self):
        """Test fetching real financials for a REIT."""
        from app.services.data_providers.yfinance_service import YFinanceService
        
        service = YFinanceService()
        financials = await service.get_financials("O")
        
        assert financials is not None
        
        qcf = financials["quarterly"]["cash_flow"]
        assert "Depreciation Amortization Depletion" in qcf or "Depreciation And Amortization" in qcf
    
    @pytest.mark.asyncio
    async def test_get_ticker_info_with_financials(self):
        """Test combined info + financials fetch."""
        from app.services.data_providers.yfinance_service import YFinanceService
        
        service = YFinanceService()
        info = await service.get_ticker_info_with_financials("JPM")
        
        assert info is not None
        assert "financials" in info
        assert info["financials"]["quarterly"]["income_statement"]
