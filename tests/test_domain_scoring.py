"""Tests for domain classification and domain-specific scoring."""

import pytest

from app.dipfinder.domain import (
    Domain,
    DomainClassification,
    classify_domain,
    get_domain_from_info,
    get_domain_metadata,
    DOMAIN_METADATA,
)
from app.dipfinder.domain_scoring import (
    DomainScoreResult,
    SubScore,
    OperatingCompanyAdapter,
    BankAdapter,
    InsuranceAdapter,
    REITAdapter,
    ETFAdapter,
    UtilityAdapter,
    BiotechAdapter,
    EnergyAdapter,
    RetailAdapter,
    SemiconductorAdapter,
    CapitalIntensiveAdapter,
    get_adapter,
    compute_domain_score,
)


class TestDomainClassification:
    """Tests for classify_domain function."""

    def test_etf_by_quote_type(self):
        """ETF should be detected by quoteType."""
        result = classify_domain(quote_type="ETF", name="SPDR S&P 500")
        assert result.domain == Domain.ETF
        assert result.confidence == 1.0
        assert "quoteType" in result.reason

    def test_mutual_fund_by_quote_type(self):
        """Mutual fund should be detected by quoteType."""
        result = classify_domain(quote_type="MUTUALFUND", name="Vanguard 500")
        assert result.domain == Domain.MUTUAL_FUND
        assert result.confidence == 1.0

    def test_index_by_quote_type(self):
        """Index should be detected by quoteType."""
        result = classify_domain(quote_type="INDEX", name="S&P 500")
        assert result.domain == Domain.INDEX
        assert result.confidence == 1.0

    def test_bank_by_industry(self):
        """Bank should be detected by industry pattern."""
        result = classify_domain(
            quote_type="EQUITY",
            sector="Financial Services",
            industry="Banks—Regional",
            name="First Republic Bank",
        )
        assert result.domain == Domain.BANK
        assert result.confidence == 0.95
        assert "industry" in result.reason

    def test_bank_by_industry_diversified(self):
        """Bank should be detected for diversified banks."""
        result = classify_domain(
            quote_type="EQUITY",
            sector="Financial Services",
            industry="Banks—Diversified",
            name="JPMorgan Chase",
        )
        assert result.domain == Domain.BANK

    def test_insurer_by_industry(self):
        """Insurer should be detected by industry."""
        result = classify_domain(
            quote_type="EQUITY",
            sector="Financial Services",
            industry="Insurance—Property & Casualty",
            name="Progressive Corp",
        )
        assert result.domain == Domain.INSURER
        assert result.confidence == 0.95

    def test_reit_by_industry(self):
        """REIT should be detected by industry."""
        result = classify_domain(
            quote_type="EQUITY",
            sector="Real Estate",
            industry="REIT—Retail",
            name="Simon Property Group",
        )
        assert result.domain == Domain.REIT
        assert result.confidence == 0.95

    def test_reit_by_name_pattern(self):
        """REIT should be detected by name when industry missing."""
        result = classify_domain(
            quote_type="EQUITY",
            name="Digital Realty Trust",
        )
        assert result.domain == Domain.REIT
        assert result.confidence == 0.70  # Lower confidence for name-only
        assert result.fallback_domain == Domain.OPERATING_COMPANY

    def test_utility_by_industry(self):
        """Utility should be detected by industry."""
        result = classify_domain(
            quote_type="EQUITY",
            sector="Utilities",
            industry="Utilities—Regulated Electric",
            name="Duke Energy",
        )
        assert result.domain == Domain.UTILITY
        assert result.confidence == 0.95

    def test_utility_by_sector_fallback(self):
        """Utility should be detected by sector when industry missing."""
        result = classify_domain(
            quote_type="EQUITY",
            sector="Utilities",
            name="Some Utility Co",
        )
        assert result.domain == Domain.UTILITY
        assert result.confidence == 0.60  # Lower confidence for sector-only

    def test_biotech_by_industry(self):
        """Biotech should be detected by industry."""
        result = classify_domain(
            quote_type="EQUITY",
            sector="Healthcare",
            industry="Biotechnology",
            name="Moderna",
        )
        assert result.domain == Domain.BIOTECH
        assert result.confidence == 0.95

    def test_pharma_by_industry(self):
        """Pharma should be detected by industry."""
        result = classify_domain(
            quote_type="EQUITY",
            sector="Healthcare",
            industry="Drug Manufacturers—General",
            name="Pfizer",
        )
        assert result.domain == Domain.PHARMA
        assert result.confidence == 0.95

    def test_saas_by_industry(self):
        """SaaS should be detected by industry."""
        result = classify_domain(
            quote_type="EQUITY",
            sector="Technology",
            industry="Software—Application",
            name="Salesforce",
        )
        assert result.domain == Domain.SAAS
        assert result.confidence == 0.95

    def test_semiconductor_by_industry(self):
        """Semiconductor should be detected by industry."""
        result = classify_domain(
            quote_type="EQUITY",
            sector="Technology",
            industry="Semiconductors",
            name="NVIDIA",
        )
        assert result.domain == Domain.SEMICONDUCTOR
        assert result.confidence == 0.95

    def test_energy_by_industry(self):
        """Energy should be detected by industry."""
        result = classify_domain(
            quote_type="EQUITY",
            sector="Energy",
            industry="Oil & Gas E&P",
            name="ConocoPhillips",
        )
        assert result.domain == Domain.ENERGY
        assert result.confidence == 0.95

    def test_mining_by_industry(self):
        """Mining should be detected by industry."""
        result = classify_domain(
            quote_type="EQUITY",
            sector="Basic Materials",
            industry="Gold",
            name="Newmont Corp",
        )
        assert result.domain == Domain.MINING
        assert result.confidence == 0.95

    def test_airline_by_industry(self):
        """Airline should be detected by industry."""
        result = classify_domain(
            quote_type="EQUITY",
            sector="Industrials",
            industry="Airlines",
            name="Delta Air Lines",
        )
        assert result.domain == Domain.AIRLINE
        assert result.confidence == 0.95

    def test_retail_by_industry(self):
        """Retail should be detected by industry."""
        result = classify_domain(
            quote_type="EQUITY",
            sector="Consumer Cyclical",
            industry="Internet Retail",
            name="Amazon",
        )
        assert result.domain == Domain.RETAIL
        assert result.confidence == 0.95

    def test_foreign_adr_by_symbol_suffix(self):
        """Foreign ADR should be detected by symbol suffix."""
        result = classify_domain(
            quote_type="EQUITY",
            name="SAP SE",
            symbol="SAP.DE",
        )
        assert result.domain == Domain.FOREIGN_ADR
        assert result.confidence == 0.80

    def test_foreign_adr_by_name(self):
        """Foreign ADR should be detected by ADR in name."""
        result = classify_domain(
            quote_type="EQUITY",
            name="Toyota Motor Corp ADR",
            symbol="TM",
        )
        assert result.domain == Domain.FOREIGN_ADR
        assert result.confidence == 0.85

    def test_operating_company_default(self):
        """Unknown securities should default to operating company."""
        result = classify_domain(
            quote_type="EQUITY",
            sector="Technology",
            industry="Consumer Electronics",
            name="Apple Inc",
        )
        assert result.domain == Domain.OPERATING_COMPANY
        assert result.confidence == 0.50

    def test_spac_by_name(self):
        """SPAC should be detected by name pattern."""
        result = classify_domain(
            quote_type="EQUITY",
            name="Pershing Square SPAC Holdings",
        )
        assert result.domain == Domain.SPAC
        assert result.confidence == 0.70

    def test_etf_by_name_pattern(self):
        """ETF should be detected by name pattern when quoteType missing."""
        result = classify_domain(
            name="iShares Core S&P 500 ETF",
        )
        assert result.domain == Domain.ETF
        assert result.confidence == 0.70

    def test_bank_by_name_bancorp(self):
        """Bank should be detected by Bancorp in name."""
        result = classify_domain(
            quote_type="EQUITY",
            name="US Bancorp",
        )
        assert result.domain == Domain.BANK
        assert result.confidence == 0.70


class TestGetDomainFromInfo:
    """Tests for get_domain_from_info convenience function."""

    def test_from_yfinance_dict(self):
        """Should correctly parse yfinance-style info dict."""
        info = {
            "symbol": "JPM",
            "quote_type": "EQUITY",
            "sector": "Financial Services",
            "industry": "Banks—Diversified",
            "name": "JPMorgan Chase & Co.",
        }
        result = get_domain_from_info(info)
        assert result.domain == Domain.BANK
        assert result.confidence == 0.95

    def test_from_empty_dict(self):
        """Should handle empty dict gracefully."""
        result = get_domain_from_info({})
        assert result.domain == Domain.OPERATING_COMPANY
        assert result.confidence == 0.50


class TestDomainMetadata:
    """Tests for domain metadata."""

    def test_bank_metadata(self):
        """Bank metadata should have correct key metrics."""
        meta = get_domain_metadata(Domain.BANK)
        assert meta.domain == Domain.BANK
        assert "return_on_equity" in meta.key_metrics
        assert "debt_to_equity" in meta.skip_metrics

    def test_reit_metadata(self):
        """REIT metadata should have correct key metrics."""
        meta = get_domain_metadata(Domain.REIT)
        assert meta.domain == Domain.REIT
        assert "dividend_yield" in meta.key_metrics
        assert "free_cash_flow" in meta.skip_metrics

    def test_etf_metadata(self):
        """ETF metadata should skip most metrics."""
        meta = get_domain_metadata(Domain.ETF)
        assert meta.domain == Domain.ETF
        assert "pe_ratio" in meta.skip_metrics
        assert "profit_margin" in meta.skip_metrics

    def test_unknown_domain_fallback(self):
        """Unknown domain should fall back to OPERATING_COMPANY."""
        # SPAC is in Domain enum but not in DOMAIN_METADATA
        meta = get_domain_metadata(Domain.SPAC)
        assert meta.domain == Domain.OPERATING_COMPANY


class TestDomainScoringAdapters:
    """Tests for domain-specific scoring adapters."""

    @pytest.fixture
    def bank_info(self):
        """Sample bank info dict."""
        return {
            "symbol": "JPM",
            "name": "JPMorgan Chase",
            "quote_type": "EQUITY",
            "sector": "Financial Services",
            "industry": "Banks—Diversified",
            "return_on_equity": 0.15,  # 15%
            "return_on_assets": 0.012,  # 1.2%
            "price_to_book": 1.2,
            "book_value": 85.0,
            "dividend_yield": 0.03,  # 3%
            "recommendation_mean": 2.0,  # Buy
            "current_price": 150.0,
        }

    @pytest.fixture
    def reit_info(self):
        """Sample REIT info dict."""
        return {
            "symbol": "O",
            "name": "Realty Income Corp",
            "quote_type": "EQUITY",
            "sector": "Real Estate",
            "industry": "REIT—Retail",
            "dividend_yield": 0.055,  # 5.5%
            "price_to_book": 1.1,
            "debt_to_equity": 80,  # 80%
            "recommendation_mean": 2.5,
            "current_price": 55.0,
        }

    @pytest.fixture
    def etf_info(self):
        """Sample ETF info dict."""
        return {
            "symbol": "SPY",
            "name": "SPDR S&P 500 ETF Trust",
            "quote_type": "ETF",
            "market_cap": 500_000_000_000,
            "current_price": 450.0,
        }

    @pytest.fixture
    def operating_company_info(self):
        """Sample operating company info dict."""
        return {
            "symbol": "AAPL",
            "name": "Apple Inc",
            "quote_type": "EQUITY",
            "sector": "Technology",
            "industry": "Consumer Electronics",
            "profit_margin": 0.25,
            "operating_margin": 0.30,
            "debt_to_equity": 150,
            "current_ratio": 0.9,
            "free_cash_flow": 100_000_000_000,
            "market_cap": 3_000_000_000_000,
            "pe_ratio": 28.0,
            "peg_ratio": 1.5,
            "revenue_growth": 0.08,
            "earnings_growth": 0.10,
            "recommendation_mean": 1.8,
        }

    @pytest.fixture
    def biotech_info(self):
        """Sample pre-revenue biotech info dict."""
        return {
            "symbol": "MRNA",
            "name": "Moderna Inc",
            "quote_type": "EQUITY",
            "sector": "Healthcare",
            "industry": "Biotechnology",
            "total_cash": 5_000_000_000,
            "market_cap": 15_000_000_000,
            "recommendation_mean": 2.2,
            "num_analyst_opinions": 25,
            "beta": 1.8,
            "short_percent_of_float": 0.08,
        }

    def test_bank_adapter_scoring(self, bank_info):
        """Bank adapter should score based on ROE, ROA, P/B, dividend."""
        adapter = BankAdapter()
        result = adapter.compute_quality_score(bank_info)
        
        assert result.domain == Domain.BANK
        assert 60 <= result.final_score <= 100  # Good bank metrics
        
        # Check sub-scores exist
        sub_score_names = [s.name for s in result.sub_scores]
        assert "returns" in sub_score_names
        assert "book_value" in sub_score_names
        assert "dividend" in sub_score_names
        assert "analyst" in sub_score_names
        
        # D/E and FCF should NOT be scored
        assert "balance_sheet" not in sub_score_names
        assert "cash_generation" not in sub_score_names

    def test_bank_adapter_high_roe(self, bank_info):
        """High ROE should result in high returns score."""
        bank_info["return_on_equity"] = 0.20  # 20% ROE
        adapter = BankAdapter()
        result = adapter.compute_quality_score(bank_info)
        
        returns_score = next(s for s in result.sub_scores if s.name == "returns")
        assert returns_score.score >= 80

    def test_bank_adapter_low_pb(self, bank_info):
        """Low P/B should result in high book value score."""
        bank_info["price_to_book"] = 0.7  # Trading below book
        adapter = BankAdapter()
        result = adapter.compute_quality_score(bank_info)
        
        book_score = next(s for s in result.sub_scores if s.name == "book_value")
        assert book_score.score >= 90

    def test_reit_adapter_scoring(self, reit_info):
        """REIT adapter should score based on dividend, P/B, leverage."""
        adapter = REITAdapter()
        result = adapter.compute_quality_score(reit_info)
        
        assert result.domain == Domain.REIT
        assert 60 <= result.final_score <= 100
        
        sub_score_names = [s.name for s in result.sub_scores]
        assert "dividend" in sub_score_names
        assert "price_to_book" in sub_score_names
        assert "leverage" in sub_score_names
        
        # FCF should NOT be scored
        assert "cash_generation" not in sub_score_names

    def test_reit_adapter_high_yield_warning(self, reit_info):
        """Very high REIT yield should be penalized (distress signal)."""
        reit_info["dividend_yield"] = 0.15  # 15% - suspicious
        adapter = REITAdapter()
        result = adapter.compute_quality_score(reit_info)
        
        dividend_score = next(s for s in result.sub_scores if s.name == "dividend")
        assert dividend_score.score < 50  # Should be penalized

    def test_etf_adapter_neutral_score(self, etf_info):
        """ETF adapter should return neutral score (quality N/A)."""
        adapter = ETFAdapter()
        result = adapter.compute_quality_score(etf_info)
        
        assert result.domain == Domain.ETF
        assert result.final_score == 70.0  # Neutral-positive
        assert result.data_completeness == 1.0  # Nothing expected
        assert "not applicable" in result.sub_scores[0].reason.lower()

    def test_operating_company_adapter_scoring(self, operating_company_info):
        """Operating company adapter should use all standard metrics."""
        adapter = OperatingCompanyAdapter()
        result = adapter.compute_quality_score(operating_company_info)
        
        assert result.domain == Domain.OPERATING_COMPANY
        
        sub_score_names = [s.name for s in result.sub_scores]
        assert "profitability" in sub_score_names
        assert "balance_sheet" in sub_score_names
        assert "cash_generation" in sub_score_names
        assert "valuation" in sub_score_names
        assert "growth" in sub_score_names
        assert "analyst" in sub_score_names

    def test_biotech_adapter_cash_focused(self, biotech_info):
        """Biotech adapter should focus on cash position."""
        adapter = BiotechAdapter()
        result = adapter.compute_quality_score(biotech_info)
        
        assert result.domain == Domain.BIOTECH
        
        sub_score_names = [s.name for s in result.sub_scores]
        assert "cash_position" in sub_score_names
        assert "analyst" in sub_score_names
        assert "risk" in sub_score_names
        
        # Cash position should have high weight
        cash_sub = next(s for s in result.sub_scores if s.name == "cash_position")
        assert cash_sub.weight == 0.40

    def test_utility_adapter_scoring(self):
        """Utility adapter should score based on dividend, margin, debt."""
        utility_info = {
            "symbol": "DUK",
            "name": "Duke Energy",
            "dividend_yield": 0.04,
            "operating_margin": 0.20,
            "debt_to_equity": 120,
            "recommendation_mean": 2.5,
        }
        adapter = UtilityAdapter()
        result = adapter.compute_quality_score(utility_info)
        
        assert result.domain == Domain.UTILITY
        
        sub_score_names = [s.name for s in result.sub_scores]
        assert "dividend" in sub_score_names
        assert "operating_margin" in sub_score_names
        assert "debt" in sub_score_names


class TestGetAdapter:
    """Tests for adapter registry."""

    def test_get_bank_adapter(self):
        """Should return BankAdapter for banks."""
        adapter = get_adapter(Domain.BANK)
        assert isinstance(adapter, BankAdapter)

    def test_get_insurer_adapter(self):
        """Insurers should use dedicated InsuranceAdapter."""
        adapter = get_adapter(Domain.INSURER)
        assert isinstance(adapter, InsuranceAdapter)

    def test_get_reit_adapter(self):
        """Should return REITAdapter for REITs."""
        adapter = get_adapter(Domain.REIT)
        assert isinstance(adapter, REITAdapter)

    def test_get_etf_adapter(self):
        """Should return ETFAdapter for ETFs."""
        adapter = get_adapter(Domain.ETF)
        assert isinstance(adapter, ETFAdapter)

    def test_get_unknown_domain_fallback(self):
        """Unknown domains should fall back to OperatingCompanyAdapter."""
        adapter = get_adapter(Domain.CONGLOMERATE)
        assert isinstance(adapter, OperatingCompanyAdapter)


class TestComputeDomainScore:
    """Tests for compute_domain_score integration function."""

    def test_high_confidence_classification(self):
        """High confidence classification should not use fallback."""
        classification = DomainClassification(
            domain=Domain.BANK,
            confidence=0.95,
            reason="industry=Banks",
        )
        info = {
            "return_on_equity": 0.12,
            "return_on_assets": 0.01,
            "price_to_book": 1.0,
            "dividend_yield": 0.03,
            "recommendation_mean": 2.0,
        }
        result = compute_domain_score(classification, info)
        
        assert result.domain == Domain.BANK
        assert result.domain_confidence == 0.95
        assert result.fallback_used is False

    def test_low_confidence_classification_blends(self):
        """Low confidence classification should blend with fallback."""
        classification = DomainClassification(
            domain=Domain.BANK,
            confidence=0.60,
            reason="name pattern",
            fallback_domain=Domain.OPERATING_COMPANY,
        )
        info = {
            "return_on_equity": 0.12,
            "return_on_assets": 0.01,
            "price_to_book": 1.0,
            "dividend_yield": 0.03,
            "recommendation_mean": 2.0,
            "profit_margin": 0.15,
            "operating_margin": 0.20,
        }
        result = compute_domain_score(classification, info)
        
        assert result.domain == Domain.BANK
        assert result.domain_confidence == 0.60
        assert result.fallback_used is True
        assert "fallback" in result.notes.lower()

    def test_result_to_dict(self):
        """DomainScoreResult.to_dict should serialize correctly."""
        result = DomainScoreResult(
            domain=Domain.BANK,
            domain_confidence=0.95,
            final_score=75.5,
            sub_scores=[
                SubScore(name="returns", score=80.0, weight=0.35, reason="ROE=12%"),
                SubScore(name="book_value", score=70.0, weight=0.25, reason="P/B=1.0"),
            ],
            data_completeness=0.80,
            fallback_used=False,
            notes="Bank scoring",
        )
        
        d = result.to_dict()
        assert d["domain"] == "bank"
        assert d["domain_confidence"] == 0.95
        assert d["final_score"] == 75.5
        assert len(d["sub_scores"]) == 2
        assert d["sub_scores"][0]["name"] == "returns"
        assert d["data_completeness"] == 0.80


class TestMissingDataHandling:
    """Tests for handling missing data gracefully."""

    def test_bank_missing_roe(self):
        """Bank with missing ROE should still score."""
        info = {
            "price_to_book": 1.0,
            "dividend_yield": 0.03,
            "recommendation_mean": 2.0,
        }
        adapter = BankAdapter()
        result = adapter.compute_quality_score(info)
        
        # Should have computed a score (with neutral ROE contribution)
        assert 50 <= result.final_score <= 85  # Score still computed
        
        # Returns sub-score should be marked unavailable
        returns_sub = next(s for s in result.sub_scores if s.name == "returns")
        assert returns_sub.available is False
        assert returns_sub.score == 50.0  # Neutral

    def test_operating_company_minimal_data(self):
        """Operating company with minimal data should still score."""
        info = {
            "profit_margin": 0.15,
        }
        adapter = OperatingCompanyAdapter()
        result = adapter.compute_quality_score(info)
        
        assert 50 <= result.final_score <= 70  # Mostly neutral with one positive
        assert result.data_completeness < 0.5  # Most data missing

    def test_reit_no_dividend(self):
        """REIT without dividend yield should get low score."""
        info = {
            "price_to_book": 0.9,
            "debt_to_equity": 100,
        }
        adapter = REITAdapter()
        result = adapter.compute_quality_score(info)
        
        # Should still compute but with lower confidence
        dividend_sub = next(s for s in result.sub_scores if s.name == "dividend")
        assert dividend_sub.available is False


class TestNewAdapters:
    """Tests for newly added domain adapters."""

    def test_energy_adapter_scoring(self):
        """Energy adapter should score based on margins, FCF, leverage, dividend."""
        energy_info = {
            "symbol": "XOM",
            "name": "Exxon Mobil",
            "operating_margin": 0.12,
            "profit_margin": 0.08,
            "free_cash_flow": 30_000_000_000,
            "market_cap": 400_000_000_000,
            "debt_to_equity": 80,
            "dividend_yield": 0.035,
            "recommendation_mean": 2.2,
        }
        adapter = EnergyAdapter()
        result = adapter.compute_quality_score(energy_info)
        
        assert result.domain == Domain.ENERGY
        
        sub_score_names = [s.name for s in result.sub_scores]
        assert "profitability" in sub_score_names
        assert "cash_flow" in sub_score_names
        assert "leverage" in sub_score_names
        assert "dividend" in sub_score_names
        assert "analyst" in sub_score_names

    def test_energy_adapter_high_leverage_penalty(self):
        """Energy with high leverage should score lower on leverage."""
        info = {
            "operating_margin": 0.15,
            "debt_to_equity": 250,
            "recommendation_mean": 2.5,
        }
        adapter = EnergyAdapter()
        result = adapter.compute_quality_score(info)
        
        leverage_sub = next(s for s in result.sub_scores if s.name == "leverage")
        assert leverage_sub.score < 40  # High leverage = low score

    def test_retail_adapter_scoring(self):
        """Retail adapter should score based on margins, growth, valuation."""
        retail_info = {
            "symbol": "TGT",
            "name": "Target Corporation",
            "profit_margin": 0.04,
            "operating_margin": 0.06,
            "revenue_growth": 0.05,
            "earnings_growth": 0.08,
            "pe_ratio": 15,
            "peg_ratio": 1.2,
            "debt_to_equity": 80,
            "current_ratio": 1.2,
            "recommendation_mean": 2.3,
        }
        adapter = RetailAdapter()
        result = adapter.compute_quality_score(retail_info)
        
        assert result.domain == Domain.RETAIL
        
        sub_score_names = [s.name for s in result.sub_scores]
        assert "margins" in sub_score_names
        assert "growth" in sub_score_names
        assert "valuation" in sub_score_names
        assert "balance_sheet" in sub_score_names
        assert "analyst" in sub_score_names

    def test_retail_adapter_thin_margins_ok(self):
        """Retail with thin margins should still score reasonably."""
        info = {
            "profit_margin": 0.025,  # 2.5% is thin but acceptable
            "operating_margin": 0.05,
            "recommendation_mean": 2.5,
        }
        adapter = RetailAdapter()
        result = adapter.compute_quality_score(info)
        
        # Thin margins shouldn't tank the score
        margin_sub = next(s for s in result.sub_scores if s.name == "margins")
        assert margin_sub.score > 25  # Not a disaster

    def test_semiconductor_adapter_scoring(self):
        """Semiconductor adapter should emphasize margins and growth."""
        semi_info = {
            "symbol": "NVDA",
            "name": "NVIDIA Corporation",
            "operating_margin": 0.45,
            "profit_margin": 0.35,
            "revenue_growth": 0.50,
            "earnings_growth": 0.80,
            "debt_to_equity": 30,
            "total_cash": 20_000_000_000,
            "market_cap": 1_000_000_000_000,
            "pe_ratio": 60,
            "recommendation_mean": 1.5,
        }
        adapter = SemiconductorAdapter()
        result = adapter.compute_quality_score(semi_info)
        
        assert result.domain == Domain.SEMICONDUCTOR
        
        sub_score_names = [s.name for s in result.sub_scores]
        assert "margins" in sub_score_names
        assert "growth" in sub_score_names
        assert "balance_sheet" in sub_score_names

    def test_semiconductor_adapter_high_margins_rewarded(self):
        """Semiconductor with high margins should score very well on margins."""
        info = {
            "operating_margin": 0.40,
            "profit_margin": 0.30,
        }
        adapter = SemiconductorAdapter()
        result = adapter.compute_quality_score(info)
        
        margin_sub = next(s for s in result.sub_scores if s.name == "margins")
        assert margin_sub.score > 80  # High margins = high score

    def test_capital_intensive_adapter_scoring(self):
        """Capital intensive adapter (airlines) should handle high debt."""
        airline_info = {
            "symbol": "DAL",
            "name": "Delta Air Lines",
            "operating_margin": 0.08,
            "profit_margin": 0.05,
            "debt_to_equity": 200,
            "free_cash_flow": 3_000_000_000,
            "market_cap": 30_000_000_000,
            "recommendation_mean": 2.0,
        }
        adapter = CapitalIntensiveAdapter()
        result = adapter.compute_quality_score(airline_info)
        
        assert result.domain == Domain.AIRLINE
        
        sub_score_names = [s.name for s in result.sub_scores]
        assert "margins" in sub_score_names
        assert "leverage" in sub_score_names
        assert "cash_flow" in sub_score_names
        assert "analyst" in sub_score_names

    def test_capital_intensive_adapter_high_debt_tolerance(self):
        """Airlines should tolerate higher D/E than normal companies."""
        info = {
            "operating_margin": 0.10,
            "debt_to_equity": 180,  # High for normal company, acceptable for airline
            "recommendation_mean": 2.5,
        }
        adapter = CapitalIntensiveAdapter()
        result = adapter.compute_quality_score(info)
        
        leverage_sub = next(s for s in result.sub_scores if s.name == "leverage")
        # D/E 180% should not be a disaster for airlines
        assert leverage_sub.score > 40


class TestAdapterRegistryNewDomains:
    """Tests for new domains in adapter registry."""

    def test_get_energy_adapter(self):
        """Should return EnergyAdapter for energy companies."""
        adapter = get_adapter(Domain.ENERGY)
        assert isinstance(adapter, EnergyAdapter)

    def test_get_mining_uses_energy_adapter(self):
        """Mining should use energy adapter (similar commodity dynamics)."""
        adapter = get_adapter(Domain.MINING)
        assert isinstance(adapter, EnergyAdapter)

    def test_get_retail_adapter(self):
        """Should return RetailAdapter for retail."""
        adapter = get_adapter(Domain.RETAIL)
        assert isinstance(adapter, RetailAdapter)

    def test_get_semiconductor_adapter(self):
        """Should return SemiconductorAdapter for semis."""
        adapter = get_adapter(Domain.SEMICONDUCTOR)
        assert isinstance(adapter, SemiconductorAdapter)

    def test_get_airline_adapter(self):
        """Should return CapitalIntensiveAdapter for airlines."""
        adapter = get_adapter(Domain.AIRLINE)
        assert isinstance(adapter, CapitalIntensiveAdapter)

    def test_get_shipping_uses_capital_intensive_adapter(self):
        """Shipping should use capital intensive adapter."""
        adapter = get_adapter(Domain.SHIPPING)
        assert isinstance(adapter, CapitalIntensiveAdapter)


class TestDomainMetadataComplete:
    """Tests for domain metadata completeness."""

    def test_all_registered_domains_have_metadata(self):
        """All domains in adapter registry should have metadata."""
        from app.dipfinder.domain_scoring import _ADAPTERS
        
        for domain in _ADAPTERS.keys():
            metadata = get_domain_metadata(domain)
            assert metadata is not None
            assert metadata.display_name
            assert metadata.description
            assert len(metadata.key_metrics) > 0

    def test_metadata_key_metrics_not_empty(self):
        """All domain metadata should have key_metrics defined."""
        for domain, metadata in DOMAIN_METADATA.items():
            assert len(metadata.key_metrics) > 0, f"{domain} has no key_metrics"

    def test_new_domains_have_metadata(self):
        """New domains should have metadata entries."""
        new_domains = [
            Domain.ENERGY,
            Domain.MINING,
            Domain.RETAIL,
            Domain.SEMICONDUCTOR,
            Domain.AIRLINE,
            Domain.SHIPPING,
            Domain.SAAS,
            Domain.TELECOM,
            Domain.PHARMA,
        ]
        for domain in new_domains:
            metadata = get_domain_metadata(domain)
            assert metadata is not None, f"{domain} has no metadata"
            assert metadata.display_name, f"{domain} has no display_name"
