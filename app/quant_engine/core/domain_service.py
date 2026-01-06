"""
DomainService - Single source of truth for domain/sector classification.

This consolidates the overlapping classification systems:
- app/dipfinder/domain.py (Domain enum - granular: BANK, REIT, ETF, etc.)
- app/quant_engine/domain_analysis.py (Sector enum - broad: TECHNOLOGY, FINANCIALS, etc.)

The Domain enum is more granular and is the PRIMARY classification.
Sectors are derived from Domains for backward compatibility.

All scoring and analysis should use this service for domain classification.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any

# Import the canonical Domain enum from dipfinder
from app.quant_engine.dipfinder.domain import (
    Domain,
    DomainClassification,
    classify_domain,
    get_domain_metadata,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Sector Enum (Broad Categories) - for backward compatibility
# =============================================================================

class Sector(str, Enum):
    """Broad sector classification (derived from Domain)."""
    
    TECHNOLOGY = "technology"
    FINANCIALS = "financials"
    HEALTHCARE = "healthcare"
    CONSUMER_DISCRETIONARY = "consumer_discretionary"
    CONSUMER_STAPLES = "consumer_staples"
    ENERGY = "energy"
    INDUSTRIALS = "industrials"
    MATERIALS = "materials"
    REAL_ESTATE = "real_estate"
    UTILITIES = "utilities"
    COMMUNICATION = "communication"
    FUNDS = "funds"  # ETF, mutual funds
    UNKNOWN = "unknown"


# Map Domain -> Sector for backward compatibility
DOMAIN_TO_SECTOR: dict[Domain, Sector] = {
    # Financials
    Domain.BANK: Sector.FINANCIALS,
    Domain.INSURER: Sector.FINANCIALS,
    Domain.ASSET_MANAGER: Sector.FINANCIALS,
    
    # Real Estate
    Domain.REIT: Sector.REAL_ESTATE,
    
    # Funds
    Domain.ETF: Sector.FUNDS,
    Domain.MUTUAL_FUND: Sector.FUNDS,
    Domain.INDEX: Sector.FUNDS,
    
    # Utilities
    Domain.UTILITY: Sector.UTILITIES,
    Domain.TELECOM: Sector.COMMUNICATION,
    
    # Energy/Materials
    Domain.ENERGY: Sector.ENERGY,
    Domain.MINING: Sector.MATERIALS,
    
    # Healthcare
    Domain.BIOTECH: Sector.HEALTHCARE,
    Domain.PHARMA: Sector.HEALTHCARE,
    
    # Technology
    Domain.SAAS: Sector.TECHNOLOGY,
    Domain.SEMICONDUCTOR: Sector.TECHNOLOGY,
    
    # Industrials
    Domain.AIRLINE: Sector.INDUSTRIALS,
    Domain.SHIPPING: Sector.INDUSTRIALS,
    Domain.CONGLOMERATE: Sector.INDUSTRIALS,
    
    # Consumer
    Domain.RETAIL: Sector.CONSUMER_DISCRETIONARY,
    
    # Other
    Domain.FOREIGN_ADR: Sector.UNKNOWN,
    Domain.SPAC: Sector.UNKNOWN,
    Domain.OPERATING_COMPANY: Sector.UNKNOWN,
}


def domain_to_sector(domain: Domain) -> Sector:
    """Convert Domain to Sector for backward compatibility."""
    return DOMAIN_TO_SECTOR.get(domain, Sector.UNKNOWN)


# =============================================================================
# DomainService - Unified Classification
# =============================================================================

@dataclass
class DomainAnalysisResult:
    """Complete domain analysis for a security."""
    
    # Primary classification
    domain: Domain
    domain_confidence: float  # 0-1
    domain_description: str
    
    # Derived sector (for backward compatibility)
    sector: Sector
    
    # Analysis flags
    skip_de_ratio: bool  # Skip Debt/Equity for banks/REITs
    skip_fcf: bool  # Skip FCF for REITs
    skip_pe: bool  # Skip P/E for growth/biotech
    use_book_value: bool  # Use book value for banks
    use_ffo: bool  # Use FFO for REITs
    use_nav: bool  # Use NAV for funds
    
    # Weight adjustments
    quality_weight: float  # How much to weight quality (0-1)
    technical_weight: float  # How much to weight technicals
    dividend_weight: float  # How much to weight dividends
    growth_weight: float  # How much to weight growth
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dict for API responses."""
        return {
            "domain": self.domain.value,
            "domain_confidence": round(self.domain_confidence, 2),
            "domain_description": self.domain_description,
            "sector": self.sector.value,
            "skip_de_ratio": self.skip_de_ratio,
            "skip_fcf": self.skip_fcf,
            "skip_pe": self.skip_pe,
            "quality_weight": round(self.quality_weight, 2),
        }


class DomainService:
    """
    Unified domain classification service.
    
    Usage:
        service = DomainService.get_instance()
        result = service.classify("AAPL", quote_type="EQUITY", sector="Technology")
        
        if result.skip_de_ratio:
            # Don't penalize for high debt (e.g., banks)
            pass
    """
    
    _instance: "DomainService | None" = None
    
    @classmethod
    def get_instance(cls) -> "DomainService":
        """Get singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def classify(
        self,
        symbol: str,
        quote_type: str | None = None,
        sector: str | None = None,
        industry: str | None = None,
        name: str | None = None,
    ) -> DomainAnalysisResult:
        """
        Classify a security and return analysis parameters.
        
        Args:
            symbol: Stock symbol
            quote_type: ETF, EQUITY, MUTUALFUND, INDEX
            sector: Sector string from data provider
            industry: Industry string from data provider
            name: Company name (for pattern matching)
            
        Returns:
            DomainAnalysisResult with classification and analysis flags
        """
        # Use dipfinder's classify_domain for the heavy lifting
        classification = classify_domain(
            quote_type=quote_type,
            sector=sector,
            industry=industry,
            name=name,
        )
        
        domain = classification.domain
        confidence = classification.confidence
        
        # Get description from metadata
        metadata = get_domain_metadata(domain)
        description = metadata.description
        
        # Derive sector for backward compatibility
        derived_sector = domain_to_sector(domain)
        
        # Determine analysis flags based on domain
        analysis = self._get_analysis_flags(domain)
        
        return DomainAnalysisResult(
            domain=domain,
            domain_confidence=confidence,
            domain_description=description,
            sector=derived_sector,
            **analysis,
        )
    
    def _get_analysis_flags(self, domain: Domain) -> dict[str, Any]:
        """Get analysis flags and weights for a domain."""
        
        # Default flags
        flags = {
            "skip_de_ratio": False,
            "skip_fcf": False,
            "skip_pe": False,
            "use_book_value": False,
            "use_ffo": False,
            "use_nav": False,
            "quality_weight": 0.25,
            "technical_weight": 0.35,
            "dividend_weight": 0.15,
            "growth_weight": 0.25,
        }
        
        # Domain-specific overrides
        if domain == Domain.BANK:
            flags.update({
                "skip_de_ratio": True,  # Banks have high leverage by design
                "use_book_value": True,
                "quality_weight": 0.35,
                "dividend_weight": 0.25,
                "growth_weight": 0.15,
            })
        
        elif domain == Domain.REIT:
            flags.update({
                "skip_de_ratio": True,  # REITs use leverage
                "skip_fcf": True,  # FCF not meaningful for REITs
                "use_ffo": True,  # FFO is the key metric
                "quality_weight": 0.20,
                "dividend_weight": 0.40,  # Dividends are key for REITs
                "growth_weight": 0.10,
            })
        
        elif domain in (Domain.ETF, Domain.MUTUAL_FUND, Domain.INDEX):
            flags.update({
                "skip_de_ratio": True,
                "skip_fcf": True,
                "skip_pe": True,
                "use_nav": True,
                "quality_weight": 0.05,  # Minimal quality scoring
                "technical_weight": 0.60,  # Focus on technicals
                "dividend_weight": 0.20,
                "growth_weight": 0.15,
            })
        
        elif domain == Domain.BIOTECH:
            flags.update({
                "skip_pe": True,  # Many biotechs have no earnings
                "quality_weight": 0.15,
                "growth_weight": 0.40,  # Growth/pipeline is key
                "technical_weight": 0.40,
            })
        
        elif domain == Domain.UTILITY:
            flags.update({
                "quality_weight": 0.30,
                "dividend_weight": 0.35,  # Utilities are dividend plays
                "growth_weight": 0.10,
                "technical_weight": 0.25,
            })
        
        elif domain == Domain.ENERGY:
            flags.update({
                "quality_weight": 0.20,
                "dividend_weight": 0.25,
                "growth_weight": 0.20,
                "technical_weight": 0.35,  # More volatile, technicals matter
            })
        
        elif domain == Domain.SEMICONDUCTOR:
            flags.update({
                "quality_weight": 0.25,
                "growth_weight": 0.35,  # Cyclical growth
                "technical_weight": 0.30,
            })
        
        elif domain == Domain.SAAS:
            flags.update({
                "skip_pe": True,  # Many SaaS have no earnings
                "quality_weight": 0.20,
                "growth_weight": 0.45,  # Growth is paramount
                "technical_weight": 0.30,
            })
        
        return flags
    
    def get_scoring_weights(
        self,
        domain: Domain,
    ) -> dict[str, float]:
        """
        Get scoring weights for a domain.
        
        Returns weights that sum to 1.0 for:
        - quality_score
        - technical_score
        - dividend_score
        - growth_score
        """
        flags = self._get_analysis_flags(domain)
        
        return {
            "quality": flags["quality_weight"],
            "technical": flags["technical_weight"],
            "dividend": flags["dividend_weight"],
            "growth": flags["growth_weight"],
        }


# =============================================================================
# Module-level helper functions for backward compatibility
# =============================================================================

def get_domain_service() -> DomainService:
    """Get the singleton DomainService instance."""
    return DomainService.get_instance()


def normalize_sector(sector_name: str | None) -> Sector:
    """
    Normalize sector name to Sector enum.
    
    This provides backward compatibility with code that used
    app/quant_engine/domain_analysis.py:normalize_sector()
    """
    if not sector_name:
        return Sector.UNKNOWN
    
    sector_lower = sector_name.lower().strip()
    
    SECTOR_MAPPING = {
        "technology": Sector.TECHNOLOGY,
        "information technology": Sector.TECHNOLOGY,
        "tech": Sector.TECHNOLOGY,
        "financials": Sector.FINANCIALS,
        "financial services": Sector.FINANCIALS,
        "banks": Sector.FINANCIALS,
        "healthcare": Sector.HEALTHCARE,
        "health care": Sector.HEALTHCARE,
        "consumer discretionary": Sector.CONSUMER_DISCRETIONARY,
        "consumer cyclical": Sector.CONSUMER_DISCRETIONARY,
        "consumer staples": Sector.CONSUMER_STAPLES,
        "consumer defensive": Sector.CONSUMER_STAPLES,
        "energy": Sector.ENERGY,
        "industrials": Sector.INDUSTRIALS,
        "materials": Sector.MATERIALS,
        "basic materials": Sector.MATERIALS,
        "real estate": Sector.REAL_ESTATE,
        "utilities": Sector.UTILITIES,
        "communication services": Sector.COMMUNICATION,
        "communication": Sector.COMMUNICATION,
    }
    
    return SECTOR_MAPPING.get(sector_lower, Sector.UNKNOWN)
