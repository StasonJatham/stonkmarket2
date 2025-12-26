"""
Domain Classification for Securities.

Maps securities to their analysis domain based on:
- quoteType (ETF, EQUITY, MUTUALFUND, INDEX)
- sector (Financial Services, Technology, etc.)
- industry (Banks—Regional, REIT—Retail, etc.)
- name patterns (contains "REIT", "Bancorp", etc.)

Domains determine which scoring adapter to use and which metrics are relevant.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional
import re

from app.core.logging import get_logger

logger = get_logger("dipfinder.domain")


class Domain(str, Enum):
    """Security domain for specialized analysis.
    
    Each domain may have different:
    - Relevant metrics (e.g., FFO for REITs, NIM for banks)
    - Scoring weights (e.g., ignore D/E for banks)
    - Quality gates (e.g., higher yield threshold for utilities)
    """
    
    # Financials
    BANK = "bank"
    INSURER = "insurer"
    ASSET_MANAGER = "asset_manager"
    
    # Real Estate
    REIT = "reit"
    
    # Funds & Indices
    ETF = "etf"
    MUTUAL_FUND = "mutual_fund"
    INDEX = "index"
    
    # Regulated Utilities
    UTILITY = "utility"
    TELECOM = "telecom"
    
    # Commodities & Energy
    ENERGY = "energy"
    MINING = "mining"
    
    # Growth / R&D Heavy
    BIOTECH = "biotech"
    PHARMA = "pharma"
    SAAS = "saas"
    
    # Capital Intensive
    AIRLINE = "airline"
    SHIPPING = "shipping"
    
    # Consumer / Cyclical
    RETAIL = "retail"
    
    # Hardware / Cyclical
    SEMICONDUCTOR = "semiconductor"
    
    # Complex Structures
    CONGLOMERATE = "conglomerate"
    FOREIGN_ADR = "foreign_adr"
    SPAC = "spac"
    
    # Default
    OPERATING_COMPANY = "operating_company"


@dataclass(frozen=True)
class DomainClassification:
    """Result of domain classification."""
    
    domain: Domain
    confidence: float  # 0.0-1.0
    reason: str
    fallback_domain: Optional[Domain] = None
    
    def __str__(self) -> str:
        return f"{self.domain.value} ({self.confidence:.0%}): {self.reason}"


# =============================================================================
# Classification Rules
# =============================================================================

# Industry patterns for each domain (case-insensitive regex)
_INDUSTRY_PATTERNS: dict[Domain, list[str]] = {
    Domain.BANK: [
        r"banks",
        r"savings.*loan",
        r"thrift",
        r"credit.*services",
        r"mortgage.*finance",
    ],
    Domain.INSURER: [
        r"insurance",
        r"reinsurance",
    ],
    Domain.ASSET_MANAGER: [
        r"asset.*management",
        r"investment.*banking",
        r"capital.*markets",
        r"financial.*data",
        r"stock.*exchange",
    ],
    Domain.REIT: [
        r"reit",
        r"real.*estate.*investment",
        r"real.*estate.*services",
    ],
    Domain.UTILITY: [
        r"utilities",
        r"regulated.*electric",
        r"regulated.*gas",
        r"water.*utilities",
        r"diversified.*utilities",
    ],
    Domain.TELECOM: [
        r"telecom",
        r"wireless.*telecom",
        r"communication.*equipment",
    ],
    Domain.ENERGY: [
        r"oil.*gas.*e&p",
        r"oil.*gas.*integrated",
        r"oil.*gas.*midstream",
        r"oil.*gas.*refin",
        r"oil.*gas.*drilling",
        r"oil.*gas.*equipment",
    ],
    Domain.MINING: [
        r"gold",
        r"silver",
        r"copper",
        r"coal",
        r"other.*industrial.*metals",
        r"steel",
        r"aluminum",
    ],
    Domain.BIOTECH: [
        r"biotechnology",
    ],
    Domain.PHARMA: [
        r"drug.*manufacturers",
        r"pharmaceutical",
    ],
    Domain.SAAS: [
        r"software.*application",
        r"software.*infrastructure",
    ],
    Domain.AIRLINE: [
        r"airlines",
    ],
    Domain.SHIPPING: [
        r"marine.*shipping",
        r"freight",
        r"railroads",
    ],
    Domain.RETAIL: [
        r"discount.*stores",
        r"department.*stores",
        r"specialty.*retail",
        r"apparel.*retail",
        r"internet.*retail",
        r"home.*improvement",
        r"auto.*dealerships",
    ],
    Domain.SEMICONDUCTOR: [
        r"semiconductor",
    ],
    Domain.CONGLOMERATE: [
        r"conglomerate",
        r"industrial.*conglomerate",
    ],
}

# Name patterns for fallback detection (case-insensitive)
_NAME_PATTERNS: dict[Domain, list[str]] = {
    Domain.BANK: [
        r"bancorp",
        r"bank\b",
        r"banc\b",
        r"financial\b",
    ],
    Domain.REIT: [
        r"\breit\b",
        r"realty",
        r"properties",
        r"real\s+estate",
    ],
    Domain.ETF: [
        r"\betf\b",
        r"ishares",
        r"vanguard",
        r"spdr\b",
        r"proshares",
        r"direxion",
        r"ark\s+",
    ],
    Domain.INSURER: [
        r"insurance",
        r"assurance",
        r"mutual\s+life",
    ],
    Domain.SPAC: [
        r"acquisition\s+corp",
        r"spac\b",
    ],
}

# Sector mappings (exact match, case-insensitive)
_SECTOR_DOMAINS: dict[str, Domain] = {
    "financial services": Domain.BANK,  # Default for financials, refined by industry
    "real estate": Domain.REIT,
    "utilities": Domain.UTILITY,
    "communication services": Domain.TELECOM,
    "energy": Domain.ENERGY,
    "basic materials": Domain.MINING,
    "healthcare": Domain.PHARMA,  # Refined by industry
}


def _matches_patterns(text: str, patterns: list[str]) -> bool:
    """Check if text matches any of the regex patterns."""
    if not text:
        return False
    text_lower = text.lower()
    for pattern in patterns:
        if re.search(pattern, text_lower):
            return True
    return False


def classify_domain(
    *,
    quote_type: Optional[str] = None,
    sector: Optional[str] = None,
    industry: Optional[str] = None,
    name: Optional[str] = None,
    symbol: Optional[str] = None,
) -> DomainClassification:
    """
    Classify a security into its analysis domain.
    
    Classification priority:
    1. quote_type (ETF, MUTUALFUND, INDEX → direct mapping)
    2. industry (most specific, highest confidence)
    3. name patterns (fallback when industry missing)
    4. sector (broad fallback)
    5. default to OPERATING_COMPANY
    
    Args:
        quote_type: From yfinance (ETF, EQUITY, MUTUALFUND, INDEX)
        sector: From yfinance (Financial Services, Technology, etc.)
        industry: From yfinance (Banks—Regional, REIT—Retail, etc.)
        name: Company name for pattern matching
        symbol: Ticker symbol (for ADR detection)
    
    Returns:
        DomainClassification with domain, confidence, and reason
    """
    
    # 1. Check quote_type first (highest priority for non-equity)
    if quote_type:
        qt = quote_type.upper()
        if qt == "ETF":
            return DomainClassification(
                domain=Domain.ETF,
                confidence=1.0,
                reason=f"quoteType={quote_type}",
            )
        if qt == "MUTUALFUND":
            return DomainClassification(
                domain=Domain.MUTUAL_FUND,
                confidence=1.0,
                reason=f"quoteType={quote_type}",
            )
        if qt == "INDEX":
            return DomainClassification(
                domain=Domain.INDEX,
                confidence=1.0,
                reason=f"quoteType={quote_type}",
            )
    
    # 2. Check industry patterns (most specific)
    if industry:
        industry_lower = industry.lower()
        for domain, patterns in _INDUSTRY_PATTERNS.items():
            if _matches_patterns(industry, patterns):
                # Refine financials by industry
                return DomainClassification(
                    domain=domain,
                    confidence=0.95,
                    reason=f"industry={industry}",
                )
    
    # 3. Check name patterns (fallback when industry missing)
    if name:
        for domain, patterns in _NAME_PATTERNS.items():
            if _matches_patterns(name, patterns):
                return DomainClassification(
                    domain=domain,
                    confidence=0.70,
                    reason=f"name pattern match: {name}",
                    fallback_domain=Domain.OPERATING_COMPANY,
                )
    
    # 4. Check sector (broad fallback)
    if sector:
        sector_lower = sector.lower()
        if sector_lower in _SECTOR_DOMAINS:
            return DomainClassification(
                domain=_SECTOR_DOMAINS[sector_lower],
                confidence=0.60,
                reason=f"sector={sector}",
                fallback_domain=Domain.OPERATING_COMPANY,
            )
    
    # 5. Check for ADR (ends in .DE, .L, .PA, etc. or has "ADR" in name)
    if symbol:
        if "." in symbol and len(symbol.split(".")[-1]) <= 3:
            suffix = symbol.split(".")[-1].upper()
            if suffix not in ("A", "B", "C", "K", "WS"):  # Not class shares or warrants
                return DomainClassification(
                    domain=Domain.FOREIGN_ADR,
                    confidence=0.80,
                    reason=f"foreign exchange suffix: .{suffix}",
                    fallback_domain=Domain.OPERATING_COMPANY,
                )
    if name and re.search(r"\bADR\b", name.upper()):
        return DomainClassification(
            domain=Domain.FOREIGN_ADR,
            confidence=0.85,
            reason="ADR in name",
            fallback_domain=Domain.OPERATING_COMPANY,
        )
    
    # Default: operating company
    return DomainClassification(
        domain=Domain.OPERATING_COMPANY,
        confidence=0.50,
        reason="default (no domain signals detected)",
    )


def get_domain_from_info(info: dict) -> DomainClassification:
    """
    Convenience function to classify from yfinance info dict.
    
    Args:
        info: Dict from yfinance_service.get_ticker_info()
    
    Returns:
        DomainClassification
    """
    return classify_domain(
        quote_type=info.get("quote_type"),
        sector=info.get("sector"),
        industry=info.get("industry"),
        name=info.get("name"),
        symbol=info.get("symbol"),
    )


# =============================================================================
# Domain Metadata
# =============================================================================

@dataclass(frozen=True)
class DomainMetadata:
    """Metadata about a domain for UI/documentation."""
    
    domain: Domain
    display_name: str
    description: str
    key_metrics: tuple[str, ...]
    skip_metrics: tuple[str, ...]
    notes: str = ""


# Domain-specific guidance for scoring
DOMAIN_METADATA: dict[Domain, DomainMetadata] = {
    Domain.BANK: DomainMetadata(
        domain=Domain.BANK,
        display_name="Bank / Financials",
        description="Deposit-taking institutions where leverage is the business model",
        key_metrics=("return_on_equity", "return_on_assets", "book_value", "dividend_yield"),
        skip_metrics=("debt_to_equity", "current_ratio", "quick_ratio", "free_cash_flow"),
        notes="D/E is meaningless for banks. ROE and book value are key.",
    ),
    Domain.INSURER: DomainMetadata(
        domain=Domain.INSURER,
        display_name="Insurance",
        description="Property, casualty, life, or reinsurance companies",
        key_metrics=("return_on_equity", "book_value", "dividend_yield"),
        skip_metrics=("debt_to_equity", "operating_margin", "free_cash_flow"),
        notes="Combined ratio is key but not in yfinance. Use ROE and book value.",
    ),
    Domain.REIT: DomainMetadata(
        domain=Domain.REIT,
        display_name="REIT",
        description="Real Estate Investment Trust",
        key_metrics=("dividend_yield", "price_to_book", "debt_to_equity"),
        skip_metrics=("free_cash_flow", "operating_cash_flow", "pe_ratio"),
        notes="FFO/AFFO are key but not in yfinance. Dividend yield is essential.",
    ),
    Domain.ETF: DomainMetadata(
        domain=Domain.ETF,
        display_name="ETF / Index Fund",
        description="Exchange-traded fund tracking an index or strategy",
        key_metrics=("market_cap",),
        skip_metrics=("pe_ratio", "debt_to_equity", "profit_margin", "free_cash_flow"),
        notes="Company-level metrics don't apply. Focus on dip/stability only.",
    ),
    Domain.UTILITY: DomainMetadata(
        domain=Domain.UTILITY,
        display_name="Utility",
        description="Regulated electric, gas, or water utility",
        key_metrics=("dividend_yield", "debt_to_equity", "operating_margin"),
        skip_metrics=("revenue_growth", "peg_ratio"),
        notes="Stable but capital-intensive. Dividend and debt coverage are key.",
    ),
    Domain.TELECOM: DomainMetadata(
        domain=Domain.TELECOM,
        display_name="Telecom",
        description="Telecommunications and wireless providers",
        key_metrics=("dividend_yield", "debt_to_equity", "operating_margin"),
        skip_metrics=("peg_ratio",),
        notes="Capital-intensive with regulatory dynamics. Similar to utilities.",
    ),
    Domain.BIOTECH: DomainMetadata(
        domain=Domain.BIOTECH,
        display_name="Biotechnology",
        description="Pre-revenue or R&D-heavy biotech company",
        key_metrics=("total_cash", "market_cap", "beta"),
        skip_metrics=("pe_ratio", "profit_margin", "free_cash_flow", "revenue_growth"),
        notes="Many are pre-revenue. Cash runway is critical.",
    ),
    Domain.PHARMA: DomainMetadata(
        domain=Domain.PHARMA,
        display_name="Pharmaceuticals",
        description="Established pharmaceutical company with revenue",
        key_metrics=("profit_margin", "free_cash_flow", "dividend_yield"),
        skip_metrics=(),
        notes="Established pharma uses standard operating company metrics.",
    ),
    Domain.ENERGY: DomainMetadata(
        domain=Domain.ENERGY,
        display_name="Energy",
        description="Oil, gas, and energy companies (E&P, midstream, refiners)",
        key_metrics=("operating_margin", "free_cash_flow", "debt_to_equity", "dividend_yield"),
        skip_metrics=("peg_ratio",),
        notes="Commodity-price dependent. FCF and leverage are key.",
    ),
    Domain.MINING: DomainMetadata(
        domain=Domain.MINING,
        display_name="Mining / Materials",
        description="Mining, metals, and basic materials companies",
        key_metrics=("operating_margin", "free_cash_flow", "debt_to_equity"),
        skip_metrics=("peg_ratio",),
        notes="Commodity exposure. Similar dynamics to energy.",
    ),
    Domain.RETAIL: DomainMetadata(
        domain=Domain.RETAIL,
        display_name="Retail / Consumer",
        description="Retail stores, e-commerce, and consumer goods",
        key_metrics=("profit_margin", "revenue_growth", "debt_to_equity"),
        skip_metrics=(),
        notes="Thin margins are normal. Growth and execution are key.",
    ),
    Domain.SEMICONDUCTOR: DomainMetadata(
        domain=Domain.SEMICONDUCTOR,
        display_name="Semiconductor",
        description="Semiconductor and hardware companies",
        key_metrics=("operating_margin", "revenue_growth", "total_cash"),
        skip_metrics=(),
        notes="Highly cyclical. High margins and growth are expected in upcycle.",
    ),
    Domain.AIRLINE: DomainMetadata(
        domain=Domain.AIRLINE,
        display_name="Airline",
        description="Commercial airlines and aviation",
        key_metrics=("operating_margin", "debt_to_equity", "free_cash_flow"),
        skip_metrics=("peg_ratio",),
        notes="Capital-intensive with thin margins. High debt is normal.",
    ),
    Domain.SHIPPING: DomainMetadata(
        domain=Domain.SHIPPING,
        display_name="Shipping / Transport",
        description="Marine shipping, freight, and transportation",
        key_metrics=("operating_margin", "debt_to_equity", "free_cash_flow"),
        skip_metrics=("peg_ratio",),
        notes="Capital-intensive. Similar characteristics to airlines.",
    ),
    Domain.SAAS: DomainMetadata(
        domain=Domain.SAAS,
        display_name="SaaS / Software",
        description="Software-as-a-service and subscription businesses",
        key_metrics=("revenue_growth", "profit_margin", "free_cash_flow"),
        skip_metrics=("pe_ratio",),  # Often negative earnings
        notes="Growth is prioritized. Rule of 40 applies (growth + margin > 40%).",
    ),
    Domain.CONGLOMERATE: DomainMetadata(
        domain=Domain.CONGLOMERATE,
        display_name="Conglomerate",
        description="Diversified holding company with multiple businesses",
        key_metrics=("profit_margin", "return_on_equity", "free_cash_flow"),
        skip_metrics=(),
        notes="Sum-of-the-parts analysis often needed. NAV discount common.",
    ),
    Domain.FOREIGN_ADR: DomainMetadata(
        domain=Domain.FOREIGN_ADR,
        display_name="Foreign ADR",
        description="American Depositary Receipt of foreign company",
        key_metrics=("profit_margin", "debt_to_equity", "dividend_yield"),
        skip_metrics=(),
        notes="Consider FX exposure and reporting lags. Apply underlying domain scoring.",
    ),
    Domain.OPERATING_COMPANY: DomainMetadata(
        domain=Domain.OPERATING_COMPANY,
        display_name="Operating Company",
        description="Standard operating company with traditional metrics",
        key_metrics=("profit_margin", "debt_to_equity", "free_cash_flow", "pe_ratio"),
        skip_metrics=(),
        notes="Default scoring with all standard metrics.",
    ),
}


def get_domain_metadata(domain: Domain) -> DomainMetadata:
    """Get metadata for a domain, with fallback to OPERATING_COMPANY."""
    return DOMAIN_METADATA.get(domain, DOMAIN_METADATA[Domain.OPERATING_COMPANY])
