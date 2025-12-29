"""
Sector-relative valuation analysis using stored fundamentals data.

Computes how a stock's valuation metrics compare to its sector median.
Uses locally stored fundamentals data from scheduled jobs - no ad-hoc API calls.
"""

from dataclasses import dataclass

from app.services.fundamentals import get_fundamentals_from_db


@dataclass
class SectorRelativeValuation:
    """Sector-relative valuation metrics."""

    symbol: str
    sector: str | None

    # Raw valuation metrics (from stored fundamentals)
    pe_ratio: float | None
    forward_pe: float | None
    peg_ratio: float | None
    price_to_book: float | None
    ev_to_ebitda: float | None

    # Sector comparison (requires sector medians to be stored/computed)
    # For now, we just expose raw metrics and flag extreme valuations
    pe_percentile: float | None = None
    pb_percentile: float | None = None

    # Simple flags based on absolute thresholds
    is_deeply_undervalued: bool = False
    is_overvalued: bool = False
    valuation_score: float = 0.5  # 0-1, higher = more attractive valuation

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "symbol": self.symbol,
            "sector": self.sector,
            "pe_ratio": self.pe_ratio,
            "forward_pe": self.forward_pe,
            "peg_ratio": self.peg_ratio,
            "price_to_book": self.price_to_book,
            "ev_to_ebitda": self.ev_to_ebitda,
            "pe_percentile": self.pe_percentile,
            "pb_percentile": self.pb_percentile,
            "is_deeply_undervalued": self.is_deeply_undervalued,
            "is_overvalued": self.is_overvalued,
            "valuation_score": self.valuation_score,
        }


def compute_sector_relative_valuation_from_stored(
    symbol: str,
    stored_fundamentals: dict,
) -> SectorRelativeValuation:
    """
    Compute sector-relative valuation from stored fundamentals data.

    Args:
        symbol: Stock ticker symbol
        stored_fundamentals: Dict from get_fundamentals_from_db() containing
            sector, pe_ratio, forward_pe, peg_ratio, price_to_book, ev_to_ebitda

    Returns:
        SectorRelativeValuation with metrics and flags
    """
    if not stored_fundamentals:
        return SectorRelativeValuation(
            symbol=symbol,
            sector=None,
            pe_ratio=None,
            forward_pe=None,
            peg_ratio=None,
            price_to_book=None,
            ev_to_ebitda=None,
        )

    sector = stored_fundamentals.get("sector")
    pe_ratio = stored_fundamentals.get("pe_ratio")
    forward_pe = stored_fundamentals.get("forward_pe")
    peg_ratio = stored_fundamentals.get("peg_ratio")
    price_to_book = stored_fundamentals.get("price_to_book")
    ev_to_ebitda = stored_fundamentals.get("ev_to_ebitda")

    # Compute valuation score based on available metrics
    valuation_score = _compute_valuation_score(
        pe_ratio=pe_ratio,
        forward_pe=forward_pe,
        peg_ratio=peg_ratio,
        price_to_book=price_to_book,
    )

    # Flag extreme valuations based on absolute thresholds
    is_deeply_undervalued = _is_deeply_undervalued(
        pe_ratio=pe_ratio,
        forward_pe=forward_pe,
        peg_ratio=peg_ratio,
        price_to_book=price_to_book,
    )

    is_overvalued = _is_overvalued(
        pe_ratio=pe_ratio,
        forward_pe=forward_pe,
        peg_ratio=peg_ratio,
    )

    return SectorRelativeValuation(
        symbol=symbol,
        sector=sector,
        pe_ratio=pe_ratio,
        forward_pe=forward_pe,
        peg_ratio=peg_ratio,
        price_to_book=price_to_book,
        ev_to_ebitda=ev_to_ebitda,
        is_deeply_undervalued=is_deeply_undervalued,
        is_overvalued=is_overvalued,
        valuation_score=valuation_score,
    )


def _compute_valuation_score(
    pe_ratio: float | None,
    forward_pe: float | None,
    peg_ratio: float | None,
    price_to_book: float | None,
) -> float:
    """
    Compute a 0-1 valuation attractiveness score.

    Higher score = more attractive (cheaper) valuation.
    Uses absolute thresholds as proxy for sector comparison.
    """
    scores: list[float] = []

    # PE ratio scoring (lower is better, but negative PE = unprofitable)
    if pe_ratio is not None and pe_ratio > 0:
        if pe_ratio < 10:
            scores.append(1.0)
        elif pe_ratio < 15:
            scores.append(0.8)
        elif pe_ratio < 20:
            scores.append(0.6)
        elif pe_ratio < 30:
            scores.append(0.4)
        elif pe_ratio < 50:
            scores.append(0.2)
        else:
            scores.append(0.1)

    # Forward PE scoring
    if forward_pe is not None and forward_pe > 0:
        if forward_pe < 10:
            scores.append(1.0)
        elif forward_pe < 15:
            scores.append(0.8)
        elif forward_pe < 20:
            scores.append(0.6)
        elif forward_pe < 30:
            scores.append(0.4)
        else:
            scores.append(0.2)

    # PEG ratio scoring (< 1 is ideal)
    if peg_ratio is not None and peg_ratio > 0:
        if peg_ratio < 0.5:
            scores.append(1.0)
        elif peg_ratio < 1.0:
            scores.append(0.8)
        elif peg_ratio < 1.5:
            scores.append(0.6)
        elif peg_ratio < 2.0:
            scores.append(0.4)
        else:
            scores.append(0.2)

    # Price to book scoring
    if price_to_book is not None and price_to_book > 0:
        if price_to_book < 1.0:
            scores.append(1.0)
        elif price_to_book < 2.0:
            scores.append(0.7)
        elif price_to_book < 4.0:
            scores.append(0.5)
        else:
            scores.append(0.3)

    if not scores:
        return 0.5  # Neutral if no data

    return sum(scores) / len(scores)


def _is_deeply_undervalued(
    pe_ratio: float | None,
    forward_pe: float | None,
    peg_ratio: float | None,
    price_to_book: float | None,
) -> bool:
    """Check if stock appears deeply undervalued based on multiple metrics."""
    signals = 0
    total = 0

    if pe_ratio is not None and pe_ratio > 0:
        total += 1
        if pe_ratio < 12:
            signals += 1

    if forward_pe is not None and forward_pe > 0:
        total += 1
        if forward_pe < 10:
            signals += 1

    if peg_ratio is not None and peg_ratio > 0:
        total += 1
        if peg_ratio < 0.8:
            signals += 1

    if price_to_book is not None and price_to_book > 0:
        total += 1
        if price_to_book < 1.5:
            signals += 1

    # Need at least 2 metrics and majority showing undervaluation
    return total >= 2 and signals >= total * 0.5


def _is_overvalued(
    pe_ratio: float | None,
    forward_pe: float | None,
    peg_ratio: float | None,
) -> bool:
    """Check if stock appears overvalued based on multiple metrics."""
    signals = 0
    total = 0

    if pe_ratio is not None and pe_ratio > 0:
        total += 1
        if pe_ratio > 40:
            signals += 1

    if forward_pe is not None and forward_pe > 0:
        total += 1
        if forward_pe > 35:
            signals += 1

    if peg_ratio is not None and peg_ratio > 0:
        total += 1
        if peg_ratio > 2.5:
            signals += 1

    # Need at least 2 metrics and majority showing overvaluation
    return total >= 2 and signals >= total * 0.5


async def compute_sector_relative_valuation(
    symbol: str,
    stored_fundamentals: dict | None = None,
) -> SectorRelativeValuation:
    """
    Compute sector-relative valuation for a symbol.

    Args:
        symbol: Stock ticker symbol
        stored_fundamentals: Optional pre-loaded fundamentals dict.
            If not provided, loads from database.

    Returns:
        SectorRelativeValuation with metrics and flags
    """
    if stored_fundamentals is None:
        stored_fundamentals = await get_fundamentals_from_db(symbol)

    if stored_fundamentals is None:
        stored_fundamentals = {}

    return compute_sector_relative_valuation_from_stored(symbol, stored_fundamentals)
