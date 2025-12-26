"""
Portfolio Manager Agent.

Aggregates signals from all agents and produces final portfolio decisions.
Pure calculation-based - deterministic aggregation logic.
"""

import logging
from collections import defaultdict

from app.hedge_fund.agents.base import AgentSignal, CalculationAgentBase
from app.hedge_fund.schemas import (
    AgentType,
    MarketData,
    PerTickerReport,
    PortfolioDecision,
    Signal,
)


logger = logging.getLogger(__name__)


# Signal to numeric mapping
SIGNAL_VALUES = {
    Signal.STRONG_BUY: 2,
    Signal.BUY: 1,
    Signal.HOLD: 0,
    Signal.SELL: -1,
    Signal.STRONG_SELL: -2,
}

# Reverse mapping
VALUE_TO_SIGNAL = {
    2: Signal.STRONG_BUY,
    1: Signal.BUY,
    0: Signal.HOLD,
    -1: Signal.SELL,
    -2: Signal.STRONG_SELL,
}


class PortfolioManager(CalculationAgentBase):
    """
    Aggregates signals from all agents and produces portfolio decisions.
    
    Uses confidence-weighted voting to determine consensus.
    """

    def __init__(
        self,
        max_allocation_per_stock: float = 0.10,
        min_confidence_threshold: float = 0.5,
    ):
        super().__init__(
            agent_id="portfolio_manager",
            agent_name="Portfolio Manager",
            agent_type=AgentType.PORTFOLIO,
        )
        self.max_allocation = max_allocation_per_stock
        self.min_confidence = min_confidence_threshold

    async def calculate(self, symbol: str, data: MarketData) -> AgentSignal:
        """Not used directly - use aggregate_signals instead."""
        raise NotImplementedError("Use aggregate_signals() instead")

    def aggregate_signals(
        self,
        signals: list[AgentSignal],
    ) -> PerTickerReport:
        """
        Aggregate multiple agent signals into a consensus report.
        
        Filters signals below min_confidence threshold, then uses
        confidence-weighted voting to determine consensus signal.
        """
        if not signals:
            return PerTickerReport(
                symbol="UNKNOWN",
                signals=[],
                consensus_signal=Signal.HOLD,
                consensus_confidence=0.0,
                summary="No signals to aggregate",
            )

        symbol = signals[0].symbol

        # Filter signals below min_confidence threshold
        # Low-confidence signals should not influence the consensus
        confident_signals = [s for s in signals if s.confidence >= self.min_confidence]

        if not confident_signals:
            # All signals below threshold - return hold with low confidence
            return PerTickerReport(
                symbol=symbol,
                signals=signals,  # Keep original signals for reference
                consensus_signal=Signal.HOLD,
                consensus_confidence=0.0,
                summary="All signals below confidence threshold",
                bullish_count=0,
                bearish_count=0,
                neutral_count=len(signals),
            )

        # Confidence-weighted voting on filtered signals
        weighted_votes = defaultdict(float)
        total_weight = 0.0

        bullish_count = 0
        bearish_count = 0
        neutral_count = 0

        for signal in confident_signals:
            weight = signal.confidence
            weighted_votes[signal.signal] += weight
            total_weight += weight

            if signal.signal in (Signal.STRONG_BUY, Signal.BUY):
                bullish_count += 1
            elif signal.signal in (Signal.STRONG_SELL, Signal.SELL):
                bearish_count += 1
            else:
                neutral_count += 1

        # Find consensus signal
        if weighted_votes:
            consensus_signal = max(weighted_votes, key=weighted_votes.get)
            consensus_weight = weighted_votes[consensus_signal]
            consensus_confidence = consensus_weight / total_weight if total_weight > 0 else 0
        else:
            consensus_signal = Signal.HOLD
            consensus_confidence = 0.0

        # Calculate weighted average score for summary
        weighted_score = sum(
            SIGNAL_VALUES[s.signal] * s.confidence for s in confident_signals
        ) / total_weight if total_weight > 0 else 0

        # Build summary
        summary = self._build_summary(
            signals, consensus_signal, consensus_confidence,
            bullish_count, bearish_count, neutral_count, weighted_score
        )

        return PerTickerReport(
            symbol=symbol,
            signals=signals,
            consensus_signal=consensus_signal,
            consensus_confidence=consensus_confidence,
            bullish_count=bullish_count,
            bearish_count=bearish_count,
            neutral_count=neutral_count,
            summary=summary,
        )

    def create_portfolio_decision(
        self,
        report: PerTickerReport,
        risk_score: float = 0.5,
        portfolio_size: float | None = None,
    ) -> PortfolioDecision:
        """
        Create a portfolio decision from an aggregated report.
        
        Args:
            report: Aggregated ticker report
            risk_score: Overall risk score from risk agent (0-1)
            portfolio_size: Total portfolio value for position sizing
        """
        # Base allocation based on signal strength
        signal_value = SIGNAL_VALUES[report.consensus_signal]

        if signal_value > 0:
            # Buy signal - allocate based on confidence
            base_allocation = min(
                self.max_allocation,
                (signal_value / 2) * report.consensus_confidence * self.max_allocation
            )
        elif signal_value < 0:
            # Sell signal - no new allocation
            base_allocation = 0.0
        else:
            # Hold - minimal allocation
            base_allocation = 0.01 if report.consensus_confidence > 0.6 else 0.0

        # Adjust for risk
        risk_factor = 1 - (risk_score * 0.5)  # Max 50% reduction for high risk
        allocation = base_allocation * risk_factor

        # Position size if portfolio size known
        position_size = allocation * portfolio_size if portfolio_size else None

        # Stop loss based on risk
        if risk_score < 0.3:
            stop_loss = 0.08  # 8% for low risk
        elif risk_score < 0.5:
            stop_loss = 0.10  # 10% for medium risk
        elif risk_score < 0.7:
            stop_loss = 0.12  # 12% for high risk
        else:
            stop_loss = 0.15  # 15% for very high risk

        # Take profit based on signal strength
        take_profit = None
        if signal_value > 0:
            take_profit = 0.15 + (signal_value * 0.05)  # 15-25%

        # Build reasoning
        reasoning = self._build_decision_reasoning(report, risk_score, allocation)

        return PortfolioDecision(
            symbol=report.symbol,
            action=report.consensus_signal,
            allocation_pct=round(allocation, 4),
            position_size=round(position_size, 2) if position_size else None,
            stop_loss_pct=stop_loss,
            take_profit_pct=take_profit,
            reasoning=reasoning,
            risk_score=risk_score,
        )

    def rank_opportunities(
        self,
        reports: list[PerTickerReport],
    ) -> list[tuple[str, float]]:
        """
        Rank stocks by opportunity score.
        
        Returns list of (symbol, score) tuples, sorted by score descending.
        """
        scored = []

        for report in reports:
            signal_value = SIGNAL_VALUES[report.consensus_signal]

            # Opportunity score combines signal strength, confidence, and agreement
            score = (
                signal_value * 0.4  # Signal direction
                + report.consensus_confidence * 0.3  # Confidence
                + report.agent_agreement * 0.3  # Agent agreement
            )

            scored.append((report.symbol, score))

        # Sort by score descending
        return sorted(scored, key=lambda x: x[1], reverse=True)

    def _build_summary(
        self,
        signals: list[AgentSignal],
        consensus: Signal,
        confidence: float,
        bullish: int,
        bearish: int,
        neutral: int,
        weighted_score: float,
    ) -> str:
        """Build a human-readable summary."""
        parts = []

        # Consensus
        parts.append(f"Consensus: {consensus.value.upper()} with {confidence:.0%} confidence.")

        # Vote breakdown
        total = bullish + bearish + neutral
        parts.append(f"Votes: {bullish} bullish, {bearish} bearish, {neutral} neutral (of {total} agents).")

        # Weighted score
        score_desc = "bullish" if weighted_score > 0.5 else "bearish" if weighted_score < -0.5 else "neutral"
        parts.append(f"Weighted score: {weighted_score:+.2f} ({score_desc}).")

        # Top factors across agents
        all_factors = []
        for s in signals:
            if (s.signal in (Signal.STRONG_BUY, Signal.BUY) and consensus in (Signal.STRONG_BUY, Signal.BUY)) or (s.signal in (Signal.STRONG_SELL, Signal.SELL) and consensus in (Signal.STRONG_SELL, Signal.SELL)):
                all_factors.extend(s.key_factors[:2])

        if all_factors:
            unique_factors = list(dict.fromkeys(all_factors))[:3]
            parts.append(f"Key factors: {'; '.join(unique_factors)}.")

        return " ".join(parts)

    def _build_decision_reasoning(
        self,
        report: PerTickerReport,
        risk_score: float,
        allocation: float,
    ) -> str:
        """Build reasoning for portfolio decision."""
        parts = []

        parts.append(f"Based on {len(report.signals)} agent analyses.")
        parts.append(f"Consensus {report.consensus_signal.value} with {report.consensus_confidence:.0%} agreement.")

        if allocation > 0:
            parts.append(f"Recommended {allocation:.1%} allocation, adjusted for {risk_score:.0%} risk score.")
        else:
            parts.append("No allocation recommended based on current signals.")

        if report.summary:
            parts.append(report.summary)

        return " ".join(parts)


# =============================================================================
# Helper Functions
# =============================================================================


def calculate_consensus(signals: list[AgentSignal]) -> Signal:
    """Calculate simple consensus from signals."""
    if not signals:
        return Signal.HOLD

    total_value = sum(SIGNAL_VALUES[s.signal] * s.confidence for s in signals)
    total_weight = sum(s.confidence for s in signals)

    if total_weight == 0:
        return Signal.HOLD

    avg_value = total_value / total_weight

    # Map to nearest signal
    if avg_value >= 1.5:
        return Signal.STRONG_BUY
    elif avg_value >= 0.5:
        return Signal.BUY
    elif avg_value >= -0.5:
        return Signal.HOLD
    elif avg_value >= -1.5:
        return Signal.SELL
    else:
        return Signal.STRONG_SELL


def calculate_confidence(signals: list[AgentSignal]) -> float:
    """Calculate overall confidence from signals."""
    if not signals:
        return 0.0

    # Average confidence weighted by how many agents agree
    consensus = calculate_consensus(signals)
    agreeing = [s for s in signals if s.signal == consensus]

    if not agreeing:
        return sum(s.confidence for s in signals) / len(signals)

    return sum(s.confidence for s in agreeing) / len(signals)


# Singleton
_portfolio_manager: PortfolioManager | None = None


def get_portfolio_manager() -> PortfolioManager:
    """Get singleton portfolio manager."""
    global _portfolio_manager
    if _portfolio_manager is None:
        _portfolio_manager = PortfolioManager()
    return _portfolio_manager
