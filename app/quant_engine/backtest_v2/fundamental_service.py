"""
FundamentalService - YFinance Data Engine with Point-in-Time Alignment.

This module provides fundamentals data for backtesting with proper
look-ahead bias prevention. All data is aligned to daily dates using
forward-fill so each backtest day sees only the LAST REPORTED quarterly numbers.

Key Metrics Calculated:
- Net Cash = Cash & Equivalents - Total Debt
- Operating Margin = Operating Income / Total Revenue
- FCF = Free Cash Flow (or Operating Cash Flow - CapEx)
- Debt to Equity ratio
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from enum import Enum
from functools import lru_cache
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# YFinance field mappings (they vary between APIs)
YFINANCE_FIELDS = {
    # Balance Sheet
    "cash": ["Cash And Cash Equivalents", "Cash", "CashAndCashEquivalents"],
    "total_debt": ["Total Debt", "TotalDebt", "Long Term Debt", "LongTermDebt"],
    "total_equity": ["Total Stockholders Equity", "Stockholders Equity", "Total Equity Gross Minority Interest"],
    "current_assets": ["Total Current Assets", "CurrentAssets"],
    "current_liabilities": ["Total Current Liabilities", "CurrentLiabilities"],
    
    # Income Statement
    "revenue": ["Total Revenue", "Revenue", "TotalRevenue"],
    "operating_income": ["Operating Income", "OperatingIncome", "EBIT"],
    "net_income": ["Net Income", "NetIncome", "Net Income Common Stockholders"],
    "gross_profit": ["Gross Profit", "GrossProfit"],
    
    # Cash Flow
    "operating_cash_flow": ["Operating Cash Flow", "Cash Flow From Continuing Operating Activities", "Total Cash From Operating Activities"],
    "capex": ["Capital Expenditure", "CapEx", "Capital Expenditures"],
    "free_cash_flow": ["Free Cash Flow", "FreeCashFlow"],
}


@dataclass
class QuarterlyFundamentals:
    """Fundamental data for a single quarter."""
    
    report_date: pd.Timestamp
    
    # Balance Sheet
    cash: float | None = None
    total_debt: float | None = None
    total_equity: float | None = None
    current_assets: float | None = None
    current_liabilities: float | None = None
    
    # Income Statement
    revenue: float | None = None
    operating_income: float | None = None
    net_income: float | None = None
    gross_profit: float | None = None
    
    # Cash Flow
    operating_cash_flow: float | None = None
    capex: float | None = None
    free_cash_flow: float | None = None
    
    # Calculated Metrics (populated by service)
    net_cash: float | None = None
    operating_margin: float | None = None
    debt_to_equity: float | None = None
    current_ratio: float | None = None
    fcf: float | None = None
    
    def calculate_derived_metrics(self) -> None:
        """Calculate derived metrics from raw data."""
        # Net Cash = Cash - Debt
        if self.cash is not None and self.total_debt is not None:
            self.net_cash = self.cash - self.total_debt
        elif self.cash is not None:
            self.net_cash = self.cash  # No debt = all cash is net
        
        # Operating Margin = Operating Income / Revenue
        if self.operating_income is not None and self.revenue is not None and self.revenue != 0:
            self.operating_margin = self.operating_income / self.revenue
        
        # Debt to Equity
        if self.total_debt is not None and self.total_equity is not None and self.total_equity != 0:
            self.debt_to_equity = self.total_debt / self.total_equity
        elif self.total_debt is None and self.total_equity is not None:
            self.debt_to_equity = 0.0  # No debt
        
        # Current Ratio
        if self.current_assets is not None and self.current_liabilities is not None and self.current_liabilities != 0:
            self.current_ratio = self.current_assets / self.current_liabilities
        
        # FCF - prefer direct, fallback to OCF - CapEx
        if self.free_cash_flow is not None:
            self.fcf = self.free_cash_flow
        elif self.operating_cash_flow is not None and self.capex is not None:
            # CapEx is usually negative in yfinance
            self.fcf = self.operating_cash_flow + self.capex  # Adding because capex is negative
        elif self.operating_cash_flow is not None:
            self.fcf = self.operating_cash_flow  # Best guess


class MetaRuleResult(str, Enum):
    """Result of the META Rule fundamental check."""
    
    APPROVED = "APPROVED"  # Buy signal valid
    APPROVED_DEEP_VALUE = "APPROVED_DEEP_VALUE"  # Approved via net cash override
    BLOCKED_CASH_BURN = "BLOCKED_CASH_BURN"  # FCF < 0
    BLOCKED_INSOLVENT = "BLOCKED_INSOLVENT"  # D/E > 2.0
    BLOCKED_VALUE_TRAP = "BLOCKED_VALUE_TRAP"  # Declining margins + no cash buffer
    BLOCKED_NO_DATA = "BLOCKED_NO_DATA"  # Insufficient data


@dataclass
class MetaRuleDecision:
    """Complete decision from the META Rule check."""
    
    result: MetaRuleResult
    reason: str
    
    # Supporting data
    fcf: float | None = None
    debt_to_equity: float | None = None
    operating_margin: float | None = None
    margin_1y_ago: float | None = None
    margin_change_pct: float | None = None
    net_cash: float | None = None
    
    @property
    def approved(self) -> bool:
        """Check if the buy is approved."""
        return self.result in [MetaRuleResult.APPROVED, MetaRuleResult.APPROVED_DEEP_VALUE]
    
    @property
    def is_deep_value(self) -> bool:
        """Check if approved via deep value override."""
        return self.result == MetaRuleResult.APPROVED_DEEP_VALUE


class FundamentalService:
    """
    YFinance-based fundamental data service with point-in-time alignment.
    
    This service:
    1. Fetches quarterly financials, balance sheet, and cash flow from yfinance
    2. Transposes data (dates as index)
    3. Aligns to daily dates using forward-fill (no look-ahead bias)
    4. Calculates derived metrics (Net Cash, Op Margin, FCF)
    """
    
    def __init__(self, cache_dir: str | None = None):
        self._cache: dict[str, pd.DataFrame] = {}
        self.cache_dir = cache_dir
    
    def get_aligned_fundamentals(
        self,
        symbol: str,
        daily_dates: pd.DatetimeIndex,
    ) -> pd.DataFrame:
        """
        Get fundamentals aligned to daily dates.
        
        Each day will have the most recent quarterly data available
        at that point in time (forward-filled).
        
        Args:
            symbol: Stock ticker
            daily_dates: DatetimeIndex of backtest dates
            
        Returns:
            DataFrame with columns for each metric, indexed by daily_dates
        """
        cache_key = f"{symbol}_aligned"
        if cache_key in self._cache:
            cached = self._cache[cache_key]
            # Reindex to requested dates
            return cached.reindex(daily_dates, method="ffill")
        
        # Fetch raw quarterly data
        quarterly_df = self._fetch_quarterly_data(symbol)
        
        if quarterly_df.empty:
            logger.warning(f"No fundamental data found for {symbol}")
            return pd.DataFrame(index=daily_dates)
        
        # Forward-fill to daily dates
        # First, reindex to include all daily dates
        all_dates = quarterly_df.index.union(daily_dates)
        aligned = quarterly_df.reindex(all_dates).sort_index()
        
        # Forward fill - each day sees the last reported data
        aligned = aligned.ffill()
        
        # Filter to requested dates
        aligned = aligned.loc[aligned.index.isin(daily_dates)]
        
        # Cache
        self._cache[cache_key] = aligned
        
        return aligned
    
    def get_fundamentals_at_date(
        self,
        symbol: str,
        target_date: pd.Timestamp | date,
    ) -> QuarterlyFundamentals | None:
        """
        Get the most recent fundamentals available at a specific date.
        
        This ensures we only use data that was PUBLICLY AVAILABLE
        at the target date (no look-ahead bias).
        
        Args:
            symbol: Stock ticker
            target_date: The date to check
            
        Returns:
            QuarterlyFundamentals or None if no data
        """
        if isinstance(target_date, date):
            target_date = pd.Timestamp(target_date)
        
        quarterly_df = self._fetch_quarterly_data(symbol)
        
        if quarterly_df.empty:
            return None
        
        # Find most recent quarter BEFORE or ON target date
        available = quarterly_df[quarterly_df.index <= target_date]
        
        if available.empty:
            return None
        
        # Get the most recent row
        latest = available.iloc[-1]
        report_date = available.index[-1]
        
        fundamentals = QuarterlyFundamentals(
            report_date=report_date,
            cash=self._safe_get(latest, "cash"),
            total_debt=self._safe_get(latest, "total_debt"),
            total_equity=self._safe_get(latest, "total_equity"),
            current_assets=self._safe_get(latest, "current_assets"),
            current_liabilities=self._safe_get(latest, "current_liabilities"),
            revenue=self._safe_get(latest, "revenue"),
            operating_income=self._safe_get(latest, "operating_income"),
            net_income=self._safe_get(latest, "net_income"),
            gross_profit=self._safe_get(latest, "gross_profit"),
            operating_cash_flow=self._safe_get(latest, "operating_cash_flow"),
            capex=self._safe_get(latest, "capex"),
            free_cash_flow=self._safe_get(latest, "free_cash_flow"),
        )
        
        fundamentals.calculate_derived_metrics()
        return fundamentals
    
    def get_margin_1y_ago(
        self,
        symbol: str,
        target_date: pd.Timestamp | date,
    ) -> float | None:
        """Get operating margin from approximately 1 year ago."""
        if isinstance(target_date, date):
            target_date = pd.Timestamp(target_date)
        
        one_year_ago = target_date - pd.Timedelta(days=365)
        
        quarterly_df = self._fetch_quarterly_data(symbol)
        
        if quarterly_df.empty or "operating_margin" not in quarterly_df.columns:
            return None
        
        # Find data around 1 year ago
        available = quarterly_df[quarterly_df.index <= one_year_ago]
        
        if available.empty:
            return None
        
        return self._safe_get(available.iloc[-1], "operating_margin")
    
    def _fetch_quarterly_data(self, symbol: str) -> pd.DataFrame:
        """
        Fetch and combine quarterly financial data from yfinance.
        
        Returns DataFrame with dates as index and metrics as columns.
        """
        cache_key = f"{symbol}_quarterly"
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        try:
            import yfinance as yf
            
            ticker = yf.Ticker(symbol)
            
            # Fetch all three statement types
            income_stmt = self._transpose_statement(ticker.quarterly_income_stmt)
            balance_sheet = self._transpose_statement(ticker.quarterly_balance_sheet)
            cash_flow = self._transpose_statement(ticker.quarterly_cashflow)
            
            # Combine into single DataFrame
            combined = pd.DataFrame()
            
            # Process each statement
            for stmt, source_name in [
                (income_stmt, "income"),
                (balance_sheet, "balance"),
                (cash_flow, "cashflow"),
            ]:
                if stmt.empty:
                    continue
                
                # Extract known fields
                for target_col, source_names in YFINANCE_FIELDS.items():
                    for src in source_names:
                        if src in stmt.columns:
                            combined[target_col] = stmt[src]
                            break
            
            if combined.empty:
                logger.warning(f"No quarterly data found for {symbol}")
                return pd.DataFrame()
            
            # Sort by date
            combined = combined.sort_index()
            
            # Calculate derived metrics
            if "operating_income" in combined.columns and "revenue" in combined.columns:
                combined["operating_margin"] = combined["operating_income"] / combined["revenue"].replace(0, np.nan)
            
            if "cash" in combined.columns and "total_debt" in combined.columns:
                combined["net_cash"] = combined["cash"] - combined["total_debt"].fillna(0)
            
            if "total_debt" in combined.columns and "total_equity" in combined.columns:
                combined["debt_to_equity"] = combined["total_debt"] / combined["total_equity"].replace(0, np.nan)
            
            if "current_assets" in combined.columns and "current_liabilities" in combined.columns:
                combined["current_ratio"] = combined["current_assets"] / combined["current_liabilities"].replace(0, np.nan)
            
            # FCF
            if "free_cash_flow" in combined.columns:
                combined["fcf"] = combined["free_cash_flow"]
            elif "operating_cash_flow" in combined.columns and "capex" in combined.columns:
                combined["fcf"] = combined["operating_cash_flow"] + combined["capex"]  # capex is negative
            
            self._cache[cache_key] = combined
            return combined
            
        except Exception as e:
            logger.error(f"Error fetching quarterly data for {symbol}: {e}")
            return pd.DataFrame()
    
    def _transpose_statement(self, stmt: pd.DataFrame) -> pd.DataFrame:
        """Transpose yfinance statement so dates are index."""
        if stmt is None or stmt.empty:
            return pd.DataFrame()
        
        # yfinance returns dates as columns, metrics as index
        # We want dates as index, metrics as columns
        transposed = stmt.T
        
        # Ensure index is DatetimeIndex
        transposed.index = pd.to_datetime(transposed.index)
        
        return transposed
    
    def _safe_get(self, row: pd.Series, col: str) -> float | None:
        """Safely get a value from a row."""
        if col not in row.index:
            return None
        val = row[col]
        if pd.isna(val):
            return None
        return float(val)
    
    def clear_cache(self) -> None:
        """Clear the internal cache."""
        self._cache.clear()


class BearMarketStrategyFilter:
    """
    META Rule implementation for bear market buying decisions.
    
    The "META Rule" decision tree:
    
    1. Hard Stops (always block):
       - FCF < 0 (cash burn)
       - Debt/Equity > 2.0 (insolvency risk)
    
    2. Deep Value Logic:
       - If margins stable/growing -> APPROVE
       - If margins declining (>10% drop):
         - If Net Cash > 0 -> APPROVE (Deep Value like META 2022)
         - If Net Cash <= 0 -> BLOCK (Value Trap)
    """
    
    def __init__(
        self,
        fundamental_service: FundamentalService | None = None,
        max_debt_to_equity: float = 2.0,
        margin_decline_threshold: float = 0.10,  # 10% decline triggers check
    ):
        self.fundamentals = fundamental_service or FundamentalService()
        self.max_debt_to_equity = max_debt_to_equity
        self.margin_decline_threshold = margin_decline_threshold
    
    def check_buy_signal(
        self,
        symbol: str,
        signal_date: pd.Timestamp | date,
    ) -> MetaRuleDecision:
        """
        Apply the META Rule to a potential buy signal.
        
        This should be called in Bear Market mode to filter buys.
        
        Args:
            symbol: Stock ticker
            signal_date: Date of the buy signal
            
        Returns:
            MetaRuleDecision with result and reasoning
        """
        if isinstance(signal_date, date):
            signal_date = pd.Timestamp(signal_date)
        
        # Get current fundamentals (point-in-time)
        current = self.fundamentals.get_fundamentals_at_date(symbol, signal_date)
        
        if current is None:
            return MetaRuleDecision(
                result=MetaRuleResult.BLOCKED_NO_DATA,
                reason=f"No fundamental data available for {symbol} at {signal_date}",
            )
        
        # Get margin from 1 year ago for trend
        margin_1y = self.fundamentals.get_margin_1y_ago(symbol, signal_date)
        
        # Calculate margin change
        margin_change_pct = None
        if current.operating_margin is not None and margin_1y is not None and margin_1y != 0:
            margin_change_pct = (current.operating_margin - margin_1y) / abs(margin_1y)
        
        # =========================================================================
        # RULE A: HARD STOPS
        # =========================================================================
        
        # Check 1: FCF < 0 (Cash Burn)
        if current.fcf is not None and current.fcf < 0:
            return MetaRuleDecision(
                result=MetaRuleResult.BLOCKED_CASH_BURN,
                reason=f"Negative FCF: ${current.fcf/1e9:.2f}B (cash burn)",
                fcf=current.fcf,
                debt_to_equity=current.debt_to_equity,
                operating_margin=current.operating_margin,
                margin_1y_ago=margin_1y,
                margin_change_pct=margin_change_pct,
                net_cash=current.net_cash,
            )
        
        # Check 2: D/E > 2.0 (Insolvency Risk)
        if current.debt_to_equity is not None and current.debt_to_equity > self.max_debt_to_equity:
            return MetaRuleDecision(
                result=MetaRuleResult.BLOCKED_INSOLVENT,
                reason=f"High debt: D/E ratio {current.debt_to_equity:.2f} > {self.max_debt_to_equity}",
                fcf=current.fcf,
                debt_to_equity=current.debt_to_equity,
                operating_margin=current.operating_margin,
                margin_1y_ago=margin_1y,
                margin_change_pct=margin_change_pct,
                net_cash=current.net_cash,
            )
        
        # =========================================================================
        # RULE B: DEEP VALUE LOGIC
        # =========================================================================
        
        # If we can't determine margin trend, approve with caution
        if margin_change_pct is None:
            return MetaRuleDecision(
                result=MetaRuleResult.APPROVED,
                reason="Passed hard stops. Margin trend unavailable - approved with caution.",
                fcf=current.fcf,
                debt_to_equity=current.debt_to_equity,
                operating_margin=current.operating_margin,
                margin_1y_ago=margin_1y,
                margin_change_pct=margin_change_pct,
                net_cash=current.net_cash,
            )
        
        # Margins stable or growing -> APPROVE
        if margin_change_pct >= -self.margin_decline_threshold:
            return MetaRuleDecision(
                result=MetaRuleResult.APPROVED,
                reason=f"Margins stable/growing ({margin_change_pct:+.1%} YoY). Quality confirmed.",
                fcf=current.fcf,
                debt_to_equity=current.debt_to_equity,
                operating_margin=current.operating_margin,
                margin_1y_ago=margin_1y,
                margin_change_pct=margin_change_pct,
                net_cash=current.net_cash,
            )
        
        # =========================================================================
        # MARGINS DECLINING - Check for Deep Value Override
        # =========================================================================
        
        # Net Cash > 0 -> Deep Value (META 2022 scenario)
        if current.net_cash is not None and current.net_cash > 0:
            return MetaRuleDecision(
                result=MetaRuleResult.APPROVED_DEEP_VALUE,
                reason=f"DEEP VALUE: Margins declining ({margin_change_pct:+.1%}) but Net Cash ${current.net_cash/1e9:.1f}B provides buffer.",
                fcf=current.fcf,
                debt_to_equity=current.debt_to_equity,
                operating_margin=current.operating_margin,
                margin_1y_ago=margin_1y,
                margin_change_pct=margin_change_pct,
                net_cash=current.net_cash,
            )
        
        # No cash buffer -> Value Trap
        return MetaRuleDecision(
            result=MetaRuleResult.BLOCKED_VALUE_TRAP,
            reason=f"VALUE TRAP: Margins declining ({margin_change_pct:+.1%}) with no cash buffer (Net Cash: ${(current.net_cash or 0)/1e9:.1f}B)",
            fcf=current.fcf,
            debt_to_equity=current.debt_to_equity,
            operating_margin=current.operating_margin,
            margin_1y_ago=margin_1y,
            margin_change_pct=margin_change_pct,
            net_cash=current.net_cash,
        )
    
    def filter_buy_signals(
        self,
        symbol: str,
        signal_dates: list[pd.Timestamp],
    ) -> list[tuple[pd.Timestamp, MetaRuleDecision]]:
        """
        Filter a list of buy signals through the META Rule.
        
        Returns list of (date, decision) tuples.
        """
        return [(d, self.check_buy_signal(symbol, d)) for d in signal_dates]
    
    def get_approved_signals(
        self,
        symbol: str,
        signal_dates: list[pd.Timestamp],
    ) -> list[pd.Timestamp]:
        """Get only the approved buy signal dates."""
        results = self.filter_buy_signals(symbol, signal_dates)
        return [d for d, decision in results if decision.approved]


def backtest_with_meta_rule(
    symbol: str,
    prices: pd.Series,
    raw_buy_signals: pd.Series,
    is_bear_market: pd.Series,
) -> tuple[pd.Series, list[MetaRuleDecision]]:
    """
    Apply META Rule to buy signals during bear markets.
    
    Args:
        symbol: Stock ticker
        prices: Price series with DatetimeIndex
        raw_buy_signals: Series with 1 for buy signals
        is_bear_market: Series with True for bear market days
        
    Returns:
        (filtered_signals, decisions) - Filtered buy signals and decision log
    """
    filter_engine = BearMarketStrategyFilter()
    filtered = raw_buy_signals.copy()
    decisions: list[MetaRuleDecision] = []
    
    # Find all buy signal dates in bear markets
    buy_dates = raw_buy_signals[raw_buy_signals == 1].index
    bear_buy_dates = [d for d in buy_dates if is_bear_market.get(d, False)]
    
    for signal_date in bear_buy_dates:
        decision = filter_engine.check_buy_signal(symbol, signal_date)
        decisions.append(decision)
        
        if not decision.approved:
            # Block this signal
            filtered.loc[signal_date] = 0
            logger.info(f"[{symbol}] {signal_date.date()}: {decision.result.value} - {decision.reason}")
    
    return filtered, decisions
