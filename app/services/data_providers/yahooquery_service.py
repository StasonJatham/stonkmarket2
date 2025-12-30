"""
YahooQuery Service - Fallback data provider when yfinance fails.

This module provides an alternative data source using yahooquery library.
It mirrors the yfinance_service interface but uses yahooquery API calls.

Key advantages over yfinance:
- More reliable during yfinance rate limiting or outages
- Richer financial data (95+ balance sheet columns vs limited yfinance data)
- Better financial sector support (interest income/expense, loan loss provisions)
- Better valuation data (PeRatio, PegRatio, PsRatio, EnterpriseValue)
- Multi-symbol batch support for efficiency

Usage:
    from app.services.data_providers import get_yahooquery_service
    
    service = get_yahooquery_service()
    
    # Get stock info
    info = await service.get_ticker_info("AAPL")
    
    # Get financial statements
    financials = await service.get_financials("AAPL")
    
    # Get valuation measures
    valuation = await service.get_valuation_measures("AAPL")
"""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Literal

import pandas as pd

from app.core.logging import get_logger

# Lazy import to allow graceful degradation if yahooquery not installed
try:
    from yahooquery import Ticker
    YAHOOQUERY_AVAILABLE = True
except ImportError:
    YAHOOQUERY_AVAILABLE = False
    Ticker = None  # type: ignore

logger = get_logger("data_providers.yahooquery")

# Shared executor for blocking yahooquery calls
_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="yahooquery")


@dataclass
class YahooQueryResult:
    """Result from yahooquery fetch."""
    symbol: str
    data: dict[str, Any] | None
    status: Literal["success", "error", "not_available"]
    error_message: str | None = None


def _safe_float(value: Any) -> float | None:
    """Safely convert value to float."""
    if value is None:
        return None
    if isinstance(value, str):
        return None
    try:
        f = float(value)
        if f != f or f == float('inf') or f == float('-inf'):
            return None
        return f
    except (ValueError, TypeError):
        return None


def _safe_int(value: Any) -> int | None:
    """Safely convert value to int."""
    if value is None:
        return None
    if isinstance(value, str):
        return None
    try:
        return int(value)
    except (ValueError, TypeError):
        return None


def _safe_str(value: Any) -> str | None:
    """Safely convert value to string."""
    if value is None:
        return None
    return str(value) if value else None


def _is_valid_response(data: Any) -> bool:
    """Check if yahooquery returned valid data (not an error message)."""
    if data is None:
        return False
    if isinstance(data, str):
        # yahooquery returns error strings like "Quote not found for ticker symbol: XYZ"
        return False
    if isinstance(data, dict) and "Quote not found" in str(data.get("error", "")):
        return False
    return True


class YahooQueryService:
    """
    Fallback data provider using yahooquery library.
    
    Provides similar interface to YFinanceService for seamless fallback.
    Focuses on financial data richness for domain-specific analysis.
    """

    def __init__(self):
        if not YAHOOQUERY_AVAILABLE:
            logger.warning("yahooquery not installed - service will be unavailable")

    def is_available(self) -> bool:
        """Check if yahooquery is available."""
        return YAHOOQUERY_AVAILABLE

    # =========================================================================
    # Ticker Info
    # =========================================================================

    def _fetch_ticker_info_sync(self, symbol: str) -> dict[str, Any] | None:
        """
        Fetch ticker info from yahooquery (blocking).
        
        Returns normalized info dict similar to yfinance format.
        """
        if not YAHOOQUERY_AVAILABLE:
            return None

        try:
            ticker = Ticker(symbol, asynchronous=False)
            
            # Get summary profile for basic info
            profile = ticker.summary_profile.get(symbol, {})
            if not _is_valid_response(profile):
                logger.debug(f"yahooquery summary_profile invalid for {symbol}")
                profile = {}

            # Get price data
            price = ticker.price.get(symbol, {})
            if not _is_valid_response(price):
                price = {}

            # Get financial data (margins, ratios)
            fin_data = ticker.financial_data.get(symbol, {})
            if not _is_valid_response(fin_data):
                fin_data = {}

            # Get key stats (valuations)
            key_stats = ticker.key_stats.get(symbol, {})
            if not _is_valid_response(key_stats):
                key_stats = {}

            # Get asset profile (for additional details)
            asset_profile = ticker.asset_profile.get(symbol, {})
            if not _is_valid_response(asset_profile):
                asset_profile = {}

            # Build normalized info dict (matching yfinance structure)
            info = {
                "symbol": symbol.upper(),
                # Price data
                "currentPrice": _safe_float(price.get("regularMarketPrice")),
                "previousClose": _safe_float(price.get("regularMarketPreviousClose")),
                "open": _safe_float(price.get("regularMarketOpen")),
                "dayHigh": _safe_float(price.get("regularMarketDayHigh")),
                "dayLow": _safe_float(price.get("regularMarketDayLow")),
                "volume": _safe_int(price.get("regularMarketVolume")),
                "averageVolume": _safe_int(price.get("averageDailyVolume3Month")),
                "marketCap": _safe_int(price.get("marketCap")),
                # Identifiers
                "shortName": _safe_str(price.get("shortName")),
                "longName": _safe_str(price.get("longName")),
                "quoteType": _safe_str(price.get("quoteType")),
                "currency": _safe_str(price.get("currency")),
                "exchange": _safe_str(price.get("exchangeName")),
                # Sector/Industry
                "sector": _safe_str(profile.get("sector")) or _safe_str(asset_profile.get("sector")),
                "industry": _safe_str(profile.get("industry")) or _safe_str(asset_profile.get("industry")),
                "longBusinessSummary": _safe_str(profile.get("longBusinessSummary")),
                "website": _safe_str(profile.get("website")) or _safe_str(asset_profile.get("website")),
                "country": _safe_str(profile.get("country")) or _safe_str(asset_profile.get("country")),
                "fullTimeEmployees": _safe_int(profile.get("fullTimeEmployees")),
                # Financial metrics
                "trailingPE": _safe_float(key_stats.get("trailingPE")),
                "forwardPE": _safe_float(key_stats.get("forwardPE")),
                "pegRatio": _safe_float(key_stats.get("pegRatio")),
                "priceToBook": _safe_float(key_stats.get("priceToBook")),
                "priceToSales": _safe_float(key_stats.get("priceToSalesTrailing12Months")),
                "enterpriseValue": _safe_int(key_stats.get("enterpriseValue")),
                "enterpriseToRevenue": _safe_float(key_stats.get("enterpriseToRevenue")),
                "enterpriseToEbitda": _safe_float(key_stats.get("enterpriseToEbitda")),
                "beta": _safe_float(key_stats.get("beta")),
                "bookValue": _safe_float(key_stats.get("bookValue")),
                "fiftyTwoWeekHigh": _safe_float(key_stats.get("fiftyTwoWeekHigh")),
                "fiftyTwoWeekLow": _safe_float(key_stats.get("fiftyTwoWeekLow")),
                "fiftyDayAverage": _safe_float(key_stats.get("fiftyDayAverage")),
                "twoHundredDayAverage": _safe_float(key_stats.get("twoHundredDayAverage")),
                # Profitability
                "profitMargins": _safe_float(fin_data.get("profitMargins")),
                "operatingMargins": _safe_float(fin_data.get("operatingMargins")),
                "grossMargins": _safe_float(fin_data.get("grossMargins")),
                "returnOnAssets": _safe_float(fin_data.get("returnOnAssets")),
                "returnOnEquity": _safe_float(fin_data.get("returnOnEquity")),
                # Financial health
                "debtToEquity": _safe_float(fin_data.get("debtToEquity")),
                "currentRatio": _safe_float(fin_data.get("currentRatio")),
                "quickRatio": _safe_float(fin_data.get("quickRatio")),
                "totalCash": _safe_int(fin_data.get("totalCash")),
                "totalDebt": _safe_int(fin_data.get("totalDebt")),
                "totalRevenue": _safe_int(fin_data.get("totalRevenue")),
                "revenuePerShare": _safe_float(fin_data.get("revenuePerShare")),
                "freeCashflow": _safe_int(fin_data.get("freeCashflow")),
                "operatingCashflow": _safe_int(fin_data.get("operatingCashflow")),
                # Dividends
                "dividendRate": _safe_float(key_stats.get("dividendRate")),
                "dividendYield": _safe_float(key_stats.get("dividendYield")),
                "payoutRatio": _safe_float(key_stats.get("payoutRatio")),
                "exDividendDate": key_stats.get("exDividendDate"),
                # Growth
                "earningsGrowth": _safe_float(fin_data.get("earningsGrowth")),
                "revenueGrowth": _safe_float(fin_data.get("revenueGrowth")),
                "earningsQuarterlyGrowth": _safe_float(key_stats.get("earningsQuarterlyGrowth")),
                # Shares
                "sharesOutstanding": _safe_int(key_stats.get("sharesOutstanding")),
                "floatShares": _safe_int(key_stats.get("floatShares")),
                "heldPercentInsiders": _safe_float(key_stats.get("heldPercentInsiders")),
                "heldPercentInstitutions": _safe_float(key_stats.get("heldPercentInstitutions")),
                "shortRatio": _safe_float(key_stats.get("shortRatio")),
                "shortPercentOfFloat": _safe_float(key_stats.get("shortPercentOfFloat")),
                # EPS
                "trailingEps": _safe_float(key_stats.get("trailingEps")),
                "forwardEps": _safe_float(key_stats.get("forwardEps")),
                # Source marker
                "_source": "yahooquery",
                "_fetched_at": datetime.now(UTC).isoformat(),
            }

            # Remove None values for cleaner output
            info = {k: v for k, v in info.items() if v is not None}

            return info

        except Exception as e:
            logger.warning(f"yahooquery ticker_info failed for {symbol}: {e}")
            return None

    async def get_ticker_info(self, symbol: str) -> dict[str, Any] | None:
        """Get ticker info from yahooquery."""
        if not YAHOOQUERY_AVAILABLE:
            logger.debug("yahooquery not available")
            return None

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(_executor, self._fetch_ticker_info_sync, symbol)

    # =========================================================================
    # Financial Statements
    # =========================================================================

    def _fetch_financials_sync(self, symbol: str) -> dict[str, Any] | None:
        """
        Fetch financial statements from yahooquery (blocking).
        
        Returns quarterly and annual financial statements with richer data
        than yfinance, especially for financial sector companies.
        
        Key additional fields for domain analysis:
        - Banks: InterestIncome, InterestExpense, ProvisionForCreditLosses
        - REITs: RealEstateRevenue, DepreciationAndAmortization (for FFO)
        - Insurance: NetPolicyholderBenefits, NetInvestmentIncome
        """
        if not YAHOOQUERY_AVAILABLE:
            return None

        try:
            ticker = Ticker(symbol, asynchronous=False)
            symbol_upper = symbol.upper()

            result = {
                "symbol": symbol_upper,
                "quarterly": {},
                "annual": {},
                "fetched_at": datetime.now(UTC).isoformat(),
                "_source": "yahooquery",
            }

            def normalize_period_key(value: Any) -> str | None:
                if value is None:
                    return None
                if isinstance(value, datetime):
                    return value.date().isoformat()
                try:
                    return datetime.fromisoformat(str(value)).date().isoformat()
                except (ValueError, TypeError):
                    return str(value)

            def sort_period_keys(keys: list[str]) -> list[str]:
                def _key(item: str) -> datetime:
                    try:
                        return datetime.fromisoformat(item)
                    except (ValueError, TypeError):
                        return datetime.min
                return sorted(keys, key=_key, reverse=True)

            def select_symbol_df(df: pd.DataFrame) -> pd.DataFrame:
                if symbol_upper in df.index:
                    return df.loc[[symbol_upper]]
                if symbol.lower() in df.index:
                    return df.loc[[symbol.lower()]]
                return df

            # Helper to extract data from yahooquery DataFrame
            def extract_from_df(df: pd.DataFrame) -> dict[str, float]:
                """Extract most recent values from yahooquery DataFrame."""
                if df is None or isinstance(df, str) or df.empty:
                    return {}
                    
                extracted = {}
                
                try:
                    # yahooquery returns DataFrame with symbol as index
                    # and columns like asOfDate, periodType, plus financial metrics
                    symbol_df = select_symbol_df(df)
                    
                    # Get the most recent row (last row after sorting by date)
                    if 'asOfDate' in symbol_df.columns:
                        symbol_df = symbol_df.sort_values('asOfDate')
                    
                    if len(symbol_df) > 0:
                        latest = symbol_df.iloc[-1]
                        for col in symbol_df.columns:
                            # Skip metadata columns
                            if col in ('asOfDate', 'periodType', 'currencyCode'):
                                continue
                            val = latest[col]
                            if pd.notna(val):
                                try:
                                    extracted[str(col)] = float(val)
                                except (ValueError, TypeError):
                                    pass
                except Exception as e:
                    logger.debug(f"extract_from_df error: {e}")
                                
                return extracted

            def extract_series_from_df(df: pd.DataFrame, max_periods: int = 12) -> dict[str, dict[str, float]]:
                """Extract time series values keyed by period from yahooquery DataFrame."""
                if df is None or isinstance(df, str) or df.empty:
                    return {}
                series: dict[str, dict[str, float]] = {}
                try:
                    symbol_df = select_symbol_df(df)
                    if 'asOfDate' in symbol_df.columns:
                        symbol_df = symbol_df.sort_values('asOfDate')

                    for _, row in symbol_df.iterrows():
                        period_key = normalize_period_key(row.get('asOfDate'))
                        if not period_key:
                            continue
                        for col in symbol_df.columns:
                            if col in ('asOfDate', 'periodType', 'currencyCode'):
                                continue
                            val = row.get(col)
                            if pd.notna(val):
                                try:
                                    series.setdefault(str(col), {})[period_key] = float(val)
                                except (ValueError, TypeError):
                                    continue
                except Exception as e:
                    logger.debug(f"extract_series_from_df error: {e}")

                for metric, values in list(series.items()):
                    ordered = sort_period_keys(list(values.keys()))[:max_periods]
                    series[metric] = {key: values[key] for key in ordered}

                return series

            # Get quarterly financial statements
            try:
                q_income = ticker.income_statement(frequency='q')
                result["quarterly"]["income_statement"] = extract_from_df(q_income)
                result["quarterly"]["income_statement_series"] = extract_series_from_df(q_income)
            except Exception as e:
                logger.debug(f"yahooquery quarterly income failed for {symbol}: {e}")
                result["quarterly"]["income_statement"] = {}
                result["quarterly"]["income_statement_series"] = {}

            try:
                q_balance = ticker.balance_sheet(frequency='q')
                result["quarterly"]["balance_sheet"] = extract_from_df(q_balance)
                result["quarterly"]["balance_sheet_series"] = extract_series_from_df(q_balance)
            except Exception as e:
                logger.debug(f"yahooquery quarterly balance failed for {symbol}: {e}")
                result["quarterly"]["balance_sheet"] = {}
                result["quarterly"]["balance_sheet_series"] = {}

            try:
                q_cash = ticker.cash_flow(frequency='q')
                result["quarterly"]["cash_flow"] = extract_from_df(q_cash)
                result["quarterly"]["cash_flow_series"] = extract_series_from_df(q_cash)
            except Exception as e:
                logger.debug(f"yahooquery quarterly cash_flow failed for {symbol}: {e}")
                result["quarterly"]["cash_flow"] = {}
                result["quarterly"]["cash_flow_series"] = {}

            # Get annual financial statements  
            try:
                a_income = ticker.income_statement(frequency='a')
                result["annual"]["income_statement"] = extract_from_df(a_income)
                result["annual"]["income_statement_series"] = extract_series_from_df(a_income, max_periods=8)
            except Exception as e:
                logger.debug(f"yahooquery annual income failed for {symbol}: {e}")
                result["annual"]["income_statement"] = {}
                result["annual"]["income_statement_series"] = {}

            try:
                a_balance = ticker.balance_sheet(frequency='a')
                result["annual"]["balance_sheet"] = extract_from_df(a_balance)
                result["annual"]["balance_sheet_series"] = extract_series_from_df(a_balance, max_periods=8)
            except Exception as e:
                logger.debug(f"yahooquery annual balance failed for {symbol}: {e}")
                result["annual"]["balance_sheet"] = {}
                result["annual"]["balance_sheet_series"] = {}

            try:
                a_cash = ticker.cash_flow(frequency='a')
                result["annual"]["cash_flow"] = extract_from_df(a_cash)
                result["annual"]["cash_flow_series"] = extract_series_from_df(a_cash, max_periods=8)
            except Exception as e:
                logger.debug(f"yahooquery annual cash_flow failed for {symbol}: {e}")
                result["annual"]["cash_flow"] = {}
                result["annual"]["cash_flow_series"] = {}

            # Log what we found
            q_income_count = len(result["quarterly"]["income_statement"])
            q_balance_count = len(result["quarterly"]["balance_sheet"])
            q_cash_count = len(result["quarterly"]["cash_flow"])
            logger.debug(
                f"yahooquery financials for {symbol}: quarterly income={q_income_count}, "
                f"balance={q_balance_count}, cash_flow={q_cash_count}"
            )

            return result

        except Exception as e:
            logger.warning(f"yahooquery financials failed for {symbol}: {e}")
            return None

    async def get_financials(self, symbol: str) -> dict[str, Any] | None:
        """Get financial statements from yahooquery."""
        if not YAHOOQUERY_AVAILABLE:
            return None

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(_executor, self._fetch_financials_sync, symbol)

    # =========================================================================
    # Valuation Measures (unique to yahooquery)
    # =========================================================================

    def _fetch_valuation_measures_sync(self, symbol: str) -> dict[str, Any] | None:
        """
        Fetch valuation measures from yahooquery (blocking).
        
        Provides time-series valuation data that yfinance doesn't have:
        - EnterpriseValue, MarketCap
        - PeRatio, ForwardPeRatio, PegRatio
        - PsRatio, PbRatio
        - EnterpriseValueRevenueMultiple, EnterpriseValueEbitdaMultiple
        """
        if not YAHOOQUERY_AVAILABLE:
            return None

        try:
            ticker = Ticker(symbol, asynchronous=False)
            
            val_measures = ticker.valuation_measures
            
            if val_measures is None or isinstance(val_measures, str):
                return None
                
            if val_measures.empty:
                return None

            # Get most recent valuation data
            if symbol in val_measures.index.get_level_values(0):
                symbol_df = val_measures.loc[symbol]
            else:
                symbol_df = val_measures

            result = {
                "symbol": symbol.upper(),
                "fetched_at": datetime.now(UTC).isoformat(),
                "_source": "yahooquery",
            }

            if len(symbol_df) > 0:
                latest = symbol_df.iloc[-1] if isinstance(symbol_df, pd.DataFrame) else symbol_df
                for col in latest.index if isinstance(latest, pd.Series) else symbol_df.columns:
                    val = latest[col] if isinstance(latest, pd.Series) else latest
                    if pd.notna(val):
                        try:
                            result[str(col)] = float(val)
                        except (ValueError, TypeError):
                            pass

            return result

        except Exception as e:
            logger.warning(f"yahooquery valuation_measures failed for {symbol}: {e}")
            return None

    async def get_valuation_measures(self, symbol: str) -> dict[str, Any] | None:
        """Get valuation measures from yahooquery."""
        if not YAHOOQUERY_AVAILABLE:
            return None

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(_executor, self._fetch_valuation_measures_sync, symbol)

    # =========================================================================
    # Price History
    # =========================================================================

    def _fetch_price_history_sync(
        self,
        symbol: str,
        period: str = "1y",
        interval: str = "1d",
    ) -> pd.DataFrame | None:
        """
        Fetch price history from yahooquery (blocking).
        
        Args:
            symbol: Stock ticker symbol
            period: Time period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
            interval: Data interval (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)
        
        Returns:
            DataFrame with date index and columns: Open, High, Low, Close, Volume, Adjclose
        """
        if not YAHOOQUERY_AVAILABLE:
            return None

        try:
            ticker = Ticker(symbol, asynchronous=False)
            
            df = ticker.history(period=period, interval=interval)
            
            if df is None or isinstance(df, str) or df.empty:
                return None

            # yahooquery returns multi-index (symbol, date), flatten it
            if isinstance(df.index, pd.MultiIndex):
                if symbol.upper() in df.index.get_level_values(0):
                    df = df.loc[symbol.upper()]
                else:
                    # Try lowercase
                    df = df.loc[symbol.lower()] if symbol.lower() in df.index.get_level_values(0) else df

            # Standardize column names to match yfinance
            column_mapping = {
                "open": "Open",
                "high": "High", 
                "low": "Low",
                "close": "Close",
                "volume": "Volume",
                "adjclose": "Adj Close",
            }
            df = df.rename(columns=column_mapping)

            return df

        except Exception as e:
            logger.warning(f"yahooquery price history failed for {symbol}: {e}")
            return None

    async def get_price_history(
        self,
        symbol: str,
        period: str = "1y",
        interval: str = "1d",
    ) -> pd.DataFrame | None:
        """Get price history from yahooquery."""
        if not YAHOOQUERY_AVAILABLE:
            return None

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            _executor,
            self._fetch_price_history_sync,
            symbol,
            period,
            interval,
        )

    # =========================================================================
    # Domain-Specific Helpers
    # =========================================================================

    def _fetch_domain_specific_data_sync(
        self,
        symbol: str,
        sector: str | None = None,
    ) -> dict[str, Any] | None:
        """
        Fetch domain-specific financial data from yahooquery.
        
        Extracts metrics relevant to specific sectors:
        - Banks: Net interest margin, loan loss provisions, tier 1 capital
        - REITs: FFO-related metrics, occupancy data
        - Insurance: Combined ratio components, investment income
        - Tech: R&D spending, SBC
        """
        if not YAHOOQUERY_AVAILABLE:
            return None

        try:
            ticker = Ticker(symbol, asynchronous=False)
            
            result = {
                "symbol": symbol.upper(),
                "sector": sector,
                "fetched_at": datetime.now(UTC).isoformat(),
                "_source": "yahooquery",
            }

            # Get latest quarterly financials
            try:
                q_income = ticker.income_statement(frequency='q', trailing=False)
                q_balance = ticker.balance_sheet(frequency='q', trailing=False)
                q_cash = ticker.cash_flow(frequency='q', trailing=False)
            except Exception:
                return None

            def get_latest_value(df: pd.DataFrame, column: str) -> float | None:
                """Get most recent value from a column."""
                if df is None or isinstance(df, str) or df.empty:
                    return None
                try:
                    if symbol.upper() in df.index.get_level_values(0):
                        symbol_df = df.loc[symbol.upper()]
                    else:
                        symbol_df = df
                    if column in symbol_df.columns:
                        val = symbol_df[column].iloc[-1]
                        return float(val) if pd.notna(val) else None
                except (KeyError, IndexError):
                    pass
                return None

            # Bank-specific metrics
            if sector and "financial" in sector.lower():
                result["interest_income"] = get_latest_value(q_income, "InterestIncome")
                result["interest_expense"] = get_latest_value(q_income, "InterestExpense")
                result["net_interest_income"] = get_latest_value(q_income, "NetInterestIncome")
                result["provision_for_credit_losses"] = get_latest_value(q_income, "ProvisionForCreditLosses")
                result["total_deposits"] = get_latest_value(q_balance, "Deposits")
                result["total_loans"] = get_latest_value(q_balance, "GrossLoans")

            # REIT-specific metrics
            if sector and "real estate" in sector.lower():
                result["depreciation"] = get_latest_value(q_cash, "DepreciationAndAmortization")
                result["real_estate_revenue"] = get_latest_value(q_income, "RealEstateRevenue")
                
            # Insurance-specific metrics
            if sector and "insurance" in sector.lower():
                result["net_investment_income"] = get_latest_value(q_income, "NetInvestmentIncome")
                result["policyholder_benefits"] = get_latest_value(q_income, "NetPolicyholderBenefitsAndClaims")

            # Tech-specific metrics
            if sector and "technology" in sector.lower():
                result["research_development"] = get_latest_value(q_income, "ResearchAndDevelopment")
                result["stock_based_compensation"] = get_latest_value(q_cash, "StockBasedCompensation")

            # Remove None values
            result = {k: v for k, v in result.items() if v is not None}

            return result

        except Exception as e:
            logger.warning(f"yahooquery domain_specific_data failed for {symbol}: {e}")
            return None

    async def get_domain_specific_data(
        self,
        symbol: str,
        sector: str | None = None,
    ) -> dict[str, Any] | None:
        """Get domain-specific financial data from yahooquery."""
        if not YAHOOQUERY_AVAILABLE:
            return None

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            _executor,
            self._fetch_domain_specific_data_sync,
            symbol,
            sector,
        )

    # =========================================================================
    # Batch Operations
    # =========================================================================

    def _fetch_batch_ticker_info_sync(
        self,
        symbols: list[str],
    ) -> dict[str, dict[str, Any]]:
        """
        Fetch ticker info for multiple symbols at once (blocking).
        
        yahooquery supports multi-symbol queries for efficiency.
        """
        if not YAHOOQUERY_AVAILABLE or not symbols:
            return {}

        try:
            ticker = Ticker(symbols, asynchronous=False)
            
            results = {}
            
            # Fetch all data types
            profiles = ticker.summary_profile
            prices = ticker.price
            fin_data = ticker.financial_data
            key_stats = ticker.key_stats
            
            for symbol in symbols:
                symbol_upper = symbol.upper()
                
                profile = profiles.get(symbol_upper, {})
                price = prices.get(symbol_upper, {})
                fin = fin_data.get(symbol_upper, {})
                stats = key_stats.get(symbol_upper, {})
                
                if not _is_valid_response(price):
                    continue

                # Build info dict (simplified for batch)
                info = {
                    "symbol": symbol_upper,
                    "currentPrice": _safe_float(price.get("regularMarketPrice")),
                    "marketCap": _safe_int(price.get("marketCap")),
                    "shortName": _safe_str(price.get("shortName")),
                    "sector": _safe_str(profile.get("sector")) if _is_valid_response(profile) else None,
                    "industry": _safe_str(profile.get("industry")) if _is_valid_response(profile) else None,
                    "returnOnEquity": _safe_float(fin.get("returnOnEquity")) if _is_valid_response(fin) else None,
                    "profitMargins": _safe_float(fin.get("profitMargins")) if _is_valid_response(fin) else None,
                    "forwardPE": _safe_float(stats.get("forwardPE")) if _is_valid_response(stats) else None,
                    "_source": "yahooquery",
                }
                
                # Remove None values
                info = {k: v for k, v in info.items() if v is not None}
                results[symbol_upper] = info

            return results

        except Exception as e:
            logger.warning(f"yahooquery batch ticker_info failed: {e}")
            return {}

    async def get_batch_ticker_info(
        self,
        symbols: list[str],
    ) -> dict[str, dict[str, Any]]:
        """Get ticker info for multiple symbols at once."""
        if not YAHOOQUERY_AVAILABLE or not symbols:
            return {}

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(_executor, self._fetch_batch_ticker_info_sync, symbols)


# Singleton instance
_yahooquery_service: YahooQueryService | None = None


def get_yahooquery_service() -> YahooQueryService:
    """Get singleton instance of YahooQueryService."""
    global _yahooquery_service
    if _yahooquery_service is None:
        _yahooquery_service = YahooQueryService()
    return _yahooquery_service
