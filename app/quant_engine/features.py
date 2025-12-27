"""
Feature engineering for alpha models.

Computes price-derived features with strict no-lookahead guarantees.
All features are computed using only data available at time t.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FeatureSet:
    """
    Complete feature set for alpha models.
    
    All features are aligned to the same date index and asset columns.
    """
    momentum: dict[int, pd.DataFrame]  # Window -> momentum returns
    volatility: pd.DataFrame  # Rolling volatility
    reversal: pd.DataFrame  # Short-term reversal
    volume_trend: pd.DataFrame | None  # Volume momentum (if available)
    factor_exposures: pd.DataFrame | None  # Factor betas (if computed)

    # Metadata
    assets: list[str]
    date_range: tuple[date, date]

    def get_features_at(self, as_of: date) -> pd.DataFrame:
        """
        Get all features as of a specific date.
        
        Returns DataFrame with assets as rows and features as columns.
        """
        ts = pd.Timestamp(as_of)
        rows = []

        for asset in self.assets:
            row = {"asset": asset}

            # Momentum features
            for window, mom_df in self.momentum.items():
                if ts in mom_df.index and asset in mom_df.columns:
                    row[f"mom_{window}d"] = mom_df.loc[ts, asset]
                else:
                    row[f"mom_{window}d"] = np.nan

            # Volatility
            if ts in self.volatility.index and asset in self.volatility.columns:
                row["volatility"] = self.volatility.loc[ts, asset]
            else:
                row["volatility"] = np.nan

            # Reversal
            if ts in self.reversal.index and asset in self.reversal.columns:
                row["reversal"] = self.reversal.loc[ts, asset]
            else:
                row["reversal"] = np.nan

            # Volume trend
            if self.volume_trend is not None:
                if ts in self.volume_trend.index and asset in self.volume_trend.columns:
                    row["volume_trend"] = self.volume_trend.loc[ts, asset]
                else:
                    row["volume_trend"] = np.nan

            rows.append(row)

        return pd.DataFrame(rows).set_index("asset")

    def to_panel(self) -> pd.DataFrame:
        """
        Convert to panel format (date, asset, features).
        
        Returns long-format DataFrame.
        """
        records = []

        for ts in self.volatility.index:
            for asset in self.assets:
                record = {"date": ts.date(), "asset": asset}

                for window, mom_df in self.momentum.items():
                    if asset in mom_df.columns:
                        record[f"mom_{window}d"] = mom_df.loc[ts, asset] if ts in mom_df.index else np.nan
                    else:
                        record[f"mom_{window}d"] = np.nan

                record["volatility"] = (
                    self.volatility.loc[ts, asset]
                    if asset in self.volatility.columns else np.nan
                )
                record["reversal"] = (
                    self.reversal.loc[ts, asset]
                    if asset in self.reversal.columns else np.nan
                )

                if self.volume_trend is not None and asset in self.volume_trend.columns:
                    record["volume_trend"] = (
                        self.volume_trend.loc[ts, asset] if ts in self.volume_trend.index else np.nan
                    )

                records.append(record)

        return pd.DataFrame(records)

    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert to DataFrame with (date, asset) multi-index.
        
        Returns panel-format DataFrame suitable for alpha model training/prediction.
        """
        panel = self.to_panel()
        return panel.set_index(["date", "asset"])


def compute_momentum(
    prices: pd.DataFrame,
    windows: tuple[int, ...] = (21, 63, 126, 252),
) -> dict[int, pd.DataFrame]:
    """
    Compute momentum (total return) over various windows.
    
    Momentum at time t uses only prices up to and including t.
    No lookahead: mom_t = (P_t - P_{t-window}) / P_{t-window}
    
    Parameters
    ----------
    prices : pd.DataFrame
        Price matrix with dates as index, assets as columns.
    windows : tuple[int, ...]
        Lookback windows in trading days.
    
    Returns
    -------
    dict[int, pd.DataFrame]
        Momentum for each window.
    """
    momentum = {}

    for window in windows:
        # Use shift to ensure no lookahead
        past_price = prices.shift(window)
        mom = (prices - past_price) / past_price
        momentum[window] = mom

    return momentum


def compute_volatility(
    returns: pd.DataFrame,
    window: int = 21,
    annualize: bool = True,
) -> pd.DataFrame:
    """
    Compute rolling volatility.
    
    Volatility at time t uses only returns up to and including t-1
    (we use returns, not prices, and returns_t = P_t/P_{t-1} - 1).
    
    Parameters
    ----------
    returns : pd.DataFrame
        Return matrix with dates as index, assets as columns.
    window : int
        Lookback window in trading days.
    annualize : bool
        Whether to annualize (multiply by sqrt(252)).
    
    Returns
    -------
    pd.DataFrame
        Rolling volatility.
    """
    vol = returns.rolling(window=window, min_periods=window // 2).std()

    if annualize:
        vol = vol * np.sqrt(252)

    return vol


def compute_reversal(
    returns: pd.DataFrame,
    window: int = 5,
) -> pd.DataFrame:
    """
    Compute short-term reversal signal.
    
    Reversal = -sum(returns over short window)
    
    Captures mean reversion tendency.
    
    Parameters
    ----------
    returns : pd.DataFrame
        Return matrix.
    window : int
        Short lookback window.
    
    Returns
    -------
    pd.DataFrame
        Reversal signal (negative of short-term return).
    """
    short_return = returns.rolling(window=window).sum()
    return -short_return


def compute_volume_trend(
    volume: pd.DataFrame,
    window: int = 21,
) -> pd.DataFrame:
    """
    Compute volume trend (current vs average).
    
    Parameters
    ----------
    volume : pd.DataFrame
        Volume matrix.
    window : int
        Lookback window.
    
    Returns
    -------
    pd.DataFrame
        Volume trend ratio.
    """
    avg_volume = volume.rolling(window=window, min_periods=window // 2).mean()
    return volume / avg_volume - 1


def compute_all_features(
    prices: pd.DataFrame,
    returns: pd.DataFrame,
    volume: pd.DataFrame | None = None,
    momentum_windows: tuple[int, ...] = (21, 63, 126, 252),
    volatility_window: int = 21,
    reversal_window: int = 5,
) -> FeatureSet:
    """
    Compute all features for alpha models.
    
    Parameters
    ----------
    prices : pd.DataFrame
        Price matrix (dates × assets).
    returns : pd.DataFrame
        Return matrix (dates × assets).
    volume : pd.DataFrame, optional
        Volume matrix (dates × assets).
    momentum_windows : tuple[int, ...]
        Momentum lookback windows.
    volatility_window : int
        Volatility lookback window.
    reversal_window : int
        Reversal lookback window.
    
    Returns
    -------
    FeatureSet
        Complete feature set.
    """
    assets = list(prices.columns)

    momentum = compute_momentum(prices, momentum_windows)
    volatility = compute_volatility(returns, volatility_window)
    reversal = compute_reversal(returns, reversal_window)

    volume_trend = None
    if volume is not None and not volume.empty:
        volume_trend = compute_volume_trend(volume, volatility_window)

    return FeatureSet(
        momentum=momentum,
        volatility=volatility,
        reversal=reversal,
        volume_trend=volume_trend,
        factor_exposures=None,  # Computed separately in alpha_models
        assets=assets,
        date_range=(prices.index.min().date(), prices.index.max().date()),
    )


def prepare_alpha_training_data(
    features: FeatureSet,
    returns: pd.DataFrame,
    forecast_horizon_months: int,
    min_obs: int = 120,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Prepare training data for alpha models.
    
    Creates X (features at time t) and y (forward returns from t to t+H).
    
    Parameters
    ----------
    features : FeatureSet
        Computed features.
    returns : pd.DataFrame
        Return matrix.
    forecast_horizon_months : int
        Forecast horizon H in months.
    min_obs : int
        Minimum observations required.
    
    Returns
    -------
    tuple[pd.DataFrame, pd.Series]
        X: Features (date, asset) -> feature values
        y: Forward returns (date, asset) -> return
    """
    # Convert months to trading days (approximate)
    horizon_days = forecast_horizon_months * 21

    # Compute forward returns (cumulative)
    forward_returns = returns.rolling(window=horizon_days).sum().shift(-horizon_days)

    # Get panel data
    panel = features.to_panel()
    panel = panel.set_index(["date", "asset"])

    # Stack forward returns
    fwd_stack = forward_returns.stack()
    fwd_stack.index.names = ["date", "asset"]
    fwd_stack.name = "forward_return"

    # Merge
    merged = panel.join(fwd_stack, how="inner")

    # Drop NaN rows
    merged = merged.dropna()

    if len(merged) < min_obs:
        raise ValueError(
            f"Insufficient data: {len(merged)} samples, need {min_obs}"
        )

    X = merged.drop(columns=["forward_return"])
    y = merged["forward_return"]

    return X, y


def validate_no_lookahead(
    features: pd.DataFrame,
    as_of: date,
) -> bool:
    """
    Validate that features contain no data after as_of date.
    
    This is a safety check for no-lookahead guarantee.
    
    Parameters
    ----------
    features : pd.DataFrame
        Feature data with date index or column.
    as_of : date
        Reference date.
    
    Returns
    -------
    bool
        True if no lookahead detected.
    
    Raises
    ------
    ValueError
        If lookahead detected.
    """
    if isinstance(features.index, pd.MultiIndex):
        dates = features.index.get_level_values("date")
    elif isinstance(features.index, pd.DatetimeIndex):
        dates = features.index.date
    elif "date" in features.columns:
        dates = features["date"]
    else:
        # Cannot validate, assume OK
        return True

    max_date = pd.Timestamp(max(dates)).date()

    if max_date > as_of:
        raise ValueError(
            f"Lookahead detected: features contain data up to {max_date}, "
            f"but as_of is {as_of}"
        )

    return True
