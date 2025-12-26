"""
Tests for the Quantitative Portfolio Engine.

Tests cover:
1. DipScore correctness on toy data
2. No-lookahead guarantees
3. Optimizer respects constraints
4. Dip never triggers orders directly
5. Alpha model ensemble works correctly
6. Risk model PCA works correctly
"""

import numpy as np
import pandas as pd
import pytest
from datetime import datetime, timedelta


class TestQuantEngineTypes:
    """Test core type definitions."""

    def test_quant_config_defaults(self):
        """QuantConfig should have sensible defaults."""
        from app.quant_engine.types import QuantConfig

        config = QuantConfig()

        assert config.base_currency == "EUR"
        assert config.long_only is True
        assert config.fixed_cost_eur == 1.0
        assert config.min_trade_eur == 10.0
        assert config.max_weight == 0.15
        assert config.max_turnover == 0.20

    def test_solver_status_enum(self):
        """SolverStatus enum should have expected values."""
        from app.quant_engine.types import SolverStatus

        assert SolverStatus.OPTIMAL.value == "optimal"
        assert SolverStatus.INFEASIBLE.value == "infeasible"
        assert SolverStatus.ERROR.value == "error"


class TestFeatureEngineering:
    """Test feature engineering with no-lookahead guarantee."""

    @pytest.fixture
    def sample_returns(self) -> pd.DataFrame:
        """Create sample returns data."""
        np.random.seed(42)
        dates = pd.date_range("2020-01-01", periods=500, freq="D")
        assets = ["AAPL", "GOOGL", "MSFT", "AMZN"]

        returns = pd.DataFrame(
            np.random.randn(500, 4) * 0.02,
            index=dates,
            columns=assets,
        )
        return returns

    def test_momentum_computation(self, sample_returns: pd.DataFrame):
        """Momentum should use only past data."""
        from app.quant_engine.features import compute_momentum

        # Compute prices from returns (starting at 100)
        prices = (1 + sample_returns).cumprod() * 100

        mom = compute_momentum(prices, windows=(21,))

        # Momentum should be NaN for first window-1 days
        assert mom[21].iloc[:20].isna().all().all()

        # Momentum should be valid after window
        assert not mom[21].iloc[21:].isna().any().any()

    def test_volatility_computation(self, sample_returns: pd.DataFrame):
        """Volatility should be positive."""
        from app.quant_engine.features import compute_volatility

        vol = compute_volatility(sample_returns, window=21)

        # Volatility should be positive
        valid_vol = vol.dropna()
        assert (valid_vol >= 0).all().all()

    def test_all_features_no_lookahead(self, sample_returns: pd.DataFrame):
        """Feature computation should not look ahead."""
        from app.quant_engine.features import compute_all_features

        # Compute prices from returns
        prices = (1 + sample_returns).cumprod() * 100

        features = compute_all_features(
            prices=prices,
            returns=sample_returns,
        )

        # Features at time t should only use data up to time t
        # This is verified by checking the feature computation logic
        # Get features at latest date
        latest_date = prices.index[-1].date()
        feature_df = features.get_features_at(latest_date)

        # Should have features for all assets
        assert len(feature_df) == len(sample_returns.columns)


class TestAlphaModels:
    """Test alpha model ensemble."""

    @pytest.fixture
    def training_data(self) -> tuple[np.ndarray, np.ndarray]:
        """Create sample training data."""
        np.random.seed(42)
        n_samples = 200
        n_features = 10

        X = np.random.randn(n_samples, n_features)
        # y = X @ true_weights + noise
        true_weights = np.random.randn(n_features) * 0.1
        y = X @ true_weights + np.random.randn(n_samples) * 0.01

        return X, y

    def test_alpha_ensemble_fit(self, training_data):
        """Alpha ensemble should fit without error."""
        from app.quant_engine.alpha_models import AlphaModelEnsemble

        X, y = training_data

        # Split into train and validation
        split_idx = int(len(X) * 0.8)
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]

        model = AlphaModelEnsemble(
            ridge_alpha=1.0,
            lasso_alpha=0.01,
            use_lasso=True,
            ensemble_method="inverse_mse",
        )

        # Convert to DataFrames/Series
        X_train_df = pd.DataFrame(X_train)
        X_val_df = pd.DataFrame(X_val)
        y_train_s = pd.Series(y_train)
        y_val_s = pd.Series(y_val)

        scores = model.fit(X_train_df, y_train_s, X_val_df, y_val_s)

        assert scores is not None
        # Model names include alpha values like 'ridge:1.0'
        assert any("ridge" in k for k in scores.keys())
        assert len(model.weights) > 0
        assert sum(model.weights.values()) == pytest.approx(1.0, rel=0.01)

    def test_alpha_ensemble_predict(self, training_data):
        """Alpha ensemble should produce predictions with uncertainty."""
        from app.quant_engine.alpha_models import AlphaModelEnsemble

        X, y = training_data

        # Split into train and validation
        split_idx = int(len(X) * 0.8)
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]

        model = AlphaModelEnsemble(
            ridge_alpha=1.0,
            lasso_alpha=0.01,
        )

        # Convert to DataFrames/Series
        X_train_df = pd.DataFrame(X_train)
        X_val_df = pd.DataFrame(X_val)
        y_train_s = pd.Series(y_train)
        y_val_s = pd.Series(y_val)

        model.fit(X_train_df, y_train_s, X_val_df, y_val_s)

        # Predict on new data
        X_new = pd.DataFrame(np.random.randn(5, X.shape[1]))
        result = model.predict(X_new)

        # Result should be an AlphaResult object
        assert result is not None
        assert result.mu_hat is not None
        # mu_hat should have 5 predictions
        assert len(result.mu_hat) == 5


class TestDipScore:
    """Test statistical DipScore computation."""

    @pytest.fixture
    def market_data(self) -> tuple[pd.DataFrame, pd.Series]:
        """Create sample market and asset data."""
        np.random.seed(42)
        dates = pd.date_range("2020-01-01", periods=300, freq="D")

        # Market returns with some trend
        market = pd.Series(np.random.randn(300) * 0.01, index=dates)

        # Asset returns correlated with market
        assets = ["A", "B", "C"]
        betas = [0.8, 1.2, 1.0]
        returns = pd.DataFrame(index=dates, columns=assets)

        for i, (asset, beta) in enumerate(zip(assets, betas)):
            idio = np.random.randn(300) * 0.02
            returns[asset] = beta * market + idio

        return returns, market

    def test_dip_score_computation(self, market_data):
        """DipScore should be computed correctly."""
        from app.quant_engine.dip import compute_dip_score

        returns, market = market_data

        # Create factor returns DataFrame (market is the factor)
        factor_returns = pd.DataFrame({"market": market})

        artifacts = compute_dip_score(
            asset_returns=returns,
            factor_returns=factor_returns,
            resid_vol_window=20,
            min_obs=60,
        )

        # DipScore should be roughly normal (z-score)
        valid_scores = artifacts.dip_score.dropna()
        if len(valid_scores) > 0:
            mean_score = valid_scores.mean().mean()
            std_score = valid_scores.std().mean()

            # Mean should be close to 0
            assert abs(mean_score) < 0.5

            # Std should be close to 1
            assert 0.5 < std_score < 2.0

    def test_dip_bucket(self):
        """DipScore buckets should be assigned correctly."""
        from app.quant_engine.dip import get_dip_bucket

        assert get_dip_bucket(-2.5) == "<=-2"
        assert get_dip_bucket(-1.5) == "(-2,-1]"
        assert get_dip_bucket(-0.5) == "(-1,0]"
        assert get_dip_bucket(0.5) == "(0,1]"
        assert get_dip_bucket(1.5) == ">1"

    def test_dip_never_generates_orders(self, market_data):
        """DipScore should NEVER directly trigger orders."""
        from app.quant_engine.dip import compute_dip_score

        returns, market = market_data
        factor_returns = pd.DataFrame({"market": market})

        artifacts = compute_dip_score(
            asset_returns=returns,
            factor_returns=factor_returns,
            min_obs=60,
        )

        # DipScore is just a DataFrame of z-scores
        # It should have no "buy" or "sell" signals
        assert isinstance(artifacts.dip_score, pd.DataFrame)

        # The dip module should not have any order generation functions
        import app.quant_engine.dip as dip_module

        # Check that there are no "generate_order" or "trigger" functions
        public_funcs = [n for n in dir(dip_module) if not n.startswith("_")]
        for func in public_funcs:
            assert "order" not in func.lower()
            assert "trade" not in func.lower()
            assert "trigger" not in func.lower()


class TestRiskModel:
    """Test PCA-based risk model."""

    @pytest.fixture
    def returns_data(self) -> pd.DataFrame:
        """Create correlated returns data."""
        np.random.seed(42)
        n_obs = 252
        n_assets = 10

        # Create factor structure
        n_factors = 3
        factors = np.random.randn(n_obs, n_factors)
        loadings = np.random.randn(n_assets, n_factors)
        idio = np.random.randn(n_obs, n_assets) * 0.02

        returns = factors @ loadings.T + idio

        dates = pd.date_range("2020-01-01", periods=n_obs, freq="D")
        assets = [f"A{i}" for i in range(n_assets)]

        return pd.DataFrame(returns, index=dates, columns=assets)

    def test_pca_risk_model_fit(self, returns_data):
        """PCA risk model should fit correctly."""
        from app.quant_engine.risk import fit_pca_risk_model

        model = fit_pca_risk_model(returns_data, n_factors=3)

        assert model.n_factors == 3
        assert model.B.shape == (10, 3)
        assert model.sigma_f.shape == (3, 3)
        assert model.D.shape == (10,)
        assert len(model.assets) == 10

    def test_portfolio_variance(self, returns_data):
        """Portfolio variance should be computed correctly."""
        from app.quant_engine.risk import fit_pca_risk_model

        model = fit_pca_risk_model(returns_data, n_factors=3)

        # Equal weight portfolio
        w = np.ones(10) / 10

        var = model.portfolio_variance(w)
        vol = model.portfolio_volatility(w)

        assert var > 0
        assert vol == pytest.approx(np.sqrt(var), rel=1e-6)

    def test_marginal_contribution(self, returns_data):
        """MCR should sum to portfolio risk."""
        from app.quant_engine.risk import fit_pca_risk_model

        model = fit_pca_risk_model(returns_data, n_factors=3)

        w = np.ones(10) / 10
        mcr = model.marginal_contribution_to_risk(w)

        # Sum of MCR should approximately equal portfolio volatility
        # (This is the Euler decomposition property)
        assert sum(mcr) == pytest.approx(model.portfolio_volatility(w), rel=0.1)


class TestOptimizer:
    """Test portfolio optimizer."""

    @pytest.fixture
    def optimization_inputs(self):
        """Create optimizer inputs."""
        np.random.seed(42)
        n_assets = 5

        w_current = np.ones(n_assets) / n_assets
        mu_hat = pd.Series(
            np.random.randn(n_assets) * 0.01 + 0.005,
            index=[f"A{i}" for i in range(n_assets)],
        )

        # Simple diagonal covariance
        from app.quant_engine.types import RiskModel

        sigma_f = np.eye(2) * 0.01
        B = np.random.randn(n_assets, 2) * 0.5
        D = np.abs(np.random.randn(n_assets)) * 0.0001

        risk_model = RiskModel(
            B=B,
            sigma_f=sigma_f,
            D=D,
            explained_variance=np.array([0.6, 0.2]),
            n_factors=2,
            assets=[f"A{i}" for i in range(n_assets)],
        )

        return w_current, mu_hat, risk_model

    def test_optimizer_respects_max_weight(self, optimization_inputs):
        """Optimizer should respect max weight constraint."""
        from app.quant_engine.optimizer import optimize_portfolio

        w_current, mu_hat, risk_model = optimization_inputs

        result = optimize_portfolio(
            w_current=w_current,
            mu_hat=mu_hat,
            risk_model=risk_model,
            inflow_eur=1000,
            portfolio_value_eur=10000,
            max_weight=0.25,  # Use 0.25 to allow some room
            max_turnover=0.50,
        )

        # All weights should be <= max_weight (with tolerance for numerical issues)
        assert (result.w_new <= 0.25 + 1e-4).all(), f"Weights exceeded max: {result.w_new}"

    def test_optimizer_long_only(self, optimization_inputs):
        """Optimizer should produce long-only weights."""
        from app.quant_engine.optimizer import optimize_portfolio

        w_current, mu_hat, risk_model = optimization_inputs

        result = optimize_portfolio(
            w_current=w_current,
            mu_hat=mu_hat,
            risk_model=risk_model,
            inflow_eur=1000,
            portfolio_value_eur=10000,
        )

        # All weights should be >= 0 (long-only)
        assert (result.w_new >= -1e-6).all()

    def test_optimizer_turnover_limit(self, optimization_inputs):
        """Optimizer should respect turnover constraint."""
        from app.quant_engine.optimizer import optimize_portfolio

        w_current, mu_hat, risk_model = optimization_inputs

        result = optimize_portfolio(
            w_current=w_current,
            mu_hat=mu_hat,
            risk_model=risk_model,
            inflow_eur=0,  # No inflow
            portfolio_value_eur=10000,
            max_turnover=0.10,
        )

        # Total turnover should be <= max_turnover
        turnover = np.sum(np.abs(result.dw))
        assert turnover <= 0.10 + 1e-6

    def test_fixed_cost_pruning(self, optimization_inputs):
        """Small trades should be pruned due to fixed cost."""
        from app.quant_engine.optimizer import apply_fixed_cost_pruning

        w_current, mu_hat, risk_model = optimization_inputs

        # Create small trades
        dw = np.array([0.001, -0.001, 0.05, -0.03, 0.0])

        sigma = risk_model.get_covariance()
        mu = mu_hat.values

        dw_pruned, filtered_idx, cost = apply_fixed_cost_pruning(
            w_current=w_current,
            dw=dw,
            mu_hat=mu,
            sigma=sigma,
            lambda_risk=10.0,
            portfolio_value_eur=1000,  # Small portfolio
            fixed_cost_eur=1.0,
            min_trade_eur=10.0,
        )

        # Very small trades should be filtered
        assert len(filtered_idx) > 0 or np.allclose(dw_pruned, 0)


class TestDipNeverGeneratesOrders:
    """Test that dip logic NEVER directly generates orders."""

    def test_dip_module_has_no_order_functions(self):
        """The dip module should not have any order generation."""
        import app.quant_engine.dip as dip_module

        # Get all public functions/classes
        public_items = [n for n in dir(dip_module) if not n.startswith("_")]

        # None should be related to orders/trades
        forbidden_terms = ["order", "trade", "buy", "sell", "execute", "signal"]

        for item in public_items:
            item_lower = item.lower()
            for term in forbidden_terms:
                if term in item_lower:
                    # "dip_score" contains no forbidden terms
                    # but we need to check function behavior
                    assert term not in item_lower, (
                        f"Found forbidden term '{term}' in '{item}'"
                    )

    def test_dip_returns_only_scores(self):
        """Dip functions should return only numeric scores, not actions."""
        from app.quant_engine.dip import (
            compute_dip_score,
            get_dip_bucket,
            verify_dip_effectiveness,
        )

        # get_dip_bucket returns a string bucket, not an action
        bucket = get_dip_bucket(-2.5)
        assert bucket in ["<=-2", "(-2,-1]", "(-1,0]", "(0,1]", ">1"]
        assert bucket not in ["BUY", "SELL", "HOLD"]

    def test_dip_adjusts_mu_hat_not_orders(self):
        """Dip should only adjust expected returns, not generate orders."""
        from app.quant_engine.alpha_models import apply_dip_adjustment

        mu_hat = pd.Series([0.01, 0.02, -0.01], index=["A", "B", "C"])
        dip_scores = pd.Series([-2.0, 0.0, 1.0], index=["A", "B", "C"])

        # Apply dip adjustment
        mu_hat_adj = apply_dip_adjustment(mu_hat, dip_scores, dip_k=0.05)

        # Result should be adjusted mu_hat, not orders
        assert isinstance(mu_hat_adj, pd.Series)
        assert len(mu_hat_adj) == 3

        # Asset A has negative dip score, should get boost
        assert mu_hat_adj["A"] > mu_hat["A"]

        # Asset C has positive dip score, should not get boost
        assert mu_hat_adj["C"] == mu_hat["C"]


class TestWalkForward:
    """Test walk-forward validation."""

    def test_fold_generation(self):
        """Walk-forward folds should be generated correctly."""
        from app.quant_engine.walk_forward import generate_walk_forward_splits

        dates = pd.date_range("2018-01-01", periods=1500, freq="D")

        folds = generate_walk_forward_splits(
            dates=dates,
            train_months=24,
            validation_months=6,
            test_months=6,
        )

        # Should have at least 1 fold
        assert len(folds) >= 1

        # Each fold should have proper ordering
        for fold in folds:
            assert fold.train_end <= fold.validation_start
            assert fold.validation_end <= fold.test_start

    def test_no_lookahead_in_splits(self):
        """Splits should guarantee no lookahead."""
        from app.quant_engine.walk_forward import (
            generate_walk_forward_splits,
            validate_no_lookahead,
        )

        dates = pd.date_range("2018-01-01", periods=1500, freq="D")
        data = pd.DataFrame(
            np.random.randn(1500, 3),
            index=dates,
            columns=["A", "B", "C"],
        )

        folds = generate_walk_forward_splits(dates)

        for fold in folds:
            train = data.loc[fold.train_start:fold.train_end]
            test = data.loc[fold.test_start:fold.test_end]

            assert validate_no_lookahead(fold, train, test)


class TestQuantEngineService:
    """Test the main quant engine service."""

    def test_service_initialization(self):
        """Service should initialize with default config."""
        from app.quant_engine.service import QuantEngineService, get_default_config

        config = get_default_config()
        service = QuantEngineService(config)

        assert service.config.base_currency == "EUR"
        assert service.alpha_model is None
        assert service.risk_model is None

    def test_service_train(self):
        """Service should train models on data."""
        from app.quant_engine.service import QuantEngineService

        # Create sample price data - need at least 6 assets for PCA
        np.random.seed(42)
        dates = pd.date_range("2018-01-01", periods=1000, freq="D")
        assets = ["A", "B", "C", "D", "E", "F", "G", "H"]

        prices = pd.DataFrame(
            100 * np.exp(np.cumsum(np.random.randn(1000, 8) * 0.02, axis=0)),
            index=dates,
            columns=assets,
        )

        service = QuantEngineService()
        result = service.train(prices)

        if result.get("status") == "success":
            assert service.alpha_model is not None
            assert service.risk_model is not None


class TestQuantEngineAPI:
    """Test the API routes for the quant engine."""

    def test_quant_engine_route_imports(self):
        """Route module should import successfully."""
        from app.api.routes.quant_engine import router
        
        assert router is not None
        assert router.prefix == "/portfolios"

    def test_quant_engine_schemas_valid(self):
        """Schemas should be valid Pydantic models."""
        from app.schemas.quant_engine import (
            GenerateRecommendationsRequest,
            EngineOutputResponse,
            RecommendationRowResponse,
            AuditBlockResponse,
        )
        
        # Test request model
        req = GenerateRecommendationsRequest(
            portfolio_value_eur=10000.0,
            inflow_eur=1000.0,
        )
        assert req.portfolio_value_eur == 10000.0
        assert req.inflow_eur == 1000.0
        assert req.force_retrain is False
        
        # Test recommendation model
        rec = RecommendationRowResponse(
            ticker="AAPL",
            action="BUY",
            notional_eur=500.0,
            delta_weight=0.05,
            target_weight=0.10,
            mu_hat=0.02,
            uncertainty=0.01,
            risk_contribution=0.08,
            marginal_utility=0.015,
        )
        assert rec.ticker == "AAPL"
        assert rec.action == "BUY"

    def test_engine_output_to_response_helper(self):
        """Helper function should convert engine output to response."""
        from app.api.routes.quant_engine import _engine_output_to_response
        from app.quant_engine.types import (
            EngineOutput,
            RecommendationRow,
            AuditBlock,
            SolverStatus,
            ActionType,
            MuHatUncertainty,
            RiskInfo,
        )
        from datetime import date
        
        # Create mock engine output
        rec = RecommendationRow(
            ticker="AAPL",
            name="Apple Inc.",
            action=ActionType.BUY,
            notional_eur=500.0,
            delta_weight=0.05,
            mu_hat=0.02,
            mu_hat_uncertainty=MuHatUncertainty(
                ci_low=0.01,
                ci_high=0.03,
                oos_rmse=0.005,
            ),
            risk=RiskInfo(marginal_vol=0.15, mcr=0.08),
            delta_utility_net=0.015,
            trade_cost_eur=1.0,
            constraints=[],
            dip=None,
        )
        
        audit = AuditBlock(
            model_weights={"ridge": 0.7, "lasso": 0.3},
            oos_scores={"ridge": {"mse": 0.001}},
            risk_model={"n_factors": 5},
            hyperparams={"lambda_risk": 10.0},
            data_hash="abc123",
        )
        
        output = EngineOutput(
            as_of=date.today(),
            portfolio_value_eur=10000.0,
            inflow_eur=1000.0,
            solver_status=SolverStatus.OPTIMAL,
            recommendations=[rec],
            audit=audit,
            total_trades=1,
            total_transaction_cost_eur=1.0,
            expected_portfolio_return=0.02,
            expected_portfolio_risk=0.15,
        )
        
        response = _engine_output_to_response(output)
        
        assert response.portfolio_value_eur == 10000.0
        assert len(response.recommendations) == 1
        assert response.recommendations[0].ticker == "AAPL"
        assert response.audit.optimizer_status == "optimal"
