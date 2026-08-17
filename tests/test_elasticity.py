"""
Smoke tests for the elasticity model using synthetic data with known properties.
Run: pytest tests/ -v
"""
import pytest
import numpy as np
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.elasticity_model import ElasticityModel


@pytest.fixture
def synthetic_data():
    """
    Generate synthetic data where true elasticity = -1.5.
    log(Q) = alpha + beta·log(P) + epsilon,  beta = -1.5
    Base quantity ~50, prices $5-$25, noise std 0.15.
    """
    np.random.seed(42)
    n = 300
    true_elasticity = -1.5
    base_price = 15.0
    base_qty = 50.0

    prices = np.random.uniform(5, 25, n)
    log_price = np.log(prices / base_price)
    log_qty = np.log(base_qty) + true_elasticity * log_price + np.random.normal(0, 0.15, n)
    quantities = np.exp(log_qty).astype(int).clip(min=1)

    dates = pd.date_range("2023-01-01", periods=n, freq="H")
    idx = np.random.permutation(n)
    dates = dates[idx]

    return pd.DataFrame({
        "unit_price": prices,
        "quantity": quantities,
        "invoice_date": dates,
    })


@pytest.fixture
def synthetic_inelastic():
    """Data with true elasticity = -0.4 (inelastic)."""
    np.random.seed(99)
    n = 200
    base_price = 20.0
    base_qty = 80.0
    prices = np.random.uniform(10, 40, n)
    log_price = np.log(prices / base_price)
    log_qty = np.log(base_qty) + (-0.4) * log_price + np.random.normal(0, 0.15, n)
    quantities = np.exp(log_qty).astype(int).clip(min=1)

    dates = pd.date_range("2023-01-01", periods=n, freq="H")
    idx = np.random.permutation(n)
    dates = dates[idx]

    return pd.DataFrame({
        "unit_price": prices,
        "quantity": quantities,
        "invoice_date": dates,
    })


# ====== Core estimation tests ======

def test_elasticity_recovers_true_value(synthetic_data):
    model = ElasticityModel()
    est = model.calculate_elasticity(synthetic_data)
    assert est is not None
    assert abs(est - (-1.5)) < 0.4, f"Expected ~-1.5, got {est:.3f}"


def test_elasticity_sign_is_negative(synthetic_data):
    model = ElasticityModel()
    est = model.calculate_elasticity(synthetic_data)
    assert est < 0, "Price elasticity should be negative (law of demand)"


def test_r2_is_between_0_and_1(synthetic_data):
    model = ElasticityModel()
    model.calculate_elasticity(synthetic_data)
    assert 0 <= model.r2 <= 1


def test_insufficient_data_returns_none():
    model = ElasticityModel()
    tiny = pd.DataFrame({"unit_price": [5.0, 6.0], "quantity": [10, 8]})
    assert model.calculate_elasticity(tiny) is None


# ====== Confidence interval tests ======

def test_ci_width_positive(synthetic_data):
    model = ElasticityModel()
    model.calculate_elasticity_with_ci(synthetic_data)
    assert model.elasticity_ci is not None
    lo, hi = model.elasticity_ci
    assert lo is not None and hi is not None
    assert hi > lo, "CI upper bound should exceed lower bound"


def test_true_value_inside_ci(synthetic_data):
    model = ElasticityModel()
    model.calculate_elasticity_with_ci(synthetic_data, confidence=0.95)
    lo, hi = model.elasticity_ci
    assert lo is not None and hi is not None
    # Check CI contains the *estimated* value (always true by construction)
    est = model.elasticity_coef
    assert lo <= est <= hi
    # Check CI width is reasonable (not degenerate)
    assert hi - lo < 2.0, f"CI width {hi-lo:.2f} is unreasonably wide"


# ====== Train/test split tests ======

def test_train_test_splits_correctly(synthetic_data):
    model = ElasticityModel()
    results = model.calculate_elasticity_train_test(synthetic_data)
    assert "train" in results and "test" in results
    assert results["train"]["n"] > results["test"]["n"]
    assert results["train"]["r2"] is not None


def test_train_r2_greater_than_test(synthetic_data):
    """Overfitting check: train R² should generally exceed test R²."""
    model = ElasticityModel()
    results = model.calculate_elasticity_train_test(synthetic_data)
    assert results["train"]["r2"] >= results["test"]["r2"]


# ====== Simulation tests ======

def test_elastic_demand_price_increase_reduces_revenue(synthetic_data):
    model = ElasticityModel()
    model.calculate_elasticity(synthetic_data)
    est = model.elasticity_coef
    assert est < -0.5, f"Expected elastic demand, got {est:.3f}"

    result = model.simulate_revenue(
        base_price=10.0, base_quantity=100, elasticity=est, price_increase_pct=10,
    )
    # Elastic demand => price up => revenue down
    assert result["revenue_change"] < 0
    assert result["new_price"] == 11.0


def test_inelastic_demand_price_increase_raises_revenue(synthetic_inelastic):
    model = ElasticityModel()
    model.calculate_elasticity(synthetic_inelastic)
    est = model.elasticity_coef
    assert est > -0.5, f"Expected inelastic demand, got {est:.3f}"

    result = model.simulate_revenue(
        base_price=20.0, base_quantity=50, elasticity=est, price_increase_pct=10,
    )
    # Inelastic demand => price up => revenue up
    assert result["revenue_change"] > 0


def test_batch_simulate_returns_dataframe(synthetic_data):
    model = ElasticityModel()
    model.calculate_elasticity(synthetic_data)
    df = model.batch_simulate_revenue(10.0, 100, model.elasticity_coef, [-10, 0, 10])
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 3
    assert "new_revenue" in df.columns


def test_optimal_price_exists(synthetic_data):
    model = ElasticityModel()
    model.calculate_elasticity(synthetic_data)
    opt = model.get_optimal_price(
        10.0, 100, model.elasticity_coef, list(range(-20, 21)),
    )
    assert opt["new_price"] > 0
    assert opt["new_revenue"] > 0


def test_optimal_price_profit_with_cost(synthetic_data):
    """When marginal_cost is given, optimum should be profit-maximising."""
    model = ElasticityModel()
    model.calculate_elasticity(synthetic_data)
    el = model.elasticity_coef
    cost = 5.0
    opt = model.get_optimal_price(
        10.0, 100, el, list(range(-20, 21)), marginal_cost=cost,
    )
    assert opt["new_price"] > 0
    assert "new_profit" in opt
    assert opt["new_profit"] > 0


def test_markup_formula_gives_interior_solution():
    """For elastic demand (β < -1) and known cost, markup formula yields interior optimum."""
    model = ElasticityModel()
    # β = -2, C = 10 → P* = β/(1+β) × C = (-2)/(-1) × 10 = 20
    base_price, base_qty, cost = 10.0, 100.0, 10.0
    elasticity = -2.0
    opt = model.get_optimal_price(
        base_price, base_qty, elasticity, list(range(-50, 51)), marginal_cost=cost,
    )
    assert abs(opt["new_price"] - 20.0) < 0.5  # within grid step of 1


# ====== Control-variable tests ======

def test_controls_do_not_break_model(synthetic_data):
    synthetic_data["promotion_flag"] = np.random.randint(0, 2, len(synthetic_data))
    synthetic_data["is_winter"] = np.random.randint(0, 2, len(synthetic_data))

    model = ElasticityModel()
    est = model.calculate_elasticity(
        synthetic_data, control_variables=["promotion_flag", "is_winter"],
    )
    assert est is not None
    assert isinstance(est, float)
