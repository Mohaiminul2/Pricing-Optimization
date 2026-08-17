"""
Price elasticity model using log-log linear regression.
Supports confidence intervals and train/test splits.
"""
import sys
from pathlib import Path

# Ensure project root is on path when running as a script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.exceptions import ConvergenceWarning
from scipy import stats
import config

# Suppress only expected sklearn convergence warnings, not all warnings
warnings.filterwarnings("ignore", category=ConvergenceWarning)


class ElasticityModel:
    """
    Estimate price elasticity of demand via log-log regression:

        log(Q) = α + β·log(P) + γ₁·control₁ + ... + ε

    β is the price elasticity (constant-elasticity demand curve).
    """

    def __init__(self):
        self.model = LinearRegression()
        self.elasticity_coef = None
        self.r2 = None
        self.rmse = None
        self.predictions = None
        self.elasticity_se = None          # standard error of β
        self.elasticity_ci = None          # (lower, upper) 95% CI
        self.control_variables = None

    # ------------------------------------------------------------------ #
    #  CORE ESTIMATION                                                     #
    # ------------------------------------------------------------------ #

    def calculate_elasticity(self, df, control_variables=None):
        """
        Fit log-log model and return elasticity coefficient β.

        Parameters
        ----------
        df : pd.DataFrame with columns ['unit_price', 'quantity', ...]
        control_variables : list of column names to include as controls

        Returns
        -------
        float or None
        """
        self.control_variables = control_variables or []

        log_df = df.copy()
        log_df["log_price"] = np.log(log_df["unit_price"] + 1e-6)
        log_df["log_quantity"] = np.log(log_df["quantity"] + 1e-6)

        feature_cols = ["log_price"] + self.control_variables
        X = log_df[feature_cols].copy()
        y = log_df["log_quantity"].copy()

        mask = X.notna().all(axis=1) & y.notna()
        X, y = X[mask].values, y[mask]

        if len(X) < 10:
            return None

        # Use unscaled OLS so the price coefficient equals elasticity directly.
        # Scaling would change the coefficient to standardized units.
        self.model.fit(X, y)
        self.elasticity_coef = float(self.model.coef_[0])

        y_pred = self.model.predict(X)
        self.r2 = float(r2_score(y, y_pred))
        self.rmse = float(np.sqrt(mean_squared_error(y, y_pred)))
        self.predictions = y_pred

        return self.elasticity_coef

    def calculate_elasticity_with_ci(self, df, control_variables=None, confidence=0.95):
        """
        Same as calculate_elasticity but also computes the confidence interval
        for the elasticity coefficient using the t-distribution.
        """
        elasticity = self.calculate_elasticity(df, control_variables)
        if elasticity is None:
            return None

        log_df = df.copy()
        log_df["log_price"] = np.log(log_df["unit_price"] + 1e-6)
        log_df["log_quantity"] = np.log(log_df["quantity"] + 1e-6)

        feature_cols = ["log_price"] + (control_variables or [])
        X_raw = log_df[feature_cols].dropna().values
        y_raw = log_df["log_quantity"].dropna().values

        mask = np.isfinite(X_raw).all(axis=1) & np.isfinite(y_raw)
        X_raw, y_raw = X_raw[mask], y_raw[mask]

        n = len(X_raw)
        p = len(feature_cols)
        if n <= p + 1:
            self.elasticity_ci = (None, None)
            return elasticity

        mse = np.sum((y_raw - self.model.predict(X_raw)) ** 2) / (n - p - 1)

        try:
            X_design = np.column_stack([np.ones(n), X_raw])
            XtX_inv = np.linalg.pinv(X_design.T @ X_design)
            se_coef = np.sqrt(max(mse * XtX_inv[1, 1], 0))
        except (np.linalg.LinAlgError, ValueError):
            se_coef = None

        if se_coef is not None and se_coef > 0:
            t_crit = stats.t.ppf(1 - (1 - confidence) / 2, n - p - 1)
            self.elasticity_se = float(se_coef)
            self.elasticity_ci = (
                float(elasticity - t_crit * se_coef),
                float(elasticity + t_crit * se_coef),
            )
        else:
            self.elasticity_se = None
            self.elasticity_ci = (None, None)

        return elasticity

    def calculate_elasticity_train_test(self, df, control_variables=None):
        """
        Split data chronologically (by index if no invoice_date), fit on train, evaluate on test.
        Returns dict with per-split metrics.

        Note: uses separate ElasticityModel instances per split so that this instance's
        state (elasticity_coef, r2, rmse) is not mutated by the train/test evaluation.
        """
        if "invoice_date" in df.columns:
            df = df.sort_values("invoice_date").reset_index(drop=True)
        else:
            df = df.reset_index(drop=True)
        split_idx = int(len(df) * config.TRAIN_TEST_SPLIT_RATIO)

        df_train, df_test = df.iloc[:split_idx], df.iloc[split_idx:]
        results = {}

        for name, split_df in [("train", df_train), ("test", df_test)]:
            # Use a fresh model per split to avoid mutating self's state
            split_model = ElasticityModel()
            elastic = split_model.calculate_elasticity(split_df, control_variables)
            if elastic is None:
                results[name] = {"elasticity": None, "r2": None, "n": len(split_df)}
                continue
            results[name] = {
                "elasticity": elastic,
                "r2": split_model.r2,
                "n": len(split_df),
            }

        return results

    # ------------------------------------------------------------------ #
    #  SIMULATION                                                          #
    # ------------------------------------------------------------------ #

    def simulate_revenue(self, base_price, base_quantity, elasticity, price_increase_pct):
        """
        Project revenue after a percentage price change.

        Uses the exact constant-elasticity (log-log) demand formula:
            Q_new = Q_base × (P_new / P_base) ^ elasticity

        The linear approximation (%ΔQ ≈ elasticity × %ΔP) is only accurate for
        small price changes; this implementation uses the exact formula to remain
        correct across the full −20% to +50% simulation range.
        """
        new_price = base_price * (1 + price_increase_pct / 100)
        price_ratio = new_price / base_price
        new_quantity = max(base_quantity * (price_ratio ** elasticity), 0)
        quantity_change_pct = (new_quantity / base_quantity - 1) * 100 if base_quantity > 0 else 0

        base_revenue = base_price * base_quantity
        new_revenue = new_price * new_quantity
        revenue_change = new_revenue - base_revenue
        revenue_change_pct = (
            (revenue_change / base_revenue * 100) if base_revenue > 0 else 0
        )

        return {
            "base_price": base_price,
            "new_price": round(new_price, 2),
            "base_quantity": base_quantity,
            "new_quantity": round(new_quantity, 1),
            "quantity_change_pct": round(quantity_change_pct, 2),
            "base_revenue": base_revenue,
            "new_revenue": new_revenue,
            "revenue_change": revenue_change,
            "revenue_change_pct": round(revenue_change_pct, 2),
            "price_increase_pct": price_increase_pct,
        }

    def batch_simulate_revenue(self, base_price, base_quantity, elasticity, price_range):
        """Simulate revenue across a list of price-change percentages."""
        scenarios = [
            self.simulate_revenue(base_price, base_quantity, elasticity, p)
            for p in price_range
        ]
        return pd.DataFrame(scenarios)

    def simulate_profit(self, base_price, base_quantity, elasticity, price_increase_pct, marginal_cost):
        """
        Project profit after a percentage price change given a marginal cost.

        Profit = (P_new - C) × Q_new
        """
        new_price = base_price * (1 + price_increase_pct / 100)
        price_ratio = new_price / base_price
        new_quantity = max(base_quantity * (price_ratio ** elasticity), 0)
        quantity_change_pct = (new_quantity / base_quantity - 1) * 100 if base_quantity > 0 else 0

        base_profit = (base_price - marginal_cost) * base_quantity
        new_profit = (new_price - marginal_cost) * new_quantity
        profit_change = new_profit - base_profit
        profit_change_pct = (
            (profit_change / base_profit * 100) if base_profit > 0 else 0
        )

        return {
            "base_price": base_price,
            "new_price": round(new_price, 2),
            "base_quantity": base_quantity,
            "new_quantity": round(new_quantity, 1),
            "quantity_change_pct": round(quantity_change_pct, 2),
            "base_profit": base_profit,
            "new_profit": new_profit,
            "profit_change": profit_change,
            "profit_change_pct": round(profit_change_pct, 2),
            "price_increase_pct": price_increase_pct,
        }

    def get_optimal_price(self, base_price, base_quantity, elasticity, price_range, marginal_cost=None):
        """
        Find the price that maximises projected revenue or profit.

        When marginal_cost is provided and demand is elastic (β < -1), uses the
        economic markup formula P* = β/(1+β) · C as an interior candidate, then
        compares profit at the clipped candidate and at both endpoints.

        Otherwise falls back to a grid search over the supplied price range.
        """
        if marginal_cost is not None and marginal_cost > 0 and elasticity < -1:
            # Markup formula: P* = β / (1 + β) × C   (valid only for |β| > 1)
            p_star = (elasticity / (1 + elasticity)) * marginal_cost
            p_star = max(p_star, 0)

            # Convert p_star → price-change percentage relative to base_price
            if base_price > 0:
                pct_star = (p_star / base_price - 1) * 100
            else:
                pct_star = 0

            # Build a candidate set: the formula optimum + boundaries
            candidates = {pct_star} | set(price_range)

            best_profit = -np.inf
            best_result = None
            for pct in sorted(candidates):
                result = self.simulate_profit(
                    base_price, base_quantity, elasticity, pct, marginal_cost,
                )
                if result["new_profit"] > best_profit:
                    best_profit = result["new_profit"]
                    best_result = result

            # Also check the two boundaries explicitly (clamp safety)
            for pct in (price_range[0], price_range[-1]):
                if pct not in candidates:
                    result = self.simulate_profit(
                        base_price, base_quantity, elasticity, pct, marginal_cost,
                    )
                    if result["new_profit"] > best_profit:
                        best_profit = result["new_profit"]
                        best_result = result

            if best_result is not None:
                return best_result

        # Fallback: revenue-maximising grid search (backward-compatible)
        scenarios = self.batch_simulate_revenue(
            base_price, base_quantity, elasticity, price_range,
        )
        optimal_idx = scenarios["new_revenue"].idxmax()
        row = scenarios.loc[optimal_idx].to_dict()
        # Annotate with placeholder profit values so callers don't break
        base_profit = (base_price - (marginal_cost or 0)) * base_quantity
        new_profit = row["new_revenue"] - (base_profit - row["base_revenue"])
        row["base_profit"] = base_profit
        row["new_profit"] = new_profit
        row["profit_change"] = new_profit - base_profit
        row["profit_change_pct"] = (
            ((new_profit - base_profit) / base_profit * 100) if base_profit > 0 else 0
        )
        return row


# ---------------------------------------------------------------------- #
#  STANDALONE DEMO                                                         #
# ---------------------------------------------------------------------- #
if __name__ == "__main__":
    np.random.seed(42)
    n = 300
    true_elasticity = -1.2
    base_price = 15.0
    base_qty = 50.0

    prices = np.random.uniform(5, 25, n)
    log_price = np.log(prices / base_price)
    log_qty = np.log(base_qty) + true_elasticity * log_price + np.random.normal(0, 0.15, n)
    quantities = np.exp(log_qty).astype(int).clip(min=1)

    demo_df = pd.DataFrame({"unit_price": prices, "quantity": quantities})

    model = ElasticityModel()
    est = model.calculate_elasticity(demo_df)
    print(f"True elasticity : {true_elasticity}")
    print(f"Estimated elasticity: {est:.3f}")
    print(f"R2               : {model.r2:.3f}")
    print(f"RMSE             : {model.rmse:.3f}")

    model.calculate_elasticity_with_ci(demo_df)
    ci = model.elasticity_ci
    ci_str = f"[{ci[0]:.3f}, {ci[1]:.3f}]" if ci[0] is not None else "N/A"
    print(f"Elasticity (with CI): {est:.3f}  {ci_str}")

    tt = model.calculate_elasticity_train_test(demo_df)
    print(f"Train R2: {tt['train']['r2']:.3f}  |  Test R2: {tt['test']['r2']:.3f}")

    opt = model.get_optimal_price(
        base_price=15.0, base_quantity=base_qty, elasticity=est,
        price_range=list(range(-20, 21, 1)),
    )
    print(f"\nOptimal: raise by {opt['price_increase_pct']:+.0f}% → "
          f"price ${opt['new_price']:.2f}, revenue ${opt['new_revenue']:.2f}")
