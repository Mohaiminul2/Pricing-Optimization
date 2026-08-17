# Pricing Optimization via Elasticity Analysis

A data-driven pricing intelligence platform that estimates price elasticity of demand from historical transaction data and recommends revenue- or profit-maximizing prices through interactive visualization.

## Description

Most retailers set prices based on intuition rather than empirical evidence, leaving money on the table through suboptimal pricing. This project solves that problem by:

1. **Estimating price elasticity** — Using log-log linear regression with optional promotion/seasonality controls to quantify how demand responds to price changes
2. **Simulating scenarios** — Projecting revenue and profit impacts across a range of price changes
3. **Recommending optimal prices** — Identifying the price point that maximizes either revenue or profit (when marginal cost is provided)

The result is an interactive Streamlit dashboard that transforms raw transaction data into actionable pricing insights.

---

## Prerequisites

| Requirement | Version | Notes |
|-------------|---------|-------|
| Python | 3.9+ | Required for pandas/numpy compatibility |
| pip | Latest | Package installer |
| SQLite | Bundled | No separate install needed |
| Memory | 4GB+ | For synthetic data generation (~100K rows) |

### Python Dependencies

All dependencies are listed in `requirements.txt` and include:

```
pandas==2.0.3
numpy==1.24.3
scikit-learn==1.3.0
scipy==1.11.0
sqlalchemy==2.0.20
streamlit==1.28.0
plotly==5.17.0
jupyter==1.0.0
pytest==7.4.3
```

---

## Installation Instructions

### Step 1: Clone or Navigate to Project
```bash
cd "/Users/mohaiminulislam/_MyWrkSpace/3.Github ready/pricing-optimization"
```

### Step 2: Create Virtual Environment
```bash
python3 -m venv venv
```

### Step 3: Activate Virtual Environment
**macOS/Linux:**
```bash
source venv/bin/activate
```

**Windows:**
```bash
venv\Scripts\activate
```

### Step 4: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 5: Verify Data Existence
```bash
ls data/transactions.db
ls data/raw/synthetic_retail.csv
```

If files are missing, generate synthetic data first:
```bash
python src/data_pipeline.py
```

### Step 6: Run the Application
```bash
streamlit run app/streamlit_app.py
```

### Step 7: Access the Dashboard
Open your browser to: **http://localhost:8501**

---

## Extended Examples

### Example 1: Basic Workflow
1. Select a product from the sidebar dropdown
2. Review the Overview tab to see current metrics and trends
3. Check Elasticity Analysis for the β coefficient and confidence interval
4. Go to Scenario Simulation and adjust the price change slider
5. View the Revenue Sensitivity Curve with both revenue and profit lines
6. Read Recommendations for the optimal price strategy

### Example 2: Profit-Based Optimization
1. Set a positive "Marginal Cost ($)" in the sidebar (e.g., $5.00)
2. The dashboard switches from revenue-maximization to profit-maximization
3. Observe the different optimal price points between revenue vs. profit
4. The Markup Formula `P* = β/(1+β) × C` is applied automatically when β < -1
5. Compare annual profit projections in the Baseline Comparison table

### Example 3: Understanding Confidence Intervals
1. Navigate to the Elasticity Analysis tab
2. Observe the "95% CI" metric card showing the confidence interval
3. If the CI crosses zero, a warning appears: "Confidence interval crosses zero — not statistically distinguishable from no effect"
4. Wide intervals trigger a precision warning: "Low precision. Consider gathering more data"
5. Use this information to assess whether the elasticity estimate is reliable enough for pricing decisions

### Example 4: Segment-Level Comparison
1. In the Elasticity Analysis tab, scroll to "Segment-Level Elasticity"
2. View the scatter plot comparing elasticities across categories and countries
3. Each point represents a segment with its own elasticity estimate
4. The methodology note explains: "Segment elasticities are estimated using within-product variation to reduce pooling bias"
5. Caution labels appear for segments with limited data (< 3 products or < 1,000 transactions)

---

## Testing Instructions

### Run All Tests
```bash
./venv/bin/python -m pytest tests/test_elasticity.py -v
```

### Expected Output
```
tests/test_elasticity.py::test_elasticity_recovers_true_value PASSED
tests/test_elasticity.py::test_elasticity_sign_is_negative PASSED
tests/test_elasticity.py::test_r2_is_between_0_and_1 PASSED
tests/test_elasticity.py::test_insufficient_data_returns_none PASSED
tests/test_elasticity.py::test_ci_width_positive PASSED
tests/test_elasticity.py::test_true_value_inside_ci PASSED
tests/test_elasticity.py::test_train_test_splits_correctly PASSED
tests/test_elasticity.py::test_train_r2_greater_than_test PASSED
tests/test_elasticity.py::test_elastic_demand_price_increase_reduces_revenue PASSED
tests/test_elasticity.py::test_inelastic_demand_price_increase_raises_revenue PASSED
tests/test_elasticity.py::test_batch_simulate_returns_dataframe PASSED
tests/test_elasticity.py::test_optimal_price_exists PASSED
tests/test_elasticity.py::test_optimal_price_profit_with_cost PASSED
tests/test_elasticity.py::test_markup_formula_gives_interior_solution PASSED
tests/test_elasticity.py::test_controls_do_not_break_model PASSED

============================== 15 passed in 1.42s ===============================
```

### Test Coverage
| Test | Purpose |
|------|---------|
| `test_elasticity_recovers_true_value` | Verifies model recovers known β from synthetic data |
| `test_elasticity_sign_is_negative` | Ensures demand law holds (β < 0) |
| `test_r2_is_between_0_and_1` | Validates R² bounds |
| `test_insufficient_data_returns_none` | Edge case: < 20 transactions |
| `test_ci_width_positive` | Confidence interval calculation |
| `test_true_value_inside_ci` | True β falls within 95% CI |
| `test_train_test_splits_correctly` | Chronological split validation |
| `test_train_r2_greater_than_test` | Overfitting detection |
| `test_elastic_demand_*` | Revenue impact for elastic goods |
| `test_inelastic_demand_*` | Revenue impact for inelastic goods |
| `test_batch_simulate_*` | Multi-scenario simulation |
| `test_optimal_price_exists` | Revenue optimization |
| `test_optimal_price_profit_with_cost` | Profit optimization with cost |
| `test_markup_formula_gives_interior_solution` | Markup formula: P* = β/(1+β) × C |
| `test_controls_do_not_break_model` | Controls (promotion/seasonality) integration |

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                    PRICING OPTIMIZATION SYSTEM                  │
└─────────────────────────────────────────────────────────────────┘

┌──────────────────┐      ┌──────────────────┐      ┌──────────────┐
│   Raw Data       │─────▶│  Data Pipeline    │─────▶│  SQLite DB    │
│  (CSV import)    │      │  • Cleaning       │      │  (transactions│
│                  │      │  • Controls       │      │   table)     │
│                  │      │  • Outlier filter │      └──────┬───────┘
└──────────────────┘      └──────────────────┘             │
                                                          ┌▼───────┐
┌──────────────────┐      ┌──────────────────┐           │ Elastic│
│  Streamlit App   │◄─────│  Elasticity Model│◄──────────│  Model │
│  (Dashboard)     │      │  (Log-log reg)   │           └────────┘
│                  │      │                  │                ▲
│  ┌────────────┐  │      │  calculate_      │                │
│  │Overview Tab│  │      │  elasticity()    │────────────────┘
│  ├────────────┤  │      │  + CI/SE         │
│  │Elasticity  │  │      │                  │
│  │Analysis Tab│  │      │  batch_simulate_ │
│  ├────────────┤  │      │  revenue()       │
│  │Simulation  │  │      │  simulate_profit │
│  │Tab         │  │      │  get_optimal_    │
│  ├────────────┤  │      │  price()         │
│  │Recommend-  │  │      └──────────────────┘
│  │ations Tab  │  │
│  └────────────┘  │
└──────────────────┘
```

### Data Flow
1. **Ingest**: CSV → `_clean_data()` → `add_controls()` → SQLite
2. **Estimate**: Query → Log-log regression → β, R², CI, SE
3. **Simulate**: β + price scenarios → Quantity & revenue forecasts
4. **Optimize**: Grid search (revenue) or markup formula (profit) → optimal price
5. **Visualize**: Plotly charts → Streamlit dashboard

---

## Configuration

### Config File: `config.py`

```python
# Paths
DB_PATH = "sqlite:///data/transactions.db"

# Data cleaning parameters
MIN_TRANSACTIONS_PER_PRODUCT = 20
MIN_PRICE_VARIANCE = 0.05          # 5% min price variation
PRICE_CHANGE_THRESHOLD = 50        # Max % price change filter
PRICE_OUTLIER_QUANTILE = 0.99      # Per-group outlier cap
MIN_PER_PRODUCT_FOR_OUTLIER = 10   # Min rows for per-product bounds
HIGH_VOLUME_THRESHOLD = 50         # Periods with qty >= this
MAX_HIGH_VOLUME_CHANGE_PCT = 300   # Cap extreme changes for high-volume

# Elasticity model parameters
TEST_SIZE = 0.2                    # 20% test split
RANDOM_STATE = 42                  # Reproducibility
ELASTICITY_THRESHOLD = -0.5        # Inelastic threshold
TRAIN_TEST_SPLIT_RATIO = 0.8       # 80/20 split
CONTROL_VARIABLES = ["promotion_flag", "is_winter", "is_holiday_season"]
MIN_TRANSACTIONS_FOR_SEGMENT = 30  # Min rows per segment

# Streamlit
STREAMLIT_THEME = "light"
DEFAULT_PRICE_INCREASE_RANGE = (-20, 50)
```

### Streamlit Theme: `.streamlit/config.toml`

```toml
[theme]
primaryColor = "#4A90E2"
backgroundColor = "#020617"
secondaryBackgroundColor = "#0E1223"
textColor = "#F8FAFC"
font = "sans serif"
```

---

## Roadmap

### Completed Features ✅
- [x] Log-log elasticity estimation with controls
- [x] Confidence intervals and standard errors
- [x] Train/test split evaluation
- [x] Revenue simulation across price ranges
- [x] Profit optimization with marginal cost
- [x] Markup formula for elastic demand (β < -1)
- [x] Interactive Streamlit dashboard
- [x] Sparkline trends in metric cards
- [x] Segment-level elasticity analysis
- [x] Dark mode analytics theme
- [x] Live sidebar preview with revenue/profit

### Planned Improvements 🚧
- [ ] Import real Kaggle Online Retail II dataset
- [ ] Cross-price elasticity (substitutes/complements)
- [ ] Customer cohort segmentation (new vs. repeat)
- [ ] Deployment to Streamlit Cloud
- [ ] A/B test tracking for price changes
- [ ] REST API for programmatic access
- [ ] Historical price change recommendations
- [ ] Export reports (PDF/Excel)

---

## License

This project is licensed under the **MIT License**.

See the [LICENSE](LICENSE) file for the full license text.

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

---

## Support / Contact

For issues, questions, or contributions:

- **GitHub Issues**: Report bugs or request features via the repository's issue tracker
- **Documentation**: See inline code comments and the `src/` module docstrings
- **Email**: [Your email here]

---

## Quick Reference

| Command | Description |
|---------|-------------|
| `streamlit run app/streamlit_app.py` | Launch dashboard |
| `python src/data_pipeline.py` | Generate/regenerate synthetic data |
| `pytest tests/ -v` | Run test suite |
| `deactivate` | Exit virtual environment |
