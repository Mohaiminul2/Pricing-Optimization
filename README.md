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

### Data Flow
1. **Ingest**: CSV → `_clean_data()` → `add_controls()` → SQLite
2. **Estimate**: Query → Log-log regression → β, R², CI, SE
3. **Simulate**: β + price scenarios → Quantity & revenue forecasts
4. **Optimize**: Grid search (revenue) or markup formula (profit) → optimal price
5. **Visualize**: Plotly charts → Streamlit dashboard

---

### Completed Features 
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

### Planned Improvements 
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

---

## Quick Reference

| Command | Description |
|---------|-------------|
| `streamlit run app/streamlit_app.py` | Launch dashboard |
| `python src/data_pipeline.py` | Generate/regenerate synthetic data |
| `pytest tests/ -v` | Run test suite |
| `deactivate` | Exit virtual environment |
