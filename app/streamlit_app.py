"""
Interactive Streamlit dashboard for pricing optimization.
Usage: streamlit run app/streamlit_app.py
"""
import sys
import traceback
import base64
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.database import get_engine, get_session
from src.data_pipeline import DataPipeline
from src.elasticity_model import ElasticityModel
from src.utils import (
    format_currency,
    format_currency_million,
    format_percent,
    format_number,
    classify_elasticity,
    build_scatter_trend_fig,
    build_revenue_curve_fig,
    build_segment_scatter,
)
import config
import streamlit.components.v1 as components

# -------------------------------------------------------------------------- #
# PAGE CONFIG                                                                #
# -------------------------------------------------------------------------- #
st.set_page_config(
    page_title="Pricing Optimization",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)


# -------------------------------------------------------------------------- #
# HELPER FUNCTIONS                                                             #
# -------------------------------------------------------------------------- #


@st.cache_resource
def get_pipeline():
    """Initialize pipeline once per app session."""
    return DataPipeline(config.DB_PATH)


def load_products_df(pipeline):
    query = """
    SELECT
        product_id,
        category,
        COUNT(*) AS transaction_count,
        AVG(unit_price) AS avg_price,
        SUM(quantity) AS total_units_sold,
        SUM(revenue) AS total_revenue
    FROM transactions
    GROUP BY product_id
    ORDER BY total_revenue DESC
    """
    return pd.read_sql(query, pipeline.engine)


def compute_elasticity(pipeline, product_id, use_controls=False):
    """Calculate elasticity for one product; returns dict or None."""
    query = """
    SELECT product_id, invoice_date, quantity, unit_price, revenue,
           promotion_flag, is_winter, is_holiday_season
    FROM transactions
    WHERE product_id = ?
    ORDER BY invoice_date
    """
    df = pd.read_sql(query, pipeline.engine, params=[(product_id,)])
    df["invoice_date"] = pd.to_datetime(df["invoice_date"])

    if len(df) < 20:
        return None

    df = DataPipeline.add_controls(df) if use_controls else df

    model = ElasticityModel()
    controls = config.CONTROL_VARIABLES if use_controls else None

    model.calculate_elasticity(df, control_variables=controls)
    model.calculate_elasticity_with_ci(df, control_variables=controls)
    train_test = model.calculate_elasticity_train_test(df, control_variables=controls)

    return {
        "model": model,
        "elasticity": model.elasticity_coef,
        "r2": model.r2,
        "rmse": model.rmse,
        "ci": model.elasticity_ci,
        "se": model.elasticity_se,
        "data": df,
        "train_test": train_test,
    }


def build_sparkline_svg(values, width=80, height=28, max_points=40):
    """Generate a raw inline SVG sparkline (rendered inside components.html)."""
    if not values or len(values) < 2:
        return ""
    clean = [v for v in values if v is not None and not (isinstance(v, float) and np.isnan(v))]
    if len(clean) < 2:
        return ""
    if len(clean) > max_points:
        indices = np.linspace(0, len(clean) - 1, max_points, dtype=int)
        clean = [clean[i] for i in indices]
    min_v = min(clean)
    max_v = max(clean)
    rng = max_v - min_v or 1
    pts = []
    for i, v in enumerate(clean):
        x = i / (len(clean) - 1) * width
        y = height - ((v - min_v) / rng) * (height - 4) - 2
        pts.append(f"{x:.1f},{y:.1f}")
    path = "M " + " L ".join(pts)
    return (
        f'<svg class="spark" width="{width}" height="{height}" '
        f'xmlns="http://www.w3.org/2000/svg">'
        f'<path class="spark-path" d="{path}" fill="none" stroke-width="1.5" '
        f'stroke-linejoin="round" stroke-linecap="round"/>'
        f'</svg>'
    )


def format_metric_card(label, value, delta=None, help_text="", sparkline=None):
    """Render a premium metric card with glassmorphism and polished design."""
    spark_html = build_sparkline_svg(sparkline) if sparkline else ""
    delta_str  = f" ({delta})" if delta else ""
    # Fixed height for consistent card sizing across all metric cards
    card_height = 135

    html = f"""<!DOCTYPE html>
<html>
<head>
<style>
@import url('https://fonts.googleapis.com/css2?family=Fira+Code:wght@400;500;600&family=Fira+Sans:wght@300;400;500;600;700&display=swap');
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{
  margin: 0; padding: 4px 2px;
  background: transparent;
}}
:root {{
  --bg:          #0E1223;
  --bg-end:      #131B2E;
  --label:       #94A3B8;
  --value:       #F8FAFC;
  --accent:      #4A90E2;
  --accent-mid:  #6BA3E8;
  --positive:    #22C55E;
  --negative:    #EF4444;
  --shadow-sm:   0 1px 2px rgba(0,0,0,0.4);
  --shadow-md:   0 4px 12px rgba(0,0,0,0.5);
  --shadow-lg:   0 8px 24px rgba(0,0,0,0.6);
  --highlight:   rgba(255,255,255,0.08);
  --glass:       rgba(255,255,255,0.03);
}}
.card {{
  background: linear-gradient(160deg, var(--bg) 0%, var(--bg-end) 100%);
  padding: 16px 22px;
  border-radius: 14px;
  border: 1px solid rgba(255,255,255,0.08);
  position: relative;
  height: 120px;
  display: flex;
  flex-direction: column;
  justify-content: space-between;
  box-shadow:
    0 4px 24px rgba(0,0,0,0.4),
    0 2px 8px rgba(0,0,0,0.3),
    inset 0 1px 0 var(--highlight),
    inset 0 0 40px var(--glass);
  font-family: 'Fira Sans', -apple-system, BlinkMacSystemFont, sans-serif;
  transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
  cursor: default;
  overflow: hidden;
}}
/* Top gradient glow */
.card::before {{
  content: "";
  position: absolute;
  top: 0; left: 0; right: 0;
  height: 1px;
  background: linear-gradient(90deg, transparent, rgba(255,255,255,0.15), transparent);
}}
/* Left accent bar with glow */
.card::after {{
  content: "";
  position: absolute;
  left: 0; top: 0; bottom: 0;
  width: 3px;
  background: linear-gradient(180deg, var(--accent) 0%, var(--accent-mid) 100%);
  border-radius: 16px 0 0 16px;
  box-shadow: 0 0 12px var(--accent);
}}
.card:hover {{
  transform: translateY(-4px);
  box-shadow:
    0 8px 32px rgba(0,0,0,0.5),
    0 4px 12px rgba(0,0,0,0.4),
    inset 0 1px 0 rgba(255,255,255,0.1),
    inset 0 0 60px rgba(255,255,255,0.02);
  border-color: rgba(255,255,255,0.12);
}}
.lbl {{
  color: var(--label);
  font-size: 0.5rem;
  font-weight: 400;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  margin-bottom: 8px;
  font-family: 'Fira Code', monospace;
  opacity: 0.9;
}}
.val {{
  color: var(--value);
  font-size: 1.05rem;
  font-weight: 400;
  line-height: 1.2;
  font-family: 'Fira Code', monospace;
  letter-spacing: -0.01em;
}}
.hlp {{
  color: var(--label);
  font-size: 0.5rem;
  margin-top: 6px;
  opacity: 0.7;
  font-family: 'Fira Code', monospace;
}}
.spark {{
  display: block;
  margin-top: 10px;
  opacity: 0.6;
  filter: drop-shadow(0 0 4px var(--accent));
}}
.spark-path {{ stroke: var(--accent); stroke-width: 1.5; }}
</style>
</head>
<body>
<div class="card">
  <div class="lbl">{label}</div>
  <div class="val">{value}{delta_str}</div>
  {f'<div class="hlp">{help_text}</div>' if help_text else ''}
  {spark_html}
</div>
</body>
</html>"""
    components.html(html, height=card_height, scrolling=False)


# -------------------------------------------------------------------------- #
# MAIN APP                                                                   #
# -------------------------------------------------------------------------- #
def main():
    st.markdown("""
    <h1 style="font-size: 1.8rem; font-weight: 700; color: #F8FAFC; margin-bottom: 0.5rem;">
        Pricing Optimization via Elasticity Analysis
    </h1>
    """, unsafe_allow_html=True)
    st.markdown(
        "<p style='font-size: 1rem; color: #94A3B8;'>Estimate price elasticity from historical transactions, "
        "simulate pricing scenarios, and receive data-driven recommendations.</p>",
        unsafe_allow_html=True,
    )

    # ---- Initialise pipeline & product list --------------------------------
    try:
        pipeline = get_pipeline()
        products_df = load_products_df(pipeline)
    except Exception as e:
        st.error(f"Failed to load data: {e}")
        st.stop()

    if products_df.empty:
        st.warning("No products found. Run `python src/data_pipeline.py` to load data.")
        st.stop()

    # ---- Sidebar Configuration --------------------------------------------

    selected_product = st.sidebar.selectbox(
        "Select Product",
        options=products_df["product_id"].values,
        format_func=lambda pid: (
            f"{pid}  |  {products_df[products_df['product_id']==pid]['category'].values[0]}  "
            f"|  Avg ${products_df[products_df['product_id']==pid]['avg_price'].values[0]:.2f}"
        ),
    )

    use_controls = st.sidebar.checkbox(
        "Include promotion/seasonality controls",
        value=False,
        help="Adds promotion flag and winter/holiday dummies to the regression.",
    )

    price_increase = st.sidebar.slider(
        "Price Change (%)",
        min_value=-20, max_value=50, value=10, step=1,
        help="Percentage change to simulate.",
    )

    marginal_cost = st.sidebar.number_input(
        "Marginal Cost ($)",
        min_value=0.0, step=0.01, value=0.0,
        help="Unit cost of goods sold. Set to 0 to optimise for revenue instead of profit.",
    )

    st.sidebar.markdown("---")

    # ---- Cache elasticity per product (single computation) -----------------
    if "elasticity_cache" not in st.session_state:
        st.session_state.elasticity_cache = {}

    cache_key = f"{selected_product}_{use_controls}"
    if cache_key not in st.session_state.elasticity_cache:
        with st.spinner(f"Computing elasticity for {selected_product} ..."):
            result = compute_elasticity(pipeline, selected_product, use_controls=use_controls)
            st.session_state.elasticity_cache[cache_key] = result

    elasticity_result = st.session_state.elasticity_cache[cache_key]
    product_data = products_df[products_df["product_id"] == selected_product].iloc[0]

    # ---- Sidebar: Live Preview ----------------------------------------------
    if elasticity_result is not None:
        _el = elasticity_result["elasticity"]
        _cur_price = product_data["avg_price"]
        _avg_qty = product_data["total_units_sold"] / max(product_data["transaction_count"], 1)
        _prev_model = ElasticityModel()
        _prev_rev = _prev_model.simulate_revenue(_cur_price, _avg_qty, _el, price_increase)

        _color_rev = "#4caf50" if _prev_rev["revenue_change"] >= 0 else "#f44336"
        _arrow_rev = "▲" if _prev_rev["revenue_change"] >= 0 else "▼"

        # Revenue section HTML
        _rev_html = f'<div style="border: 1px solid #1E293B; border-left: 3px solid #4A90E2; border-radius: 8px; padding: 12px 16px; background: linear-gradient(180deg, #0E1223 0%, #131B2E 100%); box-shadow: 0 2px 8px rgba(0,0,0,0.4); margin-top: 12px;">'
        _rev_html += '<div style="font-size: 0.7rem; color: #94A3B8; font-weight: 600; text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 8px; font-family: monospace;">Live Preview</div>'
        _rev_html += f'<div style="font-size: 0.82rem; color: #CBD5E1;">New Price: <span style="font-weight: 600;">{format_currency(_prev_rev["new_price"])}</span></div>'
        _rev_html += f'<div style="font-size: 1rem; font-weight: 700; color: {_color_rev}; margin-top: 4px; font-family: monospace;">{_arrow_rev} {format_currency(abs(_prev_rev["revenue_change"]))} ({_prev_rev["revenue_change_pct"]:+.1f}%)</div>'
        _rev_html += '<div style="font-size: 0.68rem; color: #64748B; margin-top: 2px;">est. revenue impact / txn</div>'

        # Add profit section if marginal cost is set
        if marginal_cost > 0:
            _prev_profit = _prev_model.simulate_profit(_cur_price, _avg_qty, _el, price_increase, marginal_cost)
            _color_profit = "#4caf50" if _prev_profit["profit_change"] >= 0 else "#f44336"
            _arrow_profit = "▲" if _prev_profit["profit_change"] >= 0 else "▼"
            _unit_margin = _prev_profit["new_price"] - marginal_cost
            _unit_margin_color = "#f44336" if _unit_margin < 0 else "#374151"
            _margin_pct = (_unit_margin / _prev_profit["new_price"] * 100) if _prev_profit["new_price"] > 0 else 0

            _rev_html += '<div style="border-top: 1px solid #1E293B; margin-top: 10px; padding-top: 10px;">'
            _rev_html += '<div style="font-size: 0.68rem; color: #94A3B8; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 6px; font-family: monospace;">Profit Impact</div>'
            _rev_html += f'<div style="font-size: 1rem; font-weight: 700; color: {_color_profit}; font-family: monospace;">{_arrow_profit} {format_currency(abs(_prev_profit["profit_change"]))} ({_prev_profit["profit_change_pct"]:+.1f}%)</div>'
            _rev_html += '<div style="font-size: 0.68rem; color: #64748B; margin-top: 2px;">est. profit impact / txn</div>'
            _rev_html += '<div style="display: flex; justify-content: space-between; margin-top: 8px;">'
            _rev_html += f'<div style="text-align: center;"><div style="font-size: 0.65rem; color: #94A3B8; font-family: monospace;">Unit Margin</div><div style="font-size: 0.95rem; font-weight: 600; color: {_unit_margin_color}; font-family: monospace;">{format_currency(_unit_margin)}</div></div>'
            _rev_html += f'<div style="text-align: right;"><div style="font-size: 0.65rem; color: #94A3B8; font-family: monospace;">Margin %</div><div style="font-size: 0.95rem; font-weight: 600; color: {_unit_margin_color}; font-family: monospace;">{format_percent(_margin_pct)}</div></div>'
            _rev_html += '</div></div>'
        else:
            _rev_html += '<div style="margin-top: 10px; font-size: 0.68rem; color: #64748B; font-style: italic; font-family: monospace;">Enter a positive marginal cost to estimate profit impact.</div>'

        _rev_html += '</div>'

        st.sidebar.markdown(_rev_html, unsafe_allow_html=True)

    # ---- Pre-fetch historical data (used for sparklines + overview chart) --
    hist_query = """
    SELECT DATE(invoice_date) AS date,
           AVG(unit_price) AS avg_price,
           SUM(quantity) AS daily_units,
           SUM(revenue) AS daily_revenue
    FROM transactions
    WHERE product_id = ?
    GROUP BY DATE(invoice_date)
    ORDER BY date
    """
    history = pd.read_sql(hist_query, pipeline.engine, params=[(selected_product,)])
    history["date"] = pd.to_datetime(history["date"])

    # ---- Tabs --------------------------------------------------------------
    tab_overview, tab_elasticity, tab_simulation, tab_recommendations = st.tabs([
        "Overview",
        "Elasticity Analysis",
        "Scenario Simulation",
        "Recommendations",
    ])

    # ====================================================================== #
    # TAB 1 — OVERVIEW                                                       #
    # ====================================================================== #
    with tab_overview:
        avg_daily_txns = product_data["transaction_count"] / 365
        avg_daily_qty = product_data["total_units_sold"] / 365
        avg_daily_rev = product_data["total_revenue"] / 365

        spark_price  = history["avg_price"].tolist()   if not history.empty else None
        spark_units  = history["daily_units"].tolist()  if not history.empty else None
        spark_rev    = history["daily_revenue"].tolist() if not history.empty else None

        col1, col2, col3 = st.columns(3)
        with col1:
            format_metric_card(
                "Avg Unit Price",
                format_currency(product_data["avg_price"]),
                sparkline=spark_price,
            )
        with col2:
            format_metric_card(
                "Total Units Sold",
                format_number(product_data["total_units_sold"]),
                sparkline=spark_units,
            )
        with col3:
            format_metric_card(
                "Total Revenue",
                format_currency_million(product_data["total_revenue"]),
                sparkline=spark_rev,
            )

        # Historical price/volume trend

        if not history.empty:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=history["date"], y=history["avg_price"],
                name="Avg Price ($)", yaxis="y1",
                line=dict(color="#0d47a1"),
            ))
            fig.add_trace(go.Bar(
                x=history["date"], y=history["daily_units"],
                name="Units Sold", yaxis="y2",
                marker=dict(color="#f44336", opacity=0.5),
            ))
            fig.update_layout(
                yaxis=dict(title="Price ($)", side="left"),
                yaxis2=dict(title="Units", side="right", overlaying="y"),
                hovermode="x unified",
                height=380,
            )
            st.plotly_chart(fig, use_container_width=True)

    # ====================================================================== #
    # TAB 2 — ELASTICITY ANALYSIS                                             #
    # ====================================================================== #
    
    with tab_elasticity:
        if elasticity_result is None:
            st.warning("Not enough transaction data for this product.")
            st.stop()

        el = elasticity_result["elasticity"]
        r2 = elasticity_result["r2"]
        ci = elasticity_result["ci"]
        se = elasticity_result.get("se")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            format_metric_card(
                "Price Elasticity",
                f"{el:.3f}" if el is not None else "N/A",
                help_text="β: % change in qty per 1% price change",
            )
        with col2:
            format_metric_card("Model R2", f"{r2:.3f}", help_text=f"Explains {r2*100:.1f}% of demand variance")
        with col3:
            format_metric_card(
                "Classification",
                classify_elasticity(el),
                help_text="Elasticity tier for pricing strategy",
            )
        with col4:
            ci_str = f"[{ci[0]:.2f}, {ci[1]:.2f}]" if ci and ci[0] is not None else "N/A"
            width = ci[1] - ci[0] if ci and ci[0] is not None else 0
            precision = "High" if width < 0.5 else ("Medium" if width < 1.0 else "Low")
            format_metric_card("95% CI", ci_str, help_text=f"Confidence width: {precision}")
        # Interpretation callout
        
        st.markdown("")

        if el is not None:
            if el < -1.5:
                st.info(
                    f"**Highly Elastic** (β = {el:.2f}): A 1% price increase "
                    f"cuts volume by ~{abs(el):.2f}%. Lowering price may boost revenue."
                )
            elif el < -0.5:
                st.info(
                    f"**Moderately Elastic** (β = {el:.2f}): A 1% price increase "
                    f"cuts volume by ~{abs(el):.2f}%. Price hikes risk revenue loss."
                )
            else:
                st.info(
                    f"**Inelastic** (β = {el:.2f}): Volume is relatively "
                    f"insensitive to price. Moderate increases may raise revenue."
                )

            # Trustworthiness warnings
            if ci and ci[0] is not None:
                if ci[0] <= 0 <= ci[1]:
                    st.warning(
                        f"⚠️ Confidence interval crosses zero [{ci[0]:.2f}, {ci[1]:.2f}] — "
                        f"the estimate is not statistically distinguishable from no effect. "
                        f"Treat results with caution."
                    )
                elif ci[0] > 0:
                    st.warning(
                        f"⚠️ Entire confidence interval is positive [{ci[0]:.2f}, {ci[1]:.2f}] — "
                        f"this suggests a supply-side relationship or data anomaly. "
                        f"Verify the product has genuine demand variation."
                    )
                elif se and abs(el / se) < 2:
                    ci_width = ci[1] - ci[0]
                    st.warning(
                        f"⚠️ Wide confidence interval (width = {ci_width:.2f}) relative to estimate — "
                        f"low precision. Consider gathering more data."
                    )

        # Train / test split metrics
        tt = elasticity_result.get("train_test")
        if tt and tt["train"]["r2"] is not None:
            st.markdown("#### Train / Test Split")
            c1, c2 = st.columns(2)
            with c1:
                st.metric("Train R²", f"{tt['train']['r2']:.3f}  (n={tt['train']['n']})")
            with c2:
                st.metric("Test R²", f"{tt['test']['r2']:.3f}  (n={tt['test']['n']})")
            gap = abs(tt["train"]["r2"] - tt["test"]["r2"])
            if gap > 0.15:
                st.warning(f"Train-test R² gap = {gap:.2f} — possible overfitting.")
            else:
                st.success("Model generalizes well (small train-test gap).")

        # Scatter + trend
        st.markdown("#### Price vs. Quantity (Log-Log)")
        df_plot = elasticity_result["data"]
        fig = build_scatter_trend_fig(
            df_plot, "unit_price", "quantity",
            f"Elasticity for {selected_product}  (β = {el:.3f})",
        )
        st.plotly_chart(fig, use_container_width=True)

        # Segment comparison
        st.markdown("#### Segment-Level Elasticity (by Category)")
        seg_df = pipeline.get_segment_elasticities()
        if not seg_df.empty:
            fig_seg = build_segment_scatter(seg_df)
            st.plotly_chart(fig_seg, use_container_width=True)
            
            # Methodology note
            st.caption(
                "Segment elasticities are estimated using **within-product variation** to reduce "
                "cross-product pooling bias. Reliability depends on the number of products, "
                "transaction volume, and observed price movement within each segment."
            )
            
            # Caution for small segments
            min_products = seg_df["n_products"].min()
            min_transactions = seg_df["n_transactions"].min()
            if min_products < 3 or min_transactions < 1000:
                st.warning(
                    f"⚠️ Limited data for some segments (minimum {min_products} products, "
                    f"{min_transactions:,} transactions). Treat estimates from these segments "
                    "with caution."
                )
        else:
            st.caption("Not enough segments with sufficient data for comparison.")

    # ====================================================================== #
    # TAB 3 — SCENARIO SIMULATION                                             #
    # ====================================================================== #
    with tab_simulation:
        if elasticity_result is None:
            st.warning("No elasticity estimate available.")
            st.stop()

        el = elasticity_result["elasticity"]
        current_price = product_data["avg_price"]
        avg_qty_per_txn = product_data["total_units_sold"] / max(product_data["transaction_count"], 1)

        model = ElasticityModel()
        scenario = model.simulate_revenue(
            current_price, avg_qty_per_txn, el, price_increase,
        )

        # Determine if we should show profit metrics
        has_valid_cost = marginal_cost > 0
        profit_scenario = None
        if has_valid_cost:
            profit_scenario = model.simulate_profit(
                current_price, avg_qty_per_txn, el, price_increase, marginal_cost,
            )

        # --- Revenue/KPI cards (always shown) ---
        c1, c2, c3 = st.columns(3)
        with c1:
            price_change = scenario["new_price"] - scenario["base_price"]
            help_text = f"({format_currency(abs(price_change))} {'increase' if price_change >= 0 else 'decrease'} from base)"
            format_metric_card("New Price", format_currency(scenario["new_price"]), help_text=help_text)
        with c2:
            qty_change = int(scenario["new_quantity"] - scenario["base_quantity"])
            format_metric_card(
                "Qty Change",
                format_percent(scenario["quantity_change_pct"]),
                help_text=f"({qty_change:+,} units per transaction)",
            )
        with c3:
            fmt = format_percent(scenario["revenue_change_pct"])
            help_text = f"({format_currency(abs(scenario['revenue_change']))} {'increase' if scenario['revenue_change'] >= 0 else 'decrease'})"
            format_metric_card("Revenue Impact", fmt, help_text=help_text)

        # --- Profit KPI cards (conditional) ---
        if has_valid_cost:
            profit_warning = ""
            if profit_scenario["new_price"] < marginal_cost:
                profit_warning = "\n⚠️ Price falls below marginal cost — negative unit margin!"
            
            p1, p2, p3, p4 = st.columns(4)
            with p1:
                current_unit_margin = current_price - marginal_cost
                format_metric_card("Current Unit Margin", format_currency(current_unit_margin))
            with p2:
                expected_unit_margin = profit_scenario["new_price"] - marginal_cost
                margin_color = "#f44336" if expected_unit_margin < 0 else None
                if margin_color:
                    st.markdown(f'''
                    <div class="metric-card" style="--card-bg:#fde8e8;--card-label:#991b1b;--card-value:#991b1b;--card-accent:#dc2626;">
                        <div class="lbl">Expected Unit Margin</div>
                        <div class="val" style="color:{margin_color};">{format_currency(expected_unit_margin)}</div>
                    </div>
                    ''', unsafe_allow_html=True)
                else:
                    format_metric_card("Expected Unit Margin", format_currency(expected_unit_margin))
            with p3:
                format_metric_card(
                    "Current Profit",
                    format_currency(profit_scenario["base_profit"]),
                    help_text="per transaction",
                )
            with p4:
                format_metric_card(
                    "Expected Profit",
                    format_currency(profit_scenario["new_profit"]),
                    delta=format_percent(profit_scenario["profit_change_pct"]) if profit_scenario["base_profit"] != 0 else "N/A",
                    help_text="per transaction",
                )
            
            if profit_warning:
                st.error(profit_warning)

        st.markdown("")

        # --- Financial impact table ---
        st.markdown(f"#### Scenario: {price_increase:+.0f}% Price Change")
        revenue_rows = [
            ("Current Price", format_currency(scenario["base_price"])),
            ("New Price", format_currency(scenario["new_price"])),
            ("Current Qty (per txn)", f"{scenario['base_quantity']:.1f}"),
            ("Expected Qty (per txn)", f"{scenario['new_quantity']:.1f}"),
            ("Current Revenue (per txn)", format_currency(scenario["base_revenue"])),
            ("Expected Revenue (per txn)", format_currency(scenario["new_revenue"])),
            ("Revenue Change", format_currency(scenario["revenue_change"])),
            ("Revenue Change %", format_percent(scenario["revenue_change_pct"])),
        ]
        
        if has_valid_cost:
            profit_rows = [
                ("Marginal Cost (per unit)", format_currency(marginal_cost)),
                ("Current Unit Margin", format_currency(current_price - marginal_cost)),
                ("Expected Unit Margin", format_currency(profit_scenario["new_price"] - marginal_cost)),
                ("Current Profit (per txn)", format_currency(profit_scenario["base_profit"])),
                ("Expected Profit (per txn)", format_currency(profit_scenario["new_profit"])),
                ("Profit Change", format_currency(profit_scenario["profit_change"])),
                ("Profit Change %", format_percent(profit_scenario["profit_change_pct"]) if profit_scenario["base_profit"] != 0 else "N/A"),
            ]
            table_data = pd.DataFrame({
                "Metric": [r[0] for r in revenue_rows] + [r[0] for r in profit_rows],
                "Value": [r[1] for r in revenue_rows] + [r[1] for r in profit_rows],
            })
        else:
            table_data = pd.DataFrame({
                "Metric": [r[0] for r in revenue_rows],
                "Value": [r[1] for r in revenue_rows],
            })
            if not has_valid_cost:
                st.caption("*Marginal cost not set — showing revenue metrics only. Enter a positive marginal cost in the sidebar to see profit projections.*")
        
        st.table(table_data)

        # Multi-scenario revenue/profit curve
        st.markdown("#### Revenue Sensitivity Curve")
        price_range = list(range(-20, 51, 5))
        scenarios_df = model.batch_simulate_revenue(
            current_price, avg_qty_per_txn, el, price_range,
        )
        fig = build_revenue_curve_fig(
            scenarios_df, highlight_pct=price_increase,
            marginal_cost=marginal_cost if marginal_cost > 0 else None,
        )
        st.plotly_chart(fig, use_container_width=True)

    # ====================================================================== #
    # TAB 4 — RECOMMENDATIONS                                                 #
    # ====================================================================== #
    with tab_recommendations:
        if elasticity_result is None:
            st.warning("Cannot generate recommendations.")
            st.stop()

        el = elasticity_result["elasticity"]
        current_price = product_data["avg_price"]
        avg_qty_per_txn = product_data["total_units_sold"] / max(product_data["transaction_count"], 1)

        model = ElasticityModel()
        price_range = list(np.arange(-20, 51, 1))
        optimal = model.get_optimal_price(
            current_price, avg_qty_per_txn, el, price_range,
            marginal_cost=marginal_cost if marginal_cost > 0 else None,
        )

        st.markdown("#### Optimal Pricing Strategy")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"""
            **Current Price**: {format_currency(current_price)}\n\n
            **Recommended Price**: {format_currency(optimal['new_price'])}\n\n
            **Recommended Change**: {optimal['price_increase_pct']:+.1f}%
            """)
        if "new_profit" in optimal:
            with c2:
                st.markdown(f"""
                **Profit Increase per txn**: {format_currency(optimal['profit_change'])}\n\n
                **Profit % Change**: {optimal['profit_change_pct']:+.2f}%\n\n
                **Expected New Volume**: {optimal['new_quantity']:.1f} units
                """)
        else:
            with c2:
                st.markdown(f"""
                **Revenue Increase per txn**: {format_currency(optimal['revenue_change'])}\n\n
                **Revenue % Change**: {optimal['revenue_change_pct']:+.2f}%\n\n
                **Expected New Volume**: {optimal['new_quantity']:.1f} units
                """)

        st.markdown("---")

        # Recommendation callout
        profit_metric = "profit" if ("new_profit" in optimal) else "revenue"
        _change_val = optimal.get('profit_change') if 'profit_change' in optimal else optimal.get('revenue_change', 0)
        _change_pct = optimal.get('profit_change_pct') if 'profit_change_pct' in optimal else optimal.get('revenue_change_pct', 0)
        
        if optimal["price_increase_pct"] > 5:
            st.success(
                f"**Raise price by {optimal['price_increase_pct']:.1f}%** — "
                f"This product shows inelastic demand (β = {el:.2f}). "
                f"At {format_currency(optimal['new_price'])}, estimated {profit_metric} rises "
                f"by {format_currency(_change_val)} per transaction "
                f"({_change_pct:+.1f}%). Implementation risk: LOW."
            )
        elif optimal["price_increase_pct"] < -5:
            _change_val = optimal.get('profit_change') if 'profit_change' in optimal else optimal.get('revenue_change', 0)
            st.warning(
                f"**Lower price by {abs(optimal['price_increase_pct']):.1f}%** — "
                f"Demand is elastic (β = {el:.2f}). A price cut to "
                f"{format_currency(optimal['new_price'])} could boost {profit_metric} by "
                f"{format_currency(_change_val)} through higher volume. "
                f"Implementation risk: MEDIUM (inventory planning needed)."
            )
        else:
            st.info(
                "**Maintain current price** — The price is near the revenue-maximizing "
                "point. Focus on non-price levers (promotions, bundling, channel expansion)."
            )

        # Naive baseline comparison
        st.markdown("---")
        st.markdown("#### Baseline Comparison (Naive vs. Optimized)")
        if "new_profit" in optimal:
            current_daily_profit = (current_price - (marginal_cost or 0)) * avg_qty_per_txn * product_data["transaction_count"] / 365
            opt_daily_profit = optimal["new_profit"] * product_data["transaction_count"] / 365
            annual_baseline = current_daily_profit * 365
            annual_optimized = opt_daily_profit * 365
            baseline_df = pd.DataFrame({
                "Strategy": ["Keep Current Price (Naive)", "Optimized Price"],
                "Annual Profit": [
                    format_currency(annual_baseline),
                    format_currency(annual_optimized),
                ],
                "vs. Baseline": ["—", format_percent((annual_optimized - annual_baseline) / max(annual_baseline, 1) * 100)],
            })
            st.table(baseline_df)
            current_daily_rev = current_price * avg_qty_per_txn * product_data["transaction_count"] / 365
            opt_daily_rev = optimal.get("new_revenue", 0) * product_data["transaction_count"] / 365
            rev_annual_baseline = current_daily_rev * 365
            rev_annual_optimized = opt_daily_rev * 365
        else:
            current_daily_rev = current_price * avg_qty_per_txn * product_data["transaction_count"] / 365
            opt_daily_rev = optimal.get("new_revenue", 0) * product_data["transaction_count"] / 365
            annual_baseline = current_daily_rev * 365
            annual_optimized = opt_daily_rev * 365

            baseline_df = pd.DataFrame({
                "Strategy": ["Keep Current Price (Naive)", "Optimized Price"],
                "Annual Revenue": [
                    format_currency(annual_baseline),
                    format_currency(annual_optimized),
                ],
                "vs. Baseline": ["—", format_percent((annual_optimized - annual_baseline) / max(annual_baseline, 1) * 100)],
            })
            st.table(baseline_df)
            rev_annual_baseline = annual_baseline
            rev_annual_optimized = annual_optimized

        # Action items
        st.markdown("---")
        col_actions, col_business = st.columns(2)
        with col_actions:
            st.markdown("#### Recommended Actions")
            actions = [
                "Validate elasticity estimate with product & sales teams",
                "Run a controlled A/B test on a subset of customers or regions",
                "Adjust inventory forecasts if price is lowered (expect volume bump)",
                "Communicate the change to customer-facing teams in advance",
                "Re-run elasticity analysis in 30/60/90 days to capture learned behaviour",
            ]
            for a in actions:
                st.markdown(f"- {a}")

        with col_business:
            st.markdown("#### Business Case Summary")
            baseline_metric = "profit" if marginal_cost > 0 else "revenue"
            daily_label = "daily profit" if marginal_cost > 0 else "daily revenue"
            annual_label = "Annual profit (est.)" if marginal_cost > 0 else "Annual revenue (est.)"
            impact_label = "Projected annual profit impact" if marginal_cost > 0 else "Projected annual revenue impact"
            
            summary = f"""
            **Product:** {selected_product}  |  **Category:** {product_data['category']}

            **Current:** {format_currency(current_price)} / unit · ~{avg_qty_per_txn:.0f} units/txn  
            **{daily_label.capitalize()} (est.):** {format_currency(current_daily_rev if baseline_metric == "revenue" else current_daily_profit)}  
            **{annual_label}:** {format_currency(annual_baseline)}

            **Proposed action:** {'Increase' if optimal['price_increase_pct'] > 0 else 'Decrease'} price by {abs(optimal['price_increase_pct']):.1f}%  
            **New price:** {format_currency(optimal['new_price'])} · New vol: {optimal['new_quantity']:.1f} units/txn

            **{impact_label}:** {format_currency(rev_annual_optimized - rev_annual_baseline)}  
            **Confidence:** {int(elasticity_result['r2'] * 100)}% (model R2 = {elasticity_result['r2']:.3f})
            """
            st.markdown(summary)


if __name__ == "__main__":
    main()
