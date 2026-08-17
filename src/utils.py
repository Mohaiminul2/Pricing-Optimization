"""
Helper utilities: formatting, charting, etc.
"""
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime


def format_currency(value):
    """Format a number as USD currency string."""
    if value is None:
        return "N/A"
    sign = "-" if value < 0 else ""
    return f"${sign}{abs(value):,.2f}"


def format_currency_million(value):
    """Format a number as USD currency in millions."""
    if value is None:
        return "N/A"
    sign = "-" if value < 0 else ""
    return f"${sign}{abs(value) / 1_000_000:,.2f}M"


def format_percent(value):
    """Format a number as a percentage string."""
    if value is None:
        return "N/A"
    return f"{value:+.2f}%"


def format_number(value):
    """Format an integer with thousand separators."""
    if value is None:
        return "N/A"
    return f"{int(value):,}"


def classify_elasticity(elasticity):
    """Return human-readable elasticity classification."""
    if elasticity is None:
        return "Unknown"
    if elasticity < -1.5:
        return "Highly Elastic"
    elif elasticity < -0.5:
        return "Moderately Elastic"
    else:
        return "Inelastic"


def build_scatter_trend_fig(df, x_col, y_col, title, height=400):
    """Scatter plot with regression trend line."""
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df[x_col], y=df[y_col],
        mode="markers",
        name="Observations",
        marker=dict(size=5, color="#0d47a1", opacity=0.5),
    ))

    from sklearn.linear_model import LinearRegression
    lr = LinearRegression()
    lr.fit(df[[x_col]].values, df[y_col].values)

    x_range = np.linspace(df[x_col].min(), df[x_col].max(), 100)
    y_pred = lr.predict(x_range.reshape(-1, 1))
    fig.add_trace(go.Scatter(
        x=x_range, y=y_pred,
        mode="lines",
        name="Trend",
        line=dict(color="#f44336", width=2),
    ))

    fig.update_layout(
        title=title,
        xaxis_title=x_col.replace("_", " ").title(),
        yaxis_title=y_col.replace("_", " ").title(),
        height=height,
        hovermode="closest",
    )
    return fig


def build_revenue_curve_fig(scenarios_df, highlight_pct=None, title="Revenue by Price Change", marginal_cost=None):
    """Line chart showing revenue (and optionally profit) across price-change scenarios."""
    fig = go.Figure()

    # --- Revenue curve (always shown) ---
    fig.add_trace(go.Scatter(
        x=scenarios_df["price_increase_pct"],
        y=scenarios_df["new_revenue"],
        mode="lines+markers",
        name="Projected Revenue",
        line=dict(color="#0d47a1", width=3),
        marker=dict(size=7),
    ))

    # Highlight selected scenario on revenue curve
    if highlight_pct is not None and not scenarios_df.empty:
        row = scenarios_df[scenarios_df["price_increase_pct"] == highlight_pct]
        if not row.empty:
            fig.add_trace(go.Scatter(
                x=[highlight_pct],
                y=[row.iloc[0]["new_revenue"]],
                mode="markers+text",
                name="Selected",
                marker=dict(size=14, color="#f44336"),
                text=[format_currency(row.iloc[0]["new_revenue"])],
                textposition="top center",
            ))

    # Revenue-optimal point
    if not scenarios_df.empty:
        rev_opt_idx = scenarios_df["new_revenue"].idxmax()
        rev_opt_row = scenarios_df.loc[rev_opt_idx]
        fig.add_trace(go.Scatter(
            x=[rev_opt_row["price_increase_pct"]],
            y=[rev_opt_row["new_revenue"]],
            mode="markers+text",
            name="Revenue Optimal",
            marker=dict(size=12, color="#1565c0", symbol="diamond"),
            text=[f"Rev Opt\n{rev_opt_row['price_increase_pct']:+.0f}%"],
            textposition="bottom right",
        ))

    # --- Profit curve (shown when marginal_cost is valid) ---
    if marginal_cost is not None and marginal_cost > 0 and not scenarios_df.empty:
        # Compute profit for each scenario
        base_profit = (scenarios_df["base_price"] - marginal_cost) * scenarios_df["base_quantity"]
        scenarios_df = scenarios_df.copy()
        scenarios_df["new_profit"] = (
            (scenarios_df["new_price"] - marginal_cost) * scenarios_df["new_quantity"]
        )
        scenarios_df["profit_change_pct"] = (
            ((scenarios_df["new_profit"] - base_profit) / base_profit * 100)
            .replace([np.inf, -np.inf], 0).fillna(0)
        )

        # Profit curve — show even negative values so unprofitable zones are visible
        fig.add_trace(go.Scatter(
            x=scenarios_df["price_increase_pct"],
            y=scenarios_df["new_profit"],
            mode="lines+markers",
            name="Projected Profit",
            line=dict(color="#2e7d32", width=3),
            marker=dict(size=7),
        ))

        # Highlight selected scenario on profit curve
        if highlight_pct is not None:
            row = scenarios_df[scenarios_df["price_increase_pct"] == highlight_pct]
            if not row.empty:
                fig.add_trace(go.Scatter(
                    x=[highlight_pct],
                    y=[row.iloc[0]["new_profit"]],
                    mode="markers+text",
                    name="Selected",
                    marker=dict(size=14, color="#f44336"),
                    text=[format_currency(row.iloc[0]["new_profit"])],
                    textposition="bottom left",
                ))

        # Profit-optimal point
        prof_opt_idx = scenarios_df["new_profit"].idxmax()
        prof_opt_row = scenarios_df.loc[prof_opt_idx]
        fig.add_trace(go.Scatter(
            x=[prof_opt_row["price_increase_pct"]],
            y=[prof_opt_row["new_profit"]],
            mode="markers+text",
            name="Profit Optimal",
            marker=dict(size=12, color="#2e7d32", symbol="diamond"),
            text=[f"Profit Opt\n{prof_opt_row['price_increase_pct']:+.0f}%"],
            textposition="top right",
        ))

        # Update title and y-axis label
        title = "Revenue & Profit by Price Change"
        y_title = "Projected Daily Value ($)"
    else:
        y_title = "Projected Daily Revenue ($)"

    fig.update_layout(
        title=title,
        xaxis_title="Price Change (%)",
        yaxis_title=y_title,
        height=420,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5),
    )
    return fig


def build_segment_scatter(seg_df, x_col="avg_price", y_col="elasticity", group_col="category"):
    """Scatter plot of elasticity vs avg price, coloured by group."""
    fig = px.scatter(
        seg_df,
        x=x_col, y=y_col,
        color=group_col,
        size="n_transactions",
        hover_data=["country"] if group_col == "category" else None,
        title=f"Elasticity by Average Price ({group_col})",
        labels={x_col: "Avg Price ($)", y_col: "Price Elasticity"},
    )
    fig.add_hline(y=-1, line_dash="dash", line_color="orange", opacity=0.6)
    fig.add_hline(y=0, line_dash="solid", line_color="gray", opacity=0.4)
    return fig
