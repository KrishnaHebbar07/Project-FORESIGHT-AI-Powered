"""
Project FORESIGHT: UI Components & Visualization Engine
--------------------------------------------------------
Provides sleek CSS styling, modern KPI card renderers, and interactive
Plotly visualizations for demand forecasting and inventory health analytics.
"""

from typing import Dict, List, Optional
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st


# Color Palette
COLOR_STOCKOUT = "#EF4444"      # Rose / Red
COLOR_OVERSTOCK = "#F59E0B"     # Amber / Yellow
COLOR_HEALTHY = "#10B981"       # Emerald / Green
COLOR_ACCENT = "#6366F1"        # Indigo
COLOR_FORECAST = "#38BDF8"      # Sky Blue
COLOR_HISTORICAL = "#94A3B8"    # Slate
COLOR_CONFIDENCE = "rgba(56, 189, 248, 0.15)"


def inject_custom_css():
    """Injects modern, responsive CSS styling for Project FORESIGHT."""
    custom_css = """
    <style>
    /* Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }

    /* Main Container Padding */
    .block-container {
        padding-top: 1.5rem !important;
        padding-bottom: 3rem !important;
        max-width: 96% !important;
    }

    /* Glowing App Header */
    .app-header {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.8) 0%, rgba(15, 23, 42, 0.95) 100%);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 24px 32px;
        margin-bottom: 24px;
        backdrop-filter: blur(12px);
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3), 0 8px 10px -6px rgba(0, 0, 0, 0.3);
    }
    
    .app-title {
        font-size: 2.1rem;
        font-weight: 800;
        background: linear-gradient(90deg, #60A5FA 0%, #A78BFA 50%, #F472B6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 6px;
        letter-spacing: -0.02em;
    }

    .app-subtitle {
        color: #94A3B8;
        font-size: 0.98rem;
        font-weight: 500;
        margin-bottom: 0;
    }

    /* KPI Metrics Cards */
    .kpi-card {
        background: linear-gradient(145deg, #1E293B 0%, #0F172A 100%);
        border: 1px solid rgba(255, 255, 255, 0.07);
        border-radius: 14px;
        padding: 20px;
        transition: transform 0.2s ease, box-shadow 0.2s ease, border-color 0.2s ease;
        position: relative;
        overflow: hidden;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
    }

    .kpi-card:hover {
        transform: translateY(-2px);
        border-color: rgba(99, 102, 241, 0.4);
        box-shadow: 0 12px 20px -5px rgba(0, 0, 0, 0.4);
    }

    .kpi-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: linear-gradient(90deg, #3B82F6, #8B5CF6);
    }

    .kpi-card.danger::before { background: linear-gradient(90deg, #EF4444, #F43F5E); }
    .kpi-card.warning::before { background: linear-gradient(90deg, #F59E0B, #D97706); }
    .kpi-card.success::before { background: linear-gradient(90deg, #10B981, #059669); }
    .kpi-card.info::before { background: linear-gradient(90deg, #38BDF8, #6366F1); }

    .kpi-title {
        color: #94A3B8;
        font-size: 0.82rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-bottom: 8px;
        display: flex;
        align-items: center;
        gap: 6px;
    }

    .kpi-value {
        color: #F8FAFC;
        font-size: 1.85rem;
        font-weight: 700;
        letter-spacing: -0.02em;
        line-height: 1.1;
        margin-bottom: 6px;
        font-family: 'JetBrains Mono', monospace;
    }

    .kpi-subtext {
        font-size: 0.78rem;
        color: #64748B;
        font-weight: 500;
    }

    /* Badges */
    .badge {
        display: inline-flex;
        align-items: center;
        padding: 4px 10px;
        border-radius: 9999px;
        font-size: 0.76rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }
    
    .badge-stockout {
        background-color: rgba(239, 68, 68, 0.18);
        color: #F87171;
        border: 1px solid rgba(239, 68, 68, 0.4);
    }

    .badge-overstock {
        background-color: rgba(245, 158, 11, 0.18);
        color: #FBBF24;
        border: 1px solid rgba(245, 158, 11, 0.4);
    }

    .badge-healthy {
        background-color: rgba(16, 185, 129, 0.18);
        color: #34D399;
        border: 1px solid rgba(16, 185, 129, 0.4);
    }

    /* Section Cards */
    .section-card {
        background: #1E293B;
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 14px;
        padding: 22px;
        margin-bottom: 20px;
    }

    /* Sidebar Tweaks */
    section[data-testid="stSidebar"] {
        background-color: #0F172A;
        border-right: 1px solid rgba(255, 255, 255, 0.07);
    }

    /* Streamlit DataFrame & Table Styling */
    div[data-testid="stDataFrame"] {
        border-radius: 12px;
        overflow: hidden;
        border: 1px solid rgba(255, 255, 255, 0.08);
    }

    /* Buttons */
    .stButton>button {
        border-radius: 10px;
        font-weight: 600;
        transition: all 0.2s ease;
    }

    .stButton>button:hover {
        transform: translateY(-1px);
    }
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)


def render_header():
    """Renders the executive platform header."""
    st.markdown(
        """
        <div class="app-header">
            <div class="app-title">🔮 FORESIGHT Intelligence</div>
            <div class="app-subtitle">
                Enterprise AI-Powered Demand Forecasting & Automated Multi-Echelon Inventory Optimization
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_kpi_card(
    title: str,
    value: str,
    subtext: str,
    icon: str = "📊",
    card_type: str = "info"
) -> str:
    """Returns HTML for an executive KPI metric card."""
    return f"""
    <div class="kpi-card {card_type}">
        <div class="kpi-title">
            <span>{icon}</span> {title}
        </div>
        <div class="kpi-value">{value}</div>
        <div class="kpi-subtext">{subtext}</div>
    </div>
    """


def plot_historical_and_forecast(
    historical_df: pd.DataFrame,
    forecast_df: pd.DataFrame,
    sku_info: pd.Series,
    forecast_horizon: int = 30,
) -> go.Figure:
    """
    Builds an interactive Plotly visualization for historical sales vs. 30-day forecast,
    including 95% confidence bounds and inventory threshold reference lines.
    """
    hist_clean = historical_df.sort_values("date").copy()
    hist_clean["date"] = pd.to_datetime(hist_clean["date"])
    
    # Calculate 7-day rolling average for historical smoothing
    hist_clean["rolling_7"] = hist_clean["units_sold"].rolling(7, min_periods=1).mean()

    # Create figure
    fig = go.Figure()

    # 1. Historical Actual Daily Sales
    fig.add_trace(
        go.Scatter(
            x=hist_clean["date"],
            y=hist_clean["units_sold"],
            mode="lines",
            name="Actual Daily Sales",
            line=dict(color="#94A3B8", width=1.3),
            opacity=0.65,
            hovertemplate="<b>Actual Date</b>: %{x|%b %d, %Y}<br><b>Units Sold</b>: %{y} units<extra></extra>",
        )
    )

    # 2. 7-Day Rolling Moving Average Trend
    fig.add_trace(
        go.Scatter(
            x=hist_clean["date"],
            y=hist_clean["rolling_7"],
            mode="lines",
            name="7-Day Moving Avg",
            line=dict(color="#818CF8", width=2.2),
            hovertemplate="<b>7-Day Rolling Avg</b>: %{y:.1f} units<extra></extra>",
        )
    )

    # 3. 95% Confidence Interval Band (Lower & Upper)
    fig.add_trace(
        go.Scatter(
            x=forecast_df["date"],
            y=forecast_df["forecast_upper"],
            mode="lines",
            line=dict(width=0),
            showlegend=False,
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=forecast_df["date"],
            y=forecast_df["forecast_lower"],
            mode="lines",
            line=dict(width=0),
            fill="tonexty",
            fillcolor=COLOR_CONFIDENCE,
            name="95% Confidence Band",
            hovertemplate="<b>Date</b>: %{x|%b %d, %Y}<br><b>95% CI Range</b>: [%{y:.1f} - %{customdata:.1f}]<extra></extra>",
            customdata=forecast_df["forecast_upper"],
        )
    )

    # 4. Forecast Mean Line
    fig.add_trace(
        go.Scatter(
            x=forecast_df["date"],
            y=forecast_df["forecast_mean"],
            mode="lines+markers",
            name=f"{forecast_horizon}-Day AI Forecast",
            line=dict(color=COLOR_FORECAST, width=2.8, dash="solid"),
            marker=dict(size=4.5, color=COLOR_FORECAST),
            hovertemplate="<b>Forecast Date</b>: %{x|%b %d, %Y}<br><b>Predicted Demand</b>: %{y:.1f} units<extra></extra>",
        )
    )

    # 5. Connect Historical to Forecast (Bridge line)
    last_hist_date = hist_clean["date"].iloc[-1]
    last_hist_val = hist_clean["units_sold"].iloc[-1]
    first_fc_date = forecast_df["date"].iloc[0]
    first_fc_val = forecast_df["forecast_mean"].iloc[0]

    fig.add_trace(
        go.Scatter(
            x=[last_hist_date, first_fc_date],
            y=[last_hist_val, first_fc_val],
            mode="lines",
            line=dict(color=COLOR_FORECAST, width=1.5, dash="dot"),
            showlegend=False,
            hoverinfo="skip",
        )
    )

    # Add vertical divider line at forecast start
    fig.add_vline(
        x=last_hist_date.timestamp() * 1000,
        line_width=1.5,
        line_dash="dash",
        line_color="#CBD5E1",
        annotation_text="Today / Forecast Start",
        annotation_position="top left",
        annotation_font=dict(color="#94A3B8", size=11),
    )

    # Layout enhancements
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#1E293B",
        plot_bgcolor="#0F172A",
        height=450,
        margin=dict(l=20, r=20, t=40, b=20),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=11, color="#CBD5E1"),
        ),
        xaxis=dict(
            title="",
            showgrid=True,
            gridcolor="rgba(255, 255, 255, 0.05)",
            tickformat="%b %d",
            zeroline=False,
        ),
        yaxis=dict(
            title="Daily Units Sold",
            showgrid=True,
            gridcolor="rgba(255, 255, 255, 0.07)",
            zeroline=False,
        ),
        hovermode="x unified",
    )

    return fig


def plot_inventory_health_donut(summary_df: pd.DataFrame) -> go.Figure:
    """
    Renders an interactive Donut Chart of Inventory Health & Risk status.
    """
    status_counts = summary_df["risk_status"].value_counts().reset_index()
    status_counts.columns = ["risk_status", "count"]

    color_map = {
        "Stockout Risk": COLOR_STOCKOUT,
        "Overstock Risk": COLOR_OVERSTOCK,
        "Healthy": COLOR_HEALTHY,
    }

    colors = [color_map.get(status, "#6366F1") for status in status_counts["risk_status"]]

    fig = go.Figure(
        data=[
            go.Pie(
                labels=status_counts["risk_status"],
                values=status_counts["count"],
                hole=0.62,
                marker=dict(colors=colors, line=dict(color="#1E293B", width=2)),
                textinfo="label+percent",
                textposition="outside",
                insidetextorientation="radial",
                hovertemplate="<b>%{label}</b><br>SKUs: %{value} (%{percent})<extra></extra>",
            )
        ]
    )

    total_skus = len(summary_df)
    stockout_count = int((summary_df["risk_status"] == "Stockout Risk").sum())

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#1E293B",
        plot_bgcolor="#1E293B",
        height=320,
        margin=dict(l=20, r=20, t=20, b=20),
        showlegend=False,
        annotations=[
            dict(
                text=f"<b>{total_skus}</b><br><span style='font-size:11px;color:#94A3B8;'>Total SKUs</span>",
                x=0.5,
                y=0.5,
                font=dict(size=18, color="#F8FAFC"),
                showarrow=False,
            )
        ],
    )
    return fig


def plot_category_risk_breakdown(summary_df: pd.DataFrame) -> go.Figure:
    """
    Renders a grouped or stacked horizontal bar chart of Risk Distribution by Category.
    """
    cat_risk = summary_df.groupby(["category", "risk_status"]).size().reset_index(name="count")

    color_discrete_map = {
        "Stockout Risk": COLOR_STOCKOUT,
        "Healthy": COLOR_HEALTHY,
        "Overstock Risk": COLOR_OVERSTOCK,
    }

    fig = px.bar(
        cat_risk,
        x="count",
        y="category",
        color="risk_status",
        orientation="h",
        barmode="stack",
        color_discrete_map=color_discrete_map,
        labels={"count": "Number of SKUs", "category": "", "risk_status": "Risk Status"},
    )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#1E293B",
        plot_bgcolor="#0F172A",
        height=320,
        margin=dict(l=10, r=20, t=20, b=20),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=11, color="#CBD5E1"),
        ),
        xaxis=dict(
            showgrid=True,
            gridcolor="rgba(255, 255, 255, 0.07)",
            dtick=1,
        ),
        yaxis=dict(showgrid=False),
    )
    return fig


def plot_sku_inventory_gauge(sku_row: pd.Series) -> go.Figure:
    """
    Renders a bullet/gauge comparison chart showing Current Inventory vs.
    Reorder Point (ROP), Safety Stock, and Overstock Threshold for a single SKU.
    """
    current_inv = sku_row["current_inventory"]
    rop = sku_row["reorder_point"]
    overstock = sku_row["overstock_threshold"]
    safety_stock = sku_row["safety_stock"]
    max_range = max(overstock * 1.25, current_inv * 1.15, rop * 2.5)

    bar_color = COLOR_STOCKOUT if current_inv <= rop else (COLOR_OVERSTOCK if current_inv > overstock else COLOR_HEALTHY)

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number+delta",
            value=current_inv,
            domain={"x": [0, 1], "y": [0, 1]},
            title={"text": f"<b>Inventory Health Gauge</b><br><span style='font-size:12px;color:#94A3B8;'>{sku_row['sku_id']} • {sku_row['sku_name']}</span>", "font": {"size": 14, "color": "#F8FAFC"}},
            delta={"reference": rop, "increasing": {"color": "#34D399"}, "decreasing": {"color": "#F87171"}},
            gauge={
                "axis": {"range": [0, max_range], "tickwidth": 1, "tickcolor": "#94A3B8"},
                "bar": {"color": bar_color, "thickness": 0.35},
                "bgcolor": "#0F172A",
                "borderwidth": 1,
                "bordercolor": "rgba(255,255,255,0.1)",
                "steps": [
                    {"range": [0, rop], "color": "rgba(239, 68, 68, 0.18)"},
                    {"range": [rop, overstock], "color": "rgba(16, 185, 129, 0.15)"},
                    {"range": [overstock, max_range], "color": "rgba(245, 158, 11, 0.18)"},
                ],
                "threshold": {
                    "line": {"color": "#F43F5E", "width": 3},
                    "thickness": 0.8,
                    "value": rop,
                },
            },
        )
    )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#1E293B",
        height=260,
        margin=dict(l=30, r=30, t=50, b=20),
    )
    return fig
