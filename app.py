"""
================================================================================
Project FORESIGHT – AI-Powered Demand & Inventory Intelligence Platform
================================================================================
A complete, production-ready Streamlit prototype that integrates synthetic data
generation, statistical & machine-learning demand forecasting, Reorder Point (ROP)
and Safety Stock optimization, automated risk classification, and interactive
executive visualizations with actionable procurement recommendations.

Run locally:
    streamlit run app.py
================================================================================
"""

import datetime
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import streamlit as st

# Internal platform modules
from data_generator import generate_synthetic_sales_data
from forecasting_engine import (
    analyze_inventory_metrics,
    calculate_portfolio_kpis,
    get_z_score_for_service_level,
)
from ui_components import (
    inject_custom_css,
    render_header,
    render_kpi_card,
    plot_historical_and_forecast,
    plot_inventory_health_donut,
    plot_category_risk_breakdown,
    plot_sku_inventory_gauge,
    COLOR_STOCKOUT,
    COLOR_OVERSTOCK,
    COLOR_HEALTHY,
)

# -----------------------------------------------------------------------------
# 1. STREAMLIT PAGE CONFIGURATION
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Project FORESIGHT – Demand & Inventory Intelligence",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Inject modern custom stylesheet
inject_custom_css()


# -----------------------------------------------------------------------------
# 2. STATE MANAGEMENT & DATA CACHING
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_cached_sales_data(seed: int = 42, num_days: int = 365) -> pd.DataFrame:
    """Generates and caches historical synthetic sales data."""
    return generate_synthetic_sales_data(num_days=num_days, seed=seed)


# Initialize session state for deterministic seed
if "random_seed" not in st.session_state:
    st.session_state["random_seed"] = 42


# -----------------------------------------------------------------------------
# 3. SIDEBAR CONTROLS & FILTERS
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚙️ **FORESIGHT Controls**")
    st.caption("Configure forecast parameters, service levels, and view filters.")
    st.markdown("---")

    # Dataset Generation Controls
    st.markdown("#### 🎲 **Data Simulation**")
    seed_input = st.number_input(
        "Simulation Seed",
        min_value=1,
        max_value=9999,
        value=st.session_state["random_seed"],
        step=1,
        help="Change seed to simulate different market conditions and inventory scenarios.",
    )
    if seed_input != st.session_state["random_seed"]:
        st.session_state["random_seed"] = seed_input
        st.rerun()

    if st.button("🔄 Regenerate Market Data", use_container_width=True):
        st.session_state["random_seed"] = np.random.randint(1, 9999)
        st.rerun()

    st.markdown("---")

    # Forecasting Engine Configuration
    st.markdown("#### 🧠 **Forecasting Model Engine**")
    forecast_model = st.selectbox(
        "Algorithm",
        options=["ML Trend & Seasonality (Ridge)", "Rolling Weighted Average"],
        index=0,
        help="Select between regularized ML regression (with trend & Fourier harmonics) or rolling weighted average.",
    )
    # Map friendly display name to engine parameter
    engine_model_param = "ML Trend & Seasonality" if "Ridge" in forecast_model else "Rolling Weighted Average"

    forecast_horizon = st.slider(
        "Forecast Horizon (Days)",
        min_value=7,
        max_value=90,
        value=30,
        step=1,
        help="Number of future days to project forward.",
    )

    service_level = st.select_slider(
        "Target Service Level (Safety Stock)",
        options=[0.85, 0.90, 0.95, 0.98, 0.99, 0.999],
        value=0.95,
        format_func=lambda x: f"{int(x * 100)}%" if x < 0.995 else f"{x*100:.1f}%",
        help="Higher service level increases safety buffer Z-factor to mitigate stockouts.",
    )
    current_z = get_z_score_for_service_level(service_level)
    st.caption(f"ℹ️ Derived **Z-Score**: `{current_z:.3f}` standard deviations")

    st.markdown("---")

    # Portfolio Filter Controls
    st.markdown("#### 🔍 **Portfolio Filters**")
    # Load dataset to extract available categories
    df_raw = load_cached_sales_data(seed=st.session_state["random_seed"])
    available_categories = sorted(df_raw["category"].unique())

    selected_categories = st.multiselect(
        "Filter Categories",
        options=available_categories,
        default=available_categories,
        help="Filter SKUs by one or more merchandise categories.",
    )

    available_risks = ["Stockout Risk", "Healthy", "Overstock Risk"]
    selected_risks = st.multiselect(
        "Filter Risk Status",
        options=available_risks,
        default=available_risks,
        help="Filter inventory items by risk classification.",
    )

    st.markdown("---")
    st.markdown(
        """
        <div style="font-size:0.75rem; color:#64748B; text-align:center; margin-top:20px;">
            Project FORESIGHT Intelligence v2.4<br>
            Powered by Python • Streamlit • Scikit-Learn • Plotly
        </div>
        """,
        unsafe_allow_html=True,
    )


# -----------------------------------------------------------------------------
# 4. DATA PROCESSING & INVENTORY ENGINE COMPUTATION
# -----------------------------------------------------------------------------
# Run inventory and forecasting engine
summary_df, forecasts_dict = analyze_inventory_metrics(
    df_sales=df_raw,
    service_level=service_level,
    forecast_model=engine_model_param,
    forecast_horizon=forecast_horizon,
)

# Apply Sidebar Filters
if not selected_categories:
    selected_categories = available_categories
if not selected_risks:
    selected_risks = available_risks

filtered_summary_df = summary_df[
    (summary_df["category"].isin(selected_categories)) &
    (summary_df["risk_status"].isin(selected_risks))
].copy()

# Calculate Portfolio KPIs on the filtered slice (or total)
portfolio_kpis = calculate_portfolio_kpis(filtered_summary_df if len(filtered_summary_df) > 0 else summary_df)


# -----------------------------------------------------------------------------
# 5. DASHBOARD MAIN VIEW
# -----------------------------------------------------------------------------

# Render Platform Header
render_header()

# Overview KPI Cards
st.markdown("### 📊 **Executive Overview & KPI Intelligence**")

kpi_col1, kpi_col2, kpi_col3, kpi_col4, kpi_col5 = st.columns(5)

with kpi_col1:
    st.markdown(
        render_kpi_card(
            title="Total Active SKUs",
            value=f"{portfolio_kpis['total_skus']}",
            subtext=f"Across {len(selected_categories)} active categories",
            icon="📦",
            card_type="info",
        ),
        unsafe_allow_html=True,
    )

with kpi_col2:
    st.markdown(
        render_kpi_card(
            title="Stockout Risk",
            value=f"{portfolio_kpis['stockout_skus']} SKUs",
            subtext=f"{(portfolio_kpis['stockout_skus']/max(1, portfolio_kpis['total_skus'])*100):.1f}% of active portfolio",
            icon="🚨",
            card_type="danger" if portfolio_kpis["stockout_skus"] > 0 else "success",
        ),
        unsafe_allow_html=True,
    )

with kpi_col3:
    st.markdown(
        render_kpi_card(
            title="Total Inventory Value",
            value=f"${portfolio_kpis['total_inv_value']:,.0f}",
            subtext=f"Cost Basis: ${portfolio_kpis['total_inv_cost']:,.0f}",
            icon="💰",
            card_type="info",
        ),
        unsafe_allow_html=True,
    )

with kpi_col4:
    st.markdown(
        render_kpi_card(
            title="Projected Lost Sales",
            value=f"${portfolio_kpis['total_lost_sales_projected']:,.0f}",
            subtext="Revenue at risk from stockouts",
            icon="📉",
            card_type="danger" if portfolio_kpis["total_lost_sales_projected"] > 0 else "success",
        ),
        unsafe_allow_html=True,
    )

with kpi_col5:
    st.markdown(
        render_kpi_card(
            title="Capital In Overstock",
            value=f"${portfolio_kpis['total_capital_overstock']:,.0f}",
            subtext=f"{portfolio_kpis['overstock_skus']} SKUs with > 90d supply",
            icon="⚠️",
            card_type="warning" if portfolio_kpis["overstock_skus"] > 0 else "success",
        ),
        unsafe_allow_html=True,
    )

st.markdown("<div style='margin-bottom: 24px;'></div>", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# 6. INVENTORY HEALTH & RISK DISTRIBUTION (ROW 2)
# -----------------------------------------------------------------------------
chart_col1, chart_col2 = st.columns([1, 1])

with chart_col1:
    st.markdown("#### 🍩 **Portfolio Health Distribution**")
    st.caption("Distribution of active SKUs across automated risk classifications.")
    donut_fig = plot_inventory_health_donut(filtered_summary_df if len(filtered_summary_df) > 0 else summary_df)
    st.plotly_chart(donut_fig, use_container_width=True)

with chart_col2:
    st.markdown("#### 🏢 **Category-Level Risk Exposure**")
    st.caption("Breakdown of stockout, healthy, and overstock risk per merchandise group.")
    cat_fig = plot_category_risk_breakdown(filtered_summary_df if len(filtered_summary_df) > 0 else summary_df)
    st.plotly_chart(cat_fig, use_container_width=True)

st.markdown("<div style='margin-bottom: 24px;'></div>", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# 7. SKU DEEP-DIVE & 30-DAY AI DEMAND FORECAST (ROW 3)
# -----------------------------------------------------------------------------
st.markdown("### 📈 **SKU Deep-Dive & AI Demand Forecast Studio**")

# Prepare SKU Selector with risk tags
sku_options = []
sku_id_map = {}
for _, row in summary_df.iterrows():
    label = f"{row['sku_id']} - {row['sku_name']} [{row['risk_status'].upper()}]"
    sku_options.append(label)
    sku_id_map[label] = row["sku_id"]

# Default to first stockout item or first SKU
default_index = 0
for idx, label in enumerate(sku_options):
    if "STOCKOUT" in label:
        default_index = idx
        break

selected_label = st.selectbox(
    "Select Target SKU for In-Depth AI Analysis & Demand Projection:",
    options=sku_options,
    index=default_index,
)
selected_sku_id = sku_id_map[selected_label]

# Retrieve SKU Data
sku_row = summary_df[summary_df["sku_id"] == selected_sku_id].iloc[0]
sku_hist_df = df_raw[df_raw["sku_id"] == selected_sku_id]
sku_fc_df = forecasts_dict[selected_sku_id]

deep_col1, deep_col2 = st.columns([65, 35])

with deep_col1:
    st.markdown(f"##### **Historical Sales vs. {forecast_horizon}-Day Predictive Forecast**")
    st.caption(f"Model: **{forecast_model}** • Confidence Interval: **95%** • Daily granularity")
    ts_fig = plot_historical_and_forecast(
        historical_df=sku_hist_df,
        forecast_df=sku_fc_df,
        sku_info=sku_row,
        forecast_horizon=forecast_horizon,
    )
    st.plotly_chart(ts_fig, use_container_width=True)

with deep_col2:
    st.markdown("##### **Inventory Health Diagnostics & Safety Metrics**")
    gauge_fig = plot_sku_inventory_gauge(sku_row)
    st.plotly_chart(gauge_fig, use_container_width=True)

    # Diagnostic Metric Highlights
    badge_class = "badge-stockout" if sku_row["risk_status"] == "Stockout Risk" else (
        "badge-overstock" if sku_row["risk_status"] == "Overstock Risk" else "badge-healthy"
    )

    st.markdown(
        f"""
        <div class="section-card" style="padding:16px; margin-top:-10px;">
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:12px;">
                <span style="font-weight:700; color:#F8FAFC;">{sku_row['sku_id']}</span>
                <span class="badge {badge_class}">{sku_row['risk_status']}</span>
            </div>
            <table style="width:100%; font-size:0.85rem; color:#CBD5E1; border-collapse: collapse;">
                <tr style="border-bottom: 1px solid rgba(255,255,255,0.06); padding: 4px 0;">
                    <td style="color:#94A3B8; padding: 6px 0;">Current Inventory</td>
                    <td style="text-align:right; font-weight:700; color:#F8FAFC;">{sku_row['current_inventory']:,} units</td>
                </tr>
                <tr style="border-bottom: 1px solid rgba(255,255,255,0.06);">
                    <td style="color:#94A3B8; padding: 6px 0;">Reorder Point (ROP)</td>
                    <td style="text-align:right; font-weight:700; color:#F43F5E;">{sku_row['reorder_point']:,} units</td>
                </tr>
                <tr style="border-bottom: 1px solid rgba(255,255,255,0.06);">
                    <td style="color:#94A3B8; padding: 6px 0;">Safety Stock Buffer</td>
                    <td style="text-align:right; font-weight:600;">{sku_row['safety_stock']:,} units</td>
                </tr>
                <tr style="border-bottom: 1px solid rgba(255,255,255,0.06);">
                    <td style="color:#94A3B8; padding: 6px 0;">Supplier Lead Time</td>
                    <td style="text-align:right; font-weight:600;">{sku_row['lead_time_days']} days</td>
                </tr>
                <tr style="border-bottom: 1px solid rgba(255,255,255,0.06);">
                    <td style="color:#94A3B8; padding: 6px 0;">Avg Daily Demand (μ)</td>
                    <td style="text-align:right; font-weight:600;">{sku_row['avg_daily_demand']:.1f} units/day</td>
                </tr>
                <tr style="border-bottom: 1px solid rgba(255,255,255,0.06);">
                    <td style="color:#94A3B8; padding: 6px 0;">Days of Supply (Runway)</td>
                    <td style="text-align:right; font-weight:700; color:{'#EF4444' if sku_row['days_of_supply'] <= sku_row['lead_time_days'] else '#10B981'};">
                        {sku_row['days_of_supply']:.1f} days
                    </td>
                </tr>
                <tr style="border-bottom: 1px solid rgba(255,255,255,0.06);">
                    <td style="color:#94A3B8; padding: 6px 0;">Urgency Level</td>
                    <td style="text-align:right; font-weight:700;">{sku_row['urgency']}</td>
                </tr>
                <tr>
                    <td style="color:#94A3B8; padding: 6px 0;">Recommended Action</td>
                    <td style="text-align:right; font-weight:600; color:#60A5FA;">{sku_row['action']}</td>
                </tr>
            </table>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("<div style='margin-bottom: 24px;'></div>", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# 8. ACTIONABLE RECOMMENDATIONS & PURCHASE ORDER PLAN (ROW 4)
# -----------------------------------------------------------------------------
st.markdown("### 📋 **Actionable Inventory & Procurement Recommendations**")
st.caption(
    "Automated reorder triggers, suggested procurement quantities, and financial impact across filtered SKUs."
)

if len(filtered_summary_df) == 0:
    st.warning("No SKUs match the current sidebar filter criteria. Please adjust your filters.")
else:
    # Prepare Display DataFrame
    display_df = filtered_summary_df[[
        "sku_id",
        "sku_name",
        "category",
        "current_inventory",
        "reorder_point",
        "safety_stock",
        "lead_time_days",
        "days_of_supply",
        "risk_status",
        "urgency",
        "suggested_reorder_qty",
        "estimated_reorder_cost",
        "projected_lost_sales",
        "action",
    ]].copy()

    display_df.rename(
        columns={
            "sku_id": "SKU ID",
            "sku_name": "Product Name",
            "category": "Category",
            "current_inventory": "Current Stock",
            "reorder_point": "ROP",
            "safety_stock": "Safety Stock",
            "lead_time_days": "Lead Time (Days)",
            "days_of_supply": "Runway (Days)",
            "risk_status": "Risk Status",
            "urgency": "Urgency",
            "suggested_reorder_qty": "Suggested Reorder (Units)",
            "estimated_reorder_cost": "Est. Reorder Cost ($)",
            "projected_lost_sales": "Lost Sales Risk ($)",
            "action": "Prescribed Action",
        },
        inplace=True,
    )

    # Style DataFrame with conditional background highlights
    def highlight_risk_row(row):
        status = row["Risk Status"]
        if status == "Stockout Risk":
            return ["background-color: rgba(239, 68, 68, 0.15); color: #FCA5A5;"] * len(row)
        elif status == "Overstock Risk":
            return ["background-color: rgba(245, 158, 11, 0.12); color: #FCD34D;"] * len(row)
        else:
            return ["background-color: rgba(16, 185, 129, 0.10); color: #6EE7B7;"] * len(row)

    styled_table = (
        display_df.style
        .apply(highlight_risk_row, axis=1)
        .format({
            "Current Stock": "{:,}",
            "ROP": "{:,}",
            "Safety Stock": "{:,}",
            "Runway (Days)": "{:.1f}",
            "Suggested Reorder (Units)": "{:,}",
            "Est. Reorder Cost ($)": "${:,.2f}",
            "Lost Sales Risk ($)": "${:,.2f}",
        })
    )

    st.dataframe(styled_table, use_container_width=True, height=360)

    # Action Toolbar: Download CSV and Quick Stats
    col_act1, col_act2, col_act3 = st.columns([1, 1, 2])

    with col_act1:
        # Prepare CSV for download
        csv_data = filtered_summary_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Reorder Plan (CSV)",
            data=csv_data,
            file_name=f"foresight_inventory_plan_{datetime.date.today().strftime('%Y%m%d')}.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with col_act2:
        reorder_items_count = int((filtered_summary_df["suggested_reorder_qty"] > 0).sum())
        total_reorder_spend = filtered_summary_df["estimated_reorder_cost"].sum()
        st.info(f"🛒 **{reorder_items_count} SKUs** need purchase orders (Total: **${total_reorder_spend:,.2f}**)")

st.markdown("<div style='margin-bottom: 28px;'></div>", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# 9. PLATFORM FORMULATIONS & ARCHITECTURE NOTES
# -----------------------------------------------------------------------------
with st.expander("🔬 **Mathematical Formulations & Engine Documentation**", expanded=False):
    st.markdown(
        r"""
        #### 1. Reorder Point (ROP) & Safety Stock Formulation
        $$\text{Reorder Point (ROP)} = (\mu_d \times L) + \text{Safety Stock}$$
        $$\text{Safety Stock} = Z \times \sigma_d \times \sqrt{L}$$
        - **$\mu_d$ (Average Daily Demand)**: Baseline consumption rate per day computed over empirical and predicted demand.
        - **$L$ (Lead Time)**: Supplier transit and fulfillment lead time in days.
        - **$\sigma_d$ (Standard Deviation of Daily Demand)**: Daily demand volatility metric.
        - **$Z$ (Service Level Factor)**: Standard normal inverse cumulative density function value for the specified fill-rate service level (e.g. $Z=1.645$ for 95% service level).

        #### 2. Automated Inventory Risk Logic
        - 🔴 **Stockout Risk**: Triggered when $\text{Current Inventory} \le \text{ROP}$. Urgent purchase order required before lead time elapses.
        - 🟡 **Overstock Risk**: Triggered when $\text{Current Inventory} > 3 \times \text{Average Monthly Demand}$ (i.e. $> 90 \times \mu_d$). Excess working capital locked.
        - 🟢 **Healthy**: $\text{ROP} < \text{Current Inventory} \le 3 \times \text{Average Monthly Demand}$. Optimal inventory turnover.

        #### 3. Actionable Procurement Calculations
        - **Days of Supply (Runway)**: $\text{Current Inventory} / \mu_d$
        - **Suggested Reorder Quantity**: $\max\Big(0, \ \big(\text{ROP} + 30 \cdot \mu_d\big) - \text{Current Inventory}\Big)$
        - **Projected Lost Sales**: $\max\big(0, \ \text{ROP} - \text{Current Inventory}\big) \times \text{Unit Price}$
        - **Capital Locked in Overstock**: $\max\Big(0, \ \text{Current Inventory} - \big(\text{ROP} + 30 \cdot \mu_d\big)\Big) \times \text{Unit Cost}$
        """
    )

with st.expander("🗄️ **Raw Historical Sales Data Explorer (Last 365 Days)**", expanded=False):
    st.dataframe(df_raw.tail(200), use_container_width=True)
