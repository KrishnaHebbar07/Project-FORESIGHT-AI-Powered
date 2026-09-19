"""
Automated Verification Test Suite for Project FORESIGHT
"""
import sys
import io
import pandas as pd
import numpy as np

# Ensure UTF-8 stdout if available
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

from data_generator import generate_synthetic_sales_data
from forecasting_engine import analyze_inventory_metrics, calculate_portfolio_kpis
from ui_components import (
    plot_historical_and_forecast,
    plot_inventory_health_donut,
    plot_category_risk_breakdown,
    plot_sku_inventory_gauge,
)


def run_tests():
    print("--------------------------------------------------")
    print("Starting Project FORESIGHT Test Suite...")
    print("--------------------------------------------------")

    # 1. Test Data Generator
    df = generate_synthetic_sales_data(num_days=365, seed=42)
    print(f"Dataset generated: {df.shape[0]} rows, {df['sku_id'].nunique()} unique SKUs")
    assert df.shape[0] == 365 * 12, "Row count mismatch"
    required_cols = {"date", "sku_id", "category", "units_sold", "unit_price", "lead_time_days", "current_inventory"}
    assert required_cols.issubset(df.columns), "Missing required columns"
    assert df["units_sold"].min() >= 0, "Negative sales detected"
    print("[PASS] Module 1: Synthetic Data Generation Module")

    # 2. Test Forecasting & Inventory Engine (ML Trend)
    summary_df_ml, forecasts_ml = analyze_inventory_metrics(
        df_sales=df,
        service_level=0.95,
        forecast_model="ML Trend & Seasonality",
        forecast_horizon=30,
    )
    assert len(summary_df_ml) == 12, "Summary SKU count mismatch"
    assert len(forecasts_ml) == 12, "Forecasts dict count mismatch"
    for sku_id, fc in forecasts_ml.items():
        assert len(fc) == 30, f"Forecast horizon mismatch for {sku_id}"
        assert set(["date", "forecast_mean", "forecast_lower", "forecast_upper"]).issubset(fc.columns)
        assert (fc["forecast_mean"] >= 0).all(), "Negative forecast mean detected"
    print("[PASS] Module 2A: ML Demand Forecasting Engine (Ridge + Harmonics)")

    # 3. Test Forecasting & Inventory Engine (Rolling Average)
    summary_df_roll, forecasts_roll = analyze_inventory_metrics(
        df_sales=df,
        service_level=0.95,
        forecast_model="Rolling Weighted Average",
        forecast_horizon=30,
    )
    assert len(summary_df_roll) == 12, "Summary SKU count mismatch"
    print("[PASS] Module 2B: Rolling Average Statistical Demand Forecast Engine")

    # 4. Test ROP, Safety Stock, and Risk Flagging
    for _, row in summary_df_ml.iterrows():
        # Formula verification
        lead_time = row["lead_time_days"]
        avg_d = row["avg_daily_demand"]
        std_d = row["std_daily_demand"]
        expected_ss = np.ceil(1.645 * std_d * np.sqrt(lead_time))
        assert abs(row["safety_stock"] - expected_ss) <= 1.0, f"Safety stock formula check failed for {row['sku_id']}"

        # Risk Classification verification
        curr = row["current_inventory"]
        rop = row["reorder_point"]
        overstock = row["overstock_threshold"]
        if curr <= rop:
            assert row["risk_status"] == "Stockout Risk", f"Stockout risk classification failed for {row['sku_id']}"
        elif curr > overstock:
            assert row["risk_status"] == "Overstock Risk", f"Overstock risk classification failed for {row['sku_id']}"
        else:
            assert row["risk_status"] == "Healthy", f"Healthy classification failed for {row['sku_id']}"

    stockout_count = (summary_df_ml["risk_status"] == "Stockout Risk").sum()
    overstock_count = (summary_df_ml["risk_status"] == "Overstock Risk").sum()
    healthy_count = (summary_df_ml["risk_status"] == "Healthy").sum()
    print(f"Risk breakdown across 12 SKUs: Stockout={stockout_count}, Healthy={healthy_count}, Overstock={overstock_count}")
    assert stockout_count > 0 and overstock_count > 0 and healthy_count > 0, "Expected all 3 risk states represented"
    print("[PASS] Module 2C: ROP, Safety Stock, and Automated Risk Flagging Engine")

    # 5. Test Portfolio KPIs
    kpis = calculate_portfolio_kpis(summary_df_ml)
    assert kpis["total_skus"] == 12
    assert kpis["total_inv_value"] > 0
    assert kpis["total_lost_sales_projected"] > 0
    assert kpis["health_percentage"] > 0
    print(f"Portfolio KPIs: Total Value=${kpis['total_inv_value']:,.2f}, Projected Lost Sales=${kpis['total_lost_sales_projected']:,.2f}")
    print("[PASS] Module 2D: Portfolio KPI Calculations")

    # 6. Test UI & Visualization Generators
    sample_sku_id = summary_df_ml["sku_id"].iloc[0]
    sample_sku_row = summary_df_ml.iloc[0]
    sample_hist = df[df["sku_id"] == sample_sku_id]
    sample_fc = forecasts_ml[sample_sku_id]

    fig_ts = plot_historical_and_forecast(sample_hist, sample_fc, sample_sku_row)
    assert fig_ts is not None, "Time series chart generation failed"

    fig_donut = plot_inventory_health_donut(summary_df_ml)
    assert fig_donut is not None, "Donut chart generation failed"

    fig_cat = plot_category_risk_breakdown(summary_df_ml)
    assert fig_cat is not None, "Category chart generation failed"

    fig_gauge = plot_sku_inventory_gauge(sample_sku_row)
    assert fig_gauge is not None, "Gauge chart generation failed"
    print("[PASS] Module 3: UI Plotly Visualization Engine")

    print("--------------------------------------------------")
    print("ALL TESTS PASSED SUCCESSFULLY! PROJECT FORESIGHT READY.")
    print("--------------------------------------------------")


if __name__ == "__main__":
    run_tests()
