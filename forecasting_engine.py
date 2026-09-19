"""
Project FORESIGHT: Demand Forecasting & Inventory Engine
--------------------------------------------------------
Implements statistical and machine-learning demand forecasting models,
computes Reorder Points (ROP), Safety Stock, and executes automated inventory
risk classification (Stockout, Overstock, Healthy) with actionable insights.
"""

from typing import Dict, List, Optional, Tuple
import datetime
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import Ridge, LinearRegression


# Service level to standard normal Z-score mapping
SERVICE_LEVEL_Z_MAP: Dict[float, float] = {
    0.85: 1.036,
    0.90: 1.282,
    0.95: 1.645,
    0.98: 2.054,
    0.99: 2.326,
    0.999: 3.090,
}


def get_z_score_for_service_level(service_level: float = 0.95) -> float:
    """Returns the normal distribution quantile (Z-score) for a service level."""
    if service_level in SERVICE_LEVEL_Z_MAP:
        return SERVICE_LEVEL_Z_MAP[service_level]
    try:
        return float(stats.norm.ppf(service_level))
    except Exception:
        return 1.645


def forecast_sku_demand_ml(
    historical_df: pd.DataFrame,
    forecast_horizon: int = 30,
    confidence_level: float = 0.95,
) -> pd.DataFrame:
    """
    Generates a 30-day demand forecast for a single SKU using Scikit-Learn ML
    (Ridge Regression with trend, day-of-week seasonality, and cyclical annual terms).

    Parameters:
    -----------
    historical_df : pd.DataFrame
        Time series for one SKU with 'date' and 'units_sold' columns.
    forecast_horizon : int
        Number of forward days to forecast (default 30).
    confidence_level : float
        Confidence interval percentage (e.g. 0.95).

    Returns:
    --------
    pd.DataFrame:
        Columns: [date, forecast_mean, forecast_lower, forecast_upper, is_forecast]
    """
    df = historical_df.sort_values("date").copy()
    df["date"] = pd.to_datetime(df["date"])
    
    n_hist = len(df)
    if n_hist < 14:
        raise ValueError("Insufficient historical observations for forecasting (min 14 days required).")

    # Feature Engineering
    # 1. Normalized Time Index
    t_hist = np.arange(n_hist).reshape(-1, 1)
    
    # 2. Day of Week features (Sine / Cosine harmonic encoding)
    dow_hist = df["date"].dt.dayofweek.to_numpy()
    dow_sin = np.sin(2 * np.pi * dow_hist / 7).reshape(-1, 1)
    dow_cos = np.cos(2 * np.pi * dow_hist / 7).reshape(-1, 1)

    # 3. Day of Year (Cyclical annual seasonality)
    doy_hist = df["date"].dt.dayofyear.to_numpy()
    doy_sin = np.sin(2 * np.pi * doy_hist / 365.25).reshape(-1, 1)
    doy_cos = np.cos(2 * np.pi * doy_hist / 365.25).reshape(-1, 1)

    # 4. Rolling 7-day trend signal
    rolling_7 = df["units_sold"].rolling(window=7, min_periods=1).mean().to_numpy().reshape(-1, 1)

    X_train = np.hstack([t_hist, dow_sin, dow_cos, doy_sin, doy_cos, rolling_7])
    y_train = df["units_sold"].to_numpy()

    # Fit Regularized Model
    model = Ridge(alpha=1.0)
    model.fit(X_train, y_train)

    # In-sample residuals for prediction interval estimation
    y_pred_train = model.predict(X_train)
    residuals = y_train - y_pred_train
    residual_std = np.std(residuals, ddof=2) if len(residuals) > 2 else np.std(y_train)

    # Generate Future Date Range & Future Features
    last_date = df["date"].max()
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_horizon, freq="D")
    
    t_future = np.arange(n_hist, n_hist + forecast_horizon).reshape(-1, 1)
    dow_future = future_dates.dayofweek.to_numpy()
    dow_sin_f = np.sin(2 * np.pi * dow_future / 7).reshape(-1, 1)
    dow_cos_f = np.cos(2 * np.pi * dow_future / 7).reshape(-1, 1)
    doy_future = future_dates.dayofyear.to_numpy()
    doy_sin_f = np.sin(2 * np.pi * doy_future / 365.25).reshape(-1, 1)
    doy_cos_f = np.cos(2 * np.pi * doy_future / 365.25).reshape(-1, 1)

    # Propagate rolling baseline forward
    recent_mean_7 = float(df["units_sold"].tail(7).mean())
    rolling_7_f = np.full((forecast_horizon, 1), recent_mean_7)

    X_future = np.hstack([t_future, dow_sin_f, dow_cos_f, doy_sin_f, doy_cos_f, rolling_7_f])
    y_pred_future = model.predict(X_future)
    y_pred_future = np.clip(y_pred_future, a_min=0, a_max=None)

    # Critical Value for Prediction Interval
    z_val = stats.norm.ppf((1 + confidence_level) / 2)
    # Variance expands slightly further into the future
    horizon_factor = np.sqrt(1 + (np.arange(1, forecast_horizon + 1) / forecast_horizon) * 0.35)
    margin = z_val * residual_std * horizon_factor

    lower_bound = np.clip(y_pred_future - margin, a_min=0, a_max=None)
    upper_bound = y_pred_future + margin

    forecast_df = pd.DataFrame({
        "date": future_dates,
        "forecast_mean": np.round(y_pred_future, 2),
        "forecast_lower": np.round(lower_bound, 2),
        "forecast_upper": np.round(upper_bound, 2),
        "is_forecast": True,
    })

    return forecast_df


def forecast_sku_demand_rolling(
    historical_df: pd.DataFrame,
    forecast_horizon: int = 30,
    window_days: int = 14,
    confidence_level: float = 0.95,
) -> pd.DataFrame:
    """
    Generates a 30-day forecast using a day-of-week adjusted weighted moving average.
    """
    df = historical_df.sort_values("date").copy()
    df["date"] = pd.to_datetime(df["date"])

    recent_df = df.tail(max(window_days, 28)).copy()
    dow_means = recent_df.groupby(recent_df["date"].dt.dayofweek)["units_sold"].mean()
    overall_mean = recent_df["units_sold"].mean()
    dow_multipliers = (dow_means / overall_mean).to_dict() if overall_mean > 0 else {}

    std_dev = recent_df["units_sold"].std(ddof=1) if len(recent_df) > 1 else 1.0

    last_date = df["date"].max()
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_horizon, freq="D")
    
    # 14-day exponential weighted average for baseline
    weights = np.exp(np.linspace(-1, 0, min(window_days, len(df))))
    weights /= weights.sum()
    baseline = np.dot(df["units_sold"].tail(len(weights)), weights)

    forecast_vals = []
    for d in future_dates:
        dow = d.dayofweek
        mult = dow_multipliers.get(dow, 1.0)
        forecast_vals.append(max(0.0, baseline * mult))

    forecast_vals = np.array(forecast_vals)
    z_val = stats.norm.ppf((1 + confidence_level) / 2)
    margin = z_val * std_dev

    forecast_df = pd.DataFrame({
        "date": future_dates,
        "forecast_mean": np.round(forecast_vals, 2),
        "forecast_lower": np.round(np.clip(forecast_vals - margin, a_min=0, a_max=None), 2),
        "forecast_upper": np.round(forecast_vals + margin, 2),
        "is_forecast": True,
    })

    return forecast_df


def analyze_inventory_metrics(
    df_sales: pd.DataFrame,
    service_level: float = 0.95,
    forecast_model: str = "ML Trend & Seasonality",
    forecast_horizon: int = 30,
) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """
    Computes comprehensive inventory KPIs, Reorder Points, Risk States,
    and SKU-level forecasts.

    Parameters:
    -----------
    df_sales : pd.DataFrame
        Historical sales dataset.
    service_level : float
        Target service level for safety stock (e.g. 0.95).
    forecast_model : str
        Choice of forecasting engine ("ML Trend & Seasonality" or "Rolling Weighted Average").
    forecast_horizon : int
        Horizon days to project forward (default 30).

    Returns:
    --------
    Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
        - summary_df: Summary metrics per SKU including ROP, risk, lost sales, etc.
        - forecasts_dict: Dictionary mapping sku_id -> forecast DataFrame.
    """
    z_score = get_z_score_for_service_level(service_level)
    sku_summaries = []
    forecasts_dict = {}

    grouped = df_sales.groupby("sku_id")

    for sku_id, group in grouped:
        group_sorted = group.sort_values("date")
        sku_name = group_sorted["sku_name"].iloc[0] if "sku_name" in group_sorted else sku_id
        category = group_sorted["category"].iloc[0]
        unit_price = float(group_sorted["unit_price"].iloc[-1])
        unit_cost = float(group_sorted["unit_cost"].iloc[-1]) if "unit_cost" in group_sorted else round(unit_price * 0.5, 2)
        lead_time = int(group_sorted["lead_time_days"].iloc[-1])
        current_inv = int(group_sorted["current_inventory"].iloc[-1])

        # Generate Forecast
        if forecast_model == "Rolling Weighted Average":
            fc_df = forecast_sku_demand_rolling(group_sorted, forecast_horizon=forecast_horizon)
        else:
            fc_df = forecast_sku_demand_ml(group_sorted, forecast_horizon=forecast_horizon)
        
        forecasts_dict[sku_id] = fc_df

        # Statistical Metrics over recent 60-day baseline
        recent_sales = group_sorted["units_sold"].tail(60)
        avg_daily_demand = float(recent_sales.mean())
        std_daily_demand = float(recent_sales.std(ddof=1)) if len(recent_sales) > 1 else float(group_sorted["units_sold"].std())
        if np.isnan(std_daily_demand) or std_daily_demand <= 0:
            std_daily_demand = max(1.0, avg_daily_demand * 0.20)

        # 30-Day Forecast Demand
        forecast_30d_total = float(fc_df["forecast_mean"].head(30).sum())
        forecast_daily_avg = float(fc_df["forecast_mean"].head(30).mean())

        # Inventory Formulas:
        # Lead Time Demand = Average Daily Demand * Lead Time
        lead_time_demand = avg_daily_demand * lead_time
        
        # Safety Stock = Z * std_d * sqrt(Lead Time)
        safety_stock = z_score * std_daily_demand * np.sqrt(lead_time)
        
        # Reorder Point (ROP) = (Average Daily Demand * Lead Time) + Safety Stock
        rop = lead_time_demand + safety_stock
        
        # Monthly Demand & Overstock Threshold (3x Monthly Demand)
        avg_monthly_demand = avg_daily_demand * 30.0
        overstock_threshold = 3.0 * avg_monthly_demand  # 90 days of demand

        # Target Stock Level (Order-up-to level for a 30-day review cycle)
        target_stock = rop + (avg_daily_demand * 30.0)

        # Risk Classification
        if current_inv <= rop:
            risk_status = "Stockout Risk"
            risk_color = "#EF4444"  # Red
        elif current_inv > overstock_threshold:
            risk_status = "Overstock Risk"
            risk_color = "#F59E0B"  # Amber / Yellow
        else:
            risk_status = "Healthy"
            risk_color = "#10B981"  # Emerald Green

        # Days of Inventory Remaining (DIR / Runway)
        days_of_supply = current_inv / avg_daily_demand if avg_daily_demand > 0 else 999.0

        # Urgency Level & Suggested Action
        if risk_status == "Stockout Risk":
            if days_of_supply <= (lead_time * 0.5):
                urgency = "🚨 Critical"
                action = "Express Expedite Reorder"
            elif days_of_supply <= lead_time:
                urgency = "⚠️ High"
                action = "Place Immediate Reorder"
            else:
                urgency = "🟡 Moderate"
                action = "Standard Reorder Recommended"
            suggested_reorder_qty = max(0, int(np.ceil(target_stock - current_inv)))
            # Shortfall during replenishment lead time
            shortfall_units = max(0, int(np.ceil(rop - current_inv)))
            projected_lost_sales = shortfall_units * unit_price
            capital_tied_overstock = 0.0
        elif risk_status == "Overstock Risk":
            urgency = "🟣 Excess"
            action = "Promotional Clearance / Markdown"
            suggested_reorder_qty = 0
            projected_lost_sales = 0.0
            excess_units = max(0, current_inv - int(target_stock))
            capital_tied_overstock = excess_units * unit_cost
        else:
            urgency = "🟢 Optimal"
            action = "Monitor Inventory Levels"
            suggested_reorder_qty = 0
            projected_lost_sales = 0.0
            capital_tied_overstock = 0.0

        current_inventory_value = current_inv * unit_price
        current_inventory_cost = current_inv * unit_cost
        estimated_reorder_cost = suggested_reorder_qty * unit_cost

        sku_summaries.append({
            "sku_id": sku_id,
            "sku_name": sku_name,
            "category": category,
            "unit_price": unit_price,
            "unit_cost": unit_cost,
            "lead_time_days": lead_time,
            "current_inventory": current_inv,
            "avg_daily_demand": round(avg_daily_demand, 2),
            "std_daily_demand": round(std_daily_demand, 2),
            "safety_stock": int(np.ceil(safety_stock)),
            "reorder_point": int(np.ceil(rop)),
            "monthly_demand": round(avg_monthly_demand, 1),
            "overstock_threshold": int(np.ceil(overstock_threshold)),
            "forecast_30d_units": int(np.round(forecast_30d_total)),
            "days_of_supply": round(days_of_supply, 1),
            "risk_status": risk_status,
            "risk_color": risk_color,
            "urgency": urgency,
            "action": action,
            "suggested_reorder_qty": suggested_reorder_qty,
            "projected_lost_sales": round(projected_lost_sales, 2),
            "capital_tied_overstock": round(capital_tied_overstock, 2),
            "current_inventory_value": round(current_inventory_value, 2),
            "current_inventory_cost": round(current_inventory_cost, 2),
            "estimated_reorder_cost": round(estimated_reorder_cost, 2),
        })

    summary_df = pd.DataFrame(sku_summaries)
    return summary_df, forecasts_dict


def calculate_portfolio_kpis(summary_df: pd.DataFrame) -> Dict[str, float]:
    """
    Aggregates high-level executive KPIs across the entire SKU portfolio.
    """
    total_skus = len(summary_df)
    stockout_skus = int((summary_df["risk_status"] == "Stockout Risk").sum())
    overstock_skus = int((summary_df["risk_status"] == "Overstock Risk").sum())
    healthy_skus = int((summary_df["risk_status"] == "Healthy").sum())

    total_inv_value = float(summary_df["current_inventory_value"].sum())
    total_inv_cost = float(summary_df["current_inventory_cost"].sum())
    total_lost_sales_projected = float(summary_df["projected_lost_sales"].sum())
    total_capital_overstock = float(summary_df["capital_tied_overstock"].sum())
    total_suggested_reorder_cost = float(summary_df["estimated_reorder_cost"].sum())
    
    health_percentage = (healthy_skus / total_skus * 100.0) if total_skus > 0 else 0.0

    return {
        "total_skus": total_skus,
        "stockout_skus": stockout_skus,
        "overstock_skus": overstock_skus,
        "healthy_skus": healthy_skus,
        "total_inv_value": total_inv_value,
        "total_inv_cost": total_inv_cost,
        "total_lost_sales_projected": total_lost_sales_projected,
        "total_capital_overstock": total_capital_overstock,
        "total_suggested_reorder_cost": total_suggested_reorder_cost,
        "health_percentage": health_percentage,
    }
