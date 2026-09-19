"""
Project FORESIGHT: Synthetic Data Generation Module
----------------------------------------------------
Generates realistic historical daily sales, inventory levels, lead times,
and pricing across multiple SKU categories for inventory intelligence.
"""

from typing import Dict, List, Optional
import datetime
import numpy as np
import pandas as pd


# Default SKU Catalog with rich operational metadata
DEFAULT_SKU_CATALOG: List[Dict] = [
    {
        "sku_id": "SKU-ELEC-101",
        "sku_name": "Noise-Cancelling Wireless Headphones Pro",
        "category": "Electronics",
        "base_demand": 38.0,
        "demand_std": 8.5,
        "trend_slope": 0.035,        # Growing product line
        "seasonality_type": "weekend_boost",
        "unit_price": 199.99,
        "unit_cost": 85.00,
        "lead_time_days": 14,
        # Stockout state scenario: Low stock relative to lead time demand + safety stock
        "inventory_scenario": "stockout_risk",
    },
    {
        "sku_id": "SKU-ELEC-102",
        "sku_name": "Ultra HD 4K Gaming Monitor 27-inch",
        "category": "Electronics",
        "base_demand": 22.0,
        "demand_std": 5.0,
        "trend_slope": 0.015,
        "seasonality_type": "weekend_boost",
        "unit_price": 349.50,
        "unit_cost": 180.00,
        "lead_time_days": 18,
        "inventory_scenario": "healthy",
    },
    {
        "sku_id": "SKU-APPR-201",
        "sku_name": "Merino Wool Weather-Shield Jacket",
        "category": "Apparel",
        "base_demand": 45.0,
        "demand_std": 12.0,
        "trend_slope": -0.010,
        "seasonality_type": "winter_peak",
        "unit_price": 129.00,
        "unit_cost": 48.00,
        "lead_time_days": 10,
        "inventory_scenario": "overstock_risk",
    },
    {
        "sku_id": "SKU-APPR-202",
        "sku_name": "Ergonomic Performance Running Shoes",
        "category": "Apparel",
        "base_demand": 55.0,
        "demand_std": 11.5,
        "trend_slope": 0.025,
        "seasonality_type": "weekend_boost",
        "unit_price": 110.00,
        "unit_cost": 42.00,
        "lead_time_days": 12,
        "inventory_scenario": "stockout_risk",
    },
    {
        "sku_id": "SKU-HOME-301",
        "sku_name": "Ergonomic Lumbar Mesh Office Chair",
        "category": "Home & Kitchen",
        "base_demand": 18.0,
        "demand_std": 4.2,
        "trend_slope": 0.008,
        "seasonality_type": "weekday_boost",
        "unit_price": 249.00,
        "unit_cost": 115.00,
        "lead_time_days": 21,
        "inventory_scenario": "healthy",
    },
    {
        "sku_id": "SKU-HOME-302",
        "sku_name": "Precision Cold Brew Coffee Maker 1.5L",
        "category": "Home & Kitchen",
        "base_demand": 30.0,
        "demand_std": 7.0,
        "trend_slope": 0.012,
        "seasonality_type": "summer_peak",
        "unit_price": 45.99,
        "unit_cost": 16.50,
        "lead_time_days": 8,
        "inventory_scenario": "overstock_risk",
    },
    {
        "sku_id": "SKU-BEAU-401",
        "sku_name": "Hydrating Multi-Peptide Facial Serum",
        "category": "Health & Beauty",
        "base_demand": 62.0,
        "demand_std": 14.0,
        "trend_slope": 0.040,
        "seasonality_type": "smooth",
        "unit_price": 38.50,
        "unit_cost": 9.20,
        "lead_time_days": 7,
        "inventory_scenario": "stockout_risk",
    },
    {
        "sku_id": "SKU-BEAU-402",
        "sku_name": "Organic Botanical Night Repair Elixir",
        "category": "Health & Beauty",
        "base_demand": 40.0,
        "demand_std": 8.0,
        "trend_slope": 0.018,
        "seasonality_type": "smooth",
        "unit_price": 54.00,
        "unit_cost": 14.00,
        "lead_time_days": 9,
        "inventory_scenario": "healthy",
    },
    {
        "sku_id": "SKU-FOOD-501",
        "sku_name": "Artisan Single-Origin Espresso Beans 1kg",
        "category": "Food & Beverage",
        "base_demand": 75.0,
        "demand_std": 16.0,
        "trend_slope": 0.020,
        "seasonality_type": "weekday_boost",
        "unit_price": 28.00,
        "unit_cost": 11.00,
        "lead_time_days": 5,
        "inventory_scenario": "stockout_risk",
    },
    {
        "sku_id": "SKU-FOOD-502",
        "sku_name": "Cold-Pressed Organic Extra Virgin Olive Oil 750ml",
        "category": "Food & Beverage",
        "base_demand": 48.0,
        "demand_std": 9.5,
        "trend_slope": 0.005,
        "seasonality_type": "smooth",
        "unit_price": 22.50,
        "unit_cost": 8.50,
        "lead_time_days": 6,
        "inventory_scenario": "healthy",
    },
    {
        "sku_id": "SKU-FITN-601",
        "sku_name": "Smart Bluetooth Heart Rate Fitness Band",
        "category": "Fitness & Sports",
        "base_demand": 34.0,
        "demand_std": 7.8,
        "trend_slope": 0.015,
        "seasonality_type": "new_year_boost",
        "unit_price": 79.99,
        "unit_cost": 29.00,
        "lead_time_days": 15,
        "inventory_scenario": "overstock_risk",
    },
    {
        "sku_id": "SKU-FITN-602",
        "sku_name": "Quick-Lock Adjustable Dumbbell Set 24kg",
        "category": "Fitness & Sports",
        "base_demand": 26.0,
        "demand_std": 6.2,
        "trend_slope": 0.022,
        "seasonality_type": "new_year_boost",
        "unit_price": 289.00,
        "unit_cost": 130.00,
        "lead_time_days": 20,
        "inventory_scenario": "healthy",
    }
]


def generate_synthetic_sales_data(
    num_days: int = 365,
    end_date: Optional[datetime.date] = None,
    seed: int = 42,
    sku_catalog: Optional[List[Dict]] = None
) -> pd.DataFrame:
    """
    Generates a realistic daily sales time-series DataFrame for all SKUs.

    Parameters:
    -----------
    num_days : int
        Number of historical days to simulate (default 365).
    end_date : datetime.date, optional
        Last date of historical records (defaults to yesterday).
    seed : int
        Random number generator seed for reproducibility.
    sku_catalog : List[Dict], optional
        Catalog configurations for SKUs.

    Returns:
    --------
    pd.DataFrame:
        Columns: [date, sku_id, sku_name, category, units_sold, unit_price,
                  unit_cost, lead_time_days, current_inventory]
    """
    np.random.seed(seed)
    catalog = sku_catalog or DEFAULT_SKU_CATALOG

    if end_date is None:
        end_date = datetime.date.today() - datetime.timedelta(days=1)
    
    start_date = end_date - datetime.timedelta(days=num_days - 1)
    date_range = pd.date_range(start=start_date, end=end_date, freq="D")
    
    all_rows = []

    for sku in catalog:
        sku_id = sku["sku_id"]
        sku_name = sku["sku_name"]
        category = sku["category"]
        base_demand = sku["base_demand"]
        demand_std = sku["demand_std"]
        slope = sku["trend_slope"]
        seasonality = sku["seasonality_type"]
        unit_price = sku["unit_price"]
        unit_cost = sku.get("unit_cost", round(unit_price * 0.45, 2))
        lead_time_days = sku["lead_time_days"]
        scenario = sku.get("inventory_scenario", "healthy")

        # Create realistic time series components
        t = np.arange(num_days)
        
        # 1. Linear Growth/Decline Trend
        trend_component = slope * t

        # 2. Weekly Seasonality (Day of week effects)
        dow = date_range.dayofweek.to_numpy()  # Monday=0, Sunday=6
        if seasonality == "weekend_boost":
            # Weekend surges (+35% on Fri, Sat, Sun)
            dow_multipliers = np.array([0.88, 0.92, 0.95, 1.00, 1.25, 1.40, 1.20])
        elif seasonality == "weekday_boost":
            # B2B / Office surge during mid-week
            dow_multipliers = np.array([1.15, 1.25, 1.28, 1.20, 0.95, 0.60, 0.57])
        else:
            dow_multipliers = np.array([0.96, 0.98, 1.00, 1.02, 1.08, 1.10, 0.96])
        
        weekly_component = dow_multipliers[dow]

        # 3. Annual / Holiday / Seasonal Waves
        doy = date_range.dayofyear.to_numpy()
        if seasonality == "winter_peak":
            # Q4 / Winter surge
            annual_component = 1.0 + 0.35 * np.cos((doy - 355) * 2 * np.pi / 365)
        elif seasonality == "summer_peak":
            # Mid-year surge (June/July)
            annual_component = 1.0 + 0.30 * np.cos((doy - 190) * 2 * np.pi / 365)
        elif seasonality == "new_year_boost":
            # January fitness resolution surge
            annual_component = 1.0 + 0.45 * np.exp(-0.5 * ((doy - 15) / 25) ** 2)
        else:
            # Mild smooth wave
            annual_component = 1.0 + 0.12 * np.sin(doy * 2 * np.pi / 365)

        # 4. Random Promotional Spikes (e.g. flash sales, 4-5 times a year)
        promo_mask = np.random.binomial(n=1, p=0.015, size=num_days)
        promo_multipliers = 1.0 + promo_mask * np.random.uniform(0.6, 1.2, size=num_days)

        # 5. Gaussian Noise & Non-negative Daily Sales
        noise = np.random.normal(0, demand_std, size=num_days)
        
        # Combined Daily Demand Formula
        daily_units = (base_demand + trend_component + noise) * weekly_component * annual_component * promo_multipliers
        daily_units = np.clip(np.round(daily_units), a_min=0, a_max=None).astype(int)

        # Calculate empirical average daily demand & std for stock configuration
        recent_30_sales = daily_units[-30:]
        avg_daily_demand = float(np.mean(recent_30_sales))
        std_daily_demand = float(np.std(recent_30_sales, ddof=1)) if len(recent_30_sales) > 1 else demand_std
        
        # Reference Reorder Point (Service Level 95% => Z=1.65)
        safety_stock_ref = 1.65 * std_daily_demand * np.sqrt(lead_time_days)
        rop_ref = (avg_daily_demand * lead_time_days) + safety_stock_ref
        monthly_demand_ref = avg_daily_demand * 30.0

        # Assign calibrated Current Inventory based on testing scenario
        if scenario == "stockout_risk":
            # Current inventory significantly lower than or equal to ROP
            # e.g., 20% to 80% of ROP (critical to high risk)
            current_inventory = int(np.random.uniform(0.20, 0.85) * rop_ref)
        elif scenario == "overstock_risk":
            # Current inventory exceeds 3x monthly demand (e.g., 3.2x to 5.0x)
            current_inventory = int(np.random.uniform(3.2, 4.8) * monthly_demand_ref)
        else:
            # Healthy inventory: 1.4x to 2.2x ROP (safely above ROP, below overstock limit)
            current_inventory = int(np.random.uniform(1.3, 2.2) * rop_ref)

        # Append row items
        for date_val, units in zip(date_range, daily_units):
            all_rows.append({
                "date": date_val.strftime("%Y-%m-%d"),
                "sku_id": sku_id,
                "sku_name": sku_name,
                "category": category,
                "units_sold": int(units),
                "unit_price": float(unit_price),
                "unit_cost": float(unit_cost),
                "lead_time_days": int(lead_time_days),
                "current_inventory": int(current_inventory),
            })

    df = pd.DataFrame(all_rows)
    df["date"] = pd.to_datetime(df["date"])
    return df
