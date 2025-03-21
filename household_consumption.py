import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import simulation_parameters as params

# Monthly average daily consumption in kWh
MONTHLY_AVG_CONSUMPTION = {
    1: 110,  # January
    2: 85,   # February
    3: 50,   # March
    4: 40,   # April
    5: 45,   # May
    6: 51,   # June
    7: 60,   # July
    8: 54,   # August
    9: 49,   # September
    10: 47,  # October
    11: 39,  # November
    12: 71   # December
}

def generate_atlanta_hourly_consumption(month=None, total_daily_kwh=None, days=31, base_noise=0.1, variance_pct=0.35):
    """
    Generate hourly electricity consumption for an Atlanta home.
    
    Parameters:
    - month: Month number (1-12), defaults to the value in simulation_parameters
    - total_daily_kwh: Target daily consumption in kWh, overrides monthly average if provided
    - days: Number of days to simulate
    - base_noise: Random noise factor for hourly variation
    - variance_pct: Percentage variance for daily consumption (0.35 = 35%)
    
    Returns:
    - DataFrame with hourly consumption data
    """
    # Use parameters from simulation_parameters if not specified
    if month is None:
        month = params.MONTH
    
    # Use monthly average by default, but allow override if specified
    if total_daily_kwh is None:
        # Get the monthly average daily consumption
        monthly_avg_kwh = MONTHLY_AVG_CONSUMPTION[month]
        # total_daily_kwh will be determined per day with variance
    else:
        # If explicitly provided, use this as the base value
        monthly_avg_kwh = total_daily_kwh
    
    # Month characteristics
    month_factors = {
        1: {"pattern": "winter", "daylight": 0.1},  # Jan
        2: {"pattern": "winter", "daylight": 0.15}, # Feb
        3: {"pattern": "spring", "daylight": 0.2},  # Mar
        4: {"pattern": "spring", "daylight": 0.3},  # Apr
        5: {"pattern": "spring", "daylight": 0.4},  # May
        6: {"pattern": "summer", "daylight": 0.5},  # Jun
        7: {"pattern": "summer", "daylight": 0.5},  # Jul
        8: {"pattern": "summer", "daylight": 0.45}, # Aug
        9: {"pattern": "fall", "daylight": 0.4},    # Sep
        10: {"pattern": "fall", "daylight": 0.3},   # Oct
        11: {"pattern": "fall", "daylight": 0.2},   # Nov
        12: {"pattern": "winter", "daylight": 0.1}  # Dec
    }
    
    # Create date range for the month
    year = 2025  # Use our simulation year
    start_date = datetime(year, month, 1)
    end_date = start_date + timedelta(days=days)
    date_range = pd.date_range(start=start_date, end=end_date, freq='h', inclusive='left')
    
    # Initialize DataFrame
    df = pd.DataFrame(index=date_range)
    df['hour'] = df.index.hour
    df['day'] = df.index.day
    df['weekday'] = df.index.weekday  # 0=Monday, 6=Sunday
    df['is_weekend'] = df['weekday'].apply(lambda x: 1 if x >= 5 else 0)
    
    # Hourly patterns - different for weekdays and weekends
    # These represent the general shape seen in the graph
    # Ensure hour 0 and hour 23 have similar values for continuity
    hourly_pattern_weekday = {
        0: 0.7, 1: 0.6, 2: 0.55, 3: 0.5, 4: 0.5, 5: 0.6, 
        6: 0.9, 7: 1.3, 8: 1.2, 9: 1.0, 10: 0.9, 11: 0.9,
        12: 0.95, 13: 0.9, 14: 0.9, 15: 1.0, 16: 1.1, 17: 1.4,
        18: 1.6, 19: 1.7, 20: 1.6, 21: 1.4, 22: 1.1, 23: 0.7
    }
    
    hourly_pattern_weekend = {
        0: 0.8, 1: 0.7, 2: 0.6, 3: 0.5, 4: 0.5, 5: 0.5, 
        6: 0.6, 7: 0.8, 8: 1.0, 9: 1.1, 10: 1.2, 11: 1.2,
        12: 1.2, 13: 1.2, 14: 1.1, 15: 1.1, 16: 1.2, 17: 1.3,
        18: 1.4, 19: 1.5, 20: 1.4, 21: 1.3, 22: 1.0, 23: 0.8
    }
    
    # Apply hourly patterns
    df['hourly_factor'] = df.apply(
        lambda row: hourly_pattern_weekend[row['hour']] if row['is_weekend'] else hourly_pattern_weekday[row['hour']], 
        axis=1
    )
    
    # Lighting factor (more in evening, less during daylight hours)
    daylight_hours = {
        1: (7.5, 17.5), 2: (7, 18), 3: (6.5, 18.5), 4: (6, 19),
        5: (5.5, 19.5), 6: (5.5, 20), 7: (5.5, 20), 8: (6, 19.5),
        9: (6.5, 19), 10: (7, 18.5), 11: (7, 17.5), 12: (7.5, 17.5)
    }
    
    dawn, dusk = daylight_hours[month]
    df['lighting_factor'] = df['hour'].apply(
        lambda h: 0.1 if dawn < h < dusk else 0.4 + 0.5 * (1 - min(1, abs(h - 20) / 6))
    )
    
    # Generate daily variation with specified variance
    np.random.seed(42)  # For reproducibility
    
    # Calculate variance range (e.g., 35% variance means from 0.825 to 1.175 of base value)
    variance_range = variance_pct
    
    # Generate daily factors with a normal distribution centered at 1.0
    # and standard deviation such that most values fall within our variance range
    daily_variation = np.random.normal(1.0, variance_range/2, days)
    
    # Ensure values stay within reasonable range (e.g., 0.65 to 1.35 for 35% variance)
    daily_variation = np.clip(daily_variation, 1.0 - variance_range, 1.0 + variance_range)
    
    # Apply the daily variation to create daily target kWh values
    daily_kwh_targets = {}
    for day in range(1, days + 1):
        daily_kwh_targets[day] = monthly_avg_kwh * daily_variation[day-1]
    
    # Apply daily factors to the dataframe
    df['daily_factor'] = df['day'].apply(lambda d: daily_variation[d-1])
    
    # Calculate raw consumption with the hourly pattern
    df['hourly_pattern'] = df['hourly_factor'] * (1 + 0.2 * df['lighting_factor']) * df['daily_factor']
    
    # Scale each day to match its target daily kWh
    df['daily_target_kwh'] = df['day'].map(daily_kwh_targets)
    
    # Calculate scaling factors for each day
    daily_pattern_sums = df.groupby(df.index.day)['hourly_pattern'].sum()
    
    # Create scaling map for each day
    scaling_factors = {}
    for day in range(1, days + 1):
        if day in daily_pattern_sums.index:
            target_kwh = daily_kwh_targets[day]
            pattern_sum = daily_pattern_sums.get(day, 0)
            if pattern_sum > 0:
                scaling_factors[day] = target_kwh / pattern_sum
            else:
                scaling_factors[day] = 1.0
    
    # Apply scaling factors to get final kW values
    df['scaling_factor'] = df['day'].map(scaling_factors)
    df['total_kw'] = df['hourly_pattern'] * df['scaling_factor']
    
    # Ensure no negative values
    df['total_kw'] = df['total_kw'].apply(lambda x: max(0.1, x))
    
    # Keep only relevant columns for the final output
    result = df[['total_kw']].copy()
    result.index.name = 'datetime'
    
    return result

def get_monthly_average_consumption(month):
    """
    Get the average daily consumption for a specific month.
    
    Parameters:
    - month: Month number (1-12)
    
    Returns:
    - Average daily consumption in kWh for the specified month
    """
    return MONTHLY_AVG_CONSUMPTION.get(month, 50)  # Default to 50 if month not found
