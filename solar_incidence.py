import math
import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
import simulation_parameters as params

def generate_cloud_pattern(month, day, duration_hours=24, time_step_minutes=10, seed=None):
    """
    Generate a realistic cloud cover pattern for a given day.
    
    Parameters:
    month (int): Month (1-12)
    day (int): Day of the month
    duration_hours (int): Number of hours to simulate
    time_step_minutes (int): Time step in minutes
    seed (int): Random seed for reproducibility
    
    Returns:
    list: Cloud cover values (0.0 to 1.0) for each time step
    """
    # Set random seed for reproducibility, but allow changing for different days
    if seed is None:
        # Create a seed based on the date
        seed = month * 100 + day
    
    np.random.seed(seed)
    
    # Number of time points
    n_points = int(duration_hours * 60 / time_step_minutes)
    
    # Monthly cloud cover probabilities (approximate US averages)
    monthly_cloud_prob = {
        1: 0.65,  # January - Higher cloud cover in winter
        2: 0.65,  # February
        3: 0.60,  # March
        4: 0.55,  # April
        5: 0.50,  # May
        6: 0.40,  # June - Lower cloud cover in summer
        7: 0.35,  # July
        8: 0.40,  # August
        9: 0.45,  # September
        10: 0.50, # October
        11: 0.60, # November
        12: 0.65  # December
    }
    
    # Get base cloud probability for this month
    base_cloud_prob = monthly_cloud_prob.get(month, 0.5)
    
    # Parameters for cloud event generation
    cloud_event_prob = base_cloud_prob   # Probability of a cloud event starting
    event_duration_mean = 12             # Mean duration of a cloud event in time steps
    event_duration_std = 6               # Std deviation of cloud event duration
    clear_duration_mean = 18             # Mean duration of clear periods in time steps
    clear_duration_std = 10              # Std deviation of clear period duration
    
    # Generate cloud pattern
    cloud_cover = np.zeros(n_points)
    
    # Current state (0 = clear, 1 = cloudy)
    current_state = 0 if np.random.random() > base_cloud_prob else 1
    
    # Starting position
    pos = 0
    
    while pos < n_points:
        if current_state == 0:  # Clear
            # Determine duration of clear period
            duration = max(1, int(np.random.normal(clear_duration_mean, clear_duration_std)))
            
            # Set clear sky values (small fluctuations)
            end_pos = min(pos + duration, n_points)
            cloud_cover[pos:end_pos] = np.clip(np.random.normal(0.1, 0.05, end_pos - pos), 0, 0.3)
            
            pos = end_pos
            current_state = 1  # Switch to cloudy
        else:  # Cloudy
            # Determine duration of cloudy period
            duration = max(1, int(np.random.normal(event_duration_mean, event_duration_std)))
            
            # Determine cloud intensity (higher values = more cloud cover)
            cloud_intensity = np.random.uniform(0.4, 0.9)
            
            # Create cloud pattern with peak in the middle
            end_pos = min(pos + duration, n_points)
            actual_duration = end_pos - pos
            
            if actual_duration > 1:
                # Create a bell curve for cloud intensity
                x = np.linspace(-2, 2, actual_duration)
                bell_curve = np.exp(-0.5 * x**2)
                bell_curve = bell_curve / np.max(bell_curve)  # Normalize
                
                # Scale the bell curve to the cloud intensity and add randomness
                cloud_pattern = cloud_intensity * bell_curve + np.random.normal(0, 0.05, actual_duration)
                cloud_cover[pos:end_pos] = np.clip(cloud_pattern, 0.3, 0.95)
            else:
                cloud_cover[pos:end_pos] = cloud_intensity
            
            pos = end_pos
            current_state = 0  # Switch to clear
    
    # Add some temporal correlation (smooth the pattern)
    from scipy.ndimage import gaussian_filter1d
    cloud_cover = gaussian_filter1d(cloud_cover, sigma=2)
    
    return cloud_cover.tolist()

def calculate_solar_incidence(latitude, longitude, day_of_year, hour_of_day, cloud_cover_factor=0):
    """
    Calculate solar incidence (W/m²) based on location, day of year, and hour.
    
    Parameters:
    latitude (float): Latitude in degrees (positive for North)
    longitude (float): Longitude in degrees (positive for East)
    day_of_year (int): Day of year (1-365)
    hour_of_day (float): Hour of day (0-23.99)
    cloud_cover_factor (float): Cloud cover factor (0 = clear, 1 = fully covered)
    
    Returns:
    float: Solar incidence in W/m²
    """
    # Constants
    solar_constant = 1361  # Solar constant in W/m²
    
    # Convert latitude and hour to radians
    lat_rad = math.radians(latitude)
    
    # Calculate solar declination angle (radians)
    declination = math.radians(23.45 * math.sin(math.radians((360/365) * (day_of_year - 81))))
    
    # Calculate hour angle (radians)
    hour_angle = math.radians(15 * (hour_of_day - 12))
    
    # Calculate solar altitude angle (radians)
    sin_altitude = (math.sin(lat_rad) * math.sin(declination) + 
                   math.cos(lat_rad) * math.cos(declination) * math.cos(hour_angle))
    
    # If sun is below horizon, irradiance is 0
    if sin_altitude <= 0:
        return 0
    
    # Approximate atmospheric effects
    # Air mass calculation
    air_mass = 1 / (sin_altitude + 0.50572 * pow((6.07995 + math.degrees(math.asin(sin_altitude))), -1.6364))
    air_mass = min(air_mass, 38)  # Limit air mass to reasonable value
    
    # Calculate direct normal irradiance considering atmospheric attenuation
    dni = solar_constant * 0.7 ** (air_mass ** 0.678)
    
    # Calculate global horizontal irradiance
    ghi = dni * sin_altitude
    
    # Add diffuse component (simplified model)
    diffuse_ratio = 0.3 * (1 - 0.8 * sin_altitude)
    diffuse = ghi * diffuse_ratio / (1 - diffuse_ratio)
    
    # Total irradiance without cloud effects
    total_irradiance = ghi + diffuse
    
    # Apply cloud cover effects if requested
    if cloud_cover_factor > 0:
        # Direct beam component is reduced according to cloud cover
        # Direct beam gets blocked significantly more than diffuse light
        direct_beam_reduction = 1 - (cloud_cover_factor * 0.9)  # Up to 90% reduction
        
        # Diffuse light is reduced less by clouds, and actually increases in some cases
        # For thin clouds diffuse light can increase, for thick clouds it decreases
        diffuse_adjustment = 1 + (0.3 - 0.7 * cloud_cover_factor)  # +30% for thin clouds, -40% for thick
        
        # Apply reductions
        ghi_with_clouds = ghi * direct_beam_reduction
        diffuse_with_clouds = diffuse * diffuse_adjustment
        
        # Recalculate total irradiance with cloud effects
        total_irradiance = ghi_with_clouds + diffuse_with_clouds
    
    # Account for background weather patterns (seasonal cloud patterns)
    month = day_of_year_to_month(day_of_year)
    seasonal_factor = 1  # Default seasonal adjustment
    
    return total_irradiance * seasonal_factor

def day_of_year_to_month(day_of_year):
    """Convert day of year to month."""
    days_in_month = [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    month = 1
    
    while day_of_year > sum(days_in_month[:month+1]):
        month += 1
        if month > 12:
            break
            
    return month

def simulate_day(month=None, day=None, latitude=None, longitude=None, include_clouds=0):
    """
    Simulate solar incidence for any day of the year.
    
    Parameters:
    month (int): Month (1-12), defaults to value in simulation_parameters
    day (int): Day of the month, defaults to value in simulation_parameters
    latitude (float): Latitude, defaults to value in simulation_parameters
    longitude (float): Longitude, defaults to value in simulation_parameters
    include_clouds (int): Whether to include cloud cover effects (0=no, 1=yes)
    
    Returns:
    tuple: (times, incidence values)
    """
    # Use parameters from simulation_parameters if not specified
    if month is None:
        month = params.MONTH
    if day is None:
        day = params.DAY
    if latitude is None:
        latitude = params.LATITUDE
    if longitude is None:
        longitude = params.LONGITUDE
    
    # Calculate day of year
    days_in_month = [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    day_of_year = sum(days_in_month[:month]) + day
    
    # Generate cloud cover pattern if requested
    cloud_pattern = None
    if include_clouds == 1:
        cloud_pattern = generate_cloud_pattern(month, day)
    
    # Generate times for the day (every 10 minutes)
    time_points = []
    incidence_values = []
    
    for hour in range(24):
        for minute in range(0, 60, 10):
            time_idx = hour * 6 + minute // 10
            hour_decimal = hour + minute / 60
            time_str = f"{hour:02d}:{minute:02d}"
            
            # Get cloud cover value for this time point
            cloud_cover = 0
            if include_clouds == 1 and cloud_pattern:
                cloud_cover = cloud_pattern[time_idx]
            
            # Calculate solar incidence with cloud effects
            incidence = calculate_solar_incidence(
                latitude, longitude, day_of_year, hour_decimal, cloud_cover)
            
            time_points.append(time_str)
            incidence_values.append(incidence)
    
    return time_points, incidence_values

def generate_daily_data(month=None, day=None, latitude=None, longitude=None, include_clouds=0):
    """
    Generate and return dataframe with solar incidence data for a given day.
    
    Parameters:
    month (int): Month (1-12), defaults to value in simulation_parameters
    day (int): Day of the month, defaults to value in simulation_parameters
    latitude (float): Latitude, defaults to value in simulation_parameters
    longitude (float): Longitude, defaults to value in simulation_parameters
    include_clouds (int): Whether to include cloud cover effects (0=no, 1=yes)
    
    Returns:
    pandas.DataFrame: DataFrame with time and solar incidence data
    """
    times, incidence = simulate_day(month, day, latitude, longitude, include_clouds)
    data = {
        'Time': times,
        'Solar_Incidence_W_m2': incidence
    }
    df = pd.DataFrame(data)
    return df

def calculate_daily_energy(month=None, day=None, latitude=None, longitude=None, include_clouds=0):
    """
    Calculate the total solar energy (watt-hours per square meter) received over a day.
    
    Parameters:
    month (int): Month (1-12), defaults to value in simulation_parameters
    day (int): Day of the month, defaults to value in simulation_parameters
    latitude (float): Latitude, defaults to value in simulation_parameters
    longitude (float): Longitude, defaults to value in simulation_parameters
    include_clouds (int): Whether to include cloud cover effects (0=no, 1=yes)
    
    Returns:
    float: Total watt-hours per square meter for the day
    """
    times, incidence = simulate_day(month, day, latitude, longitude, include_clouds)
    
    # Each measurement represents a 10-minute interval (1/6 hour)
    time_interval_hours = 1/6
    
    # Sum up the energy for all time intervals (W/m² * hours = Wh/m²)
    # Using trapezoidal integration for more accuracy
    total_energy = 0
    for i in range(1, len(incidence)):
        # Average power between consecutive time points * time interval
        average_power = (incidence[i-1] + incidence[i]) / 2
        total_energy += average_power * time_interval_hours
    
    return total_energy
