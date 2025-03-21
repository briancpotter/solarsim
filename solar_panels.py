import numpy as np
from solar_incidence import simulate_day
import simulation_parameters as params

def simulate_solar_panels(month=None, day=None, panel_area_sqm=None, panel_efficiency=None, 
                         latitude=None, longitude=None, include_clouds=0):
    """
    Simulate power generation from solar panels based on solar incidence.
    
    Parameters:
    month (int): Month (1-12), defaults to value in simulation_parameters
    day (int): Day of the month, defaults to value in simulation_parameters
    panel_area_sqm (float): Total solar panel area in square meters, defaults to value in simulation_parameters
    panel_efficiency (float): Panel efficiency as a decimal, defaults to value in simulation_parameters
    latitude (float): Latitude, defaults to value in simulation_parameters
    longitude (float): Longitude, defaults to value in simulation_parameters
    include_clouds (int): Whether to include cloud cover effects (0=no, 1=yes)
    
    Returns:
    tuple: (times, power_generation in watts, daily_energy in kWh)
    """
    # Use parameters from simulation_parameters if not specified
    if month is None:
        month = params.MONTH
    if day is None:
        day = params.DAY
    if panel_area_sqm is None:
        panel_area_sqm = params.PANEL_AREA_SQM
    if panel_efficiency is None:
        panel_efficiency = params.PANEL_EFFICIENCY
    if latitude is None:
        latitude = params.LATITUDE
    if longitude is None:
        longitude = params.LONGITUDE
    
    times, incidence = simulate_day(month, day, latitude, longitude, include_clouds)
    
    # Calculate power generation at each time point (W/m² * area * efficiency = Watts)
    power_generation = [inc * panel_area_sqm * panel_efficiency for inc in incidence]
    
    # Calculate total energy generated (using trapezoidal integration)
    time_interval_hours = 1/6  # 10-minute intervals
    daily_energy_wh = 0
    
    for i in range(1, len(power_generation)):
        # Average power between consecutive time points * time interval
        average_power = (power_generation[i-1] + power_generation[i]) / 2
        daily_energy_wh += average_power * time_interval_hours
    
    # Convert Wh to kWh
    daily_energy_kwh = daily_energy_wh / 1000
    
    return times, power_generation, daily_energy_kwh

def calculate_actual_coverage(month=None, day=None, panel_area_sqm=None, panel_efficiency=None, 
                             home_energy_kwh=None, get_hourly_consumption=None, include_clouds=0):
    """
    Calculate the actual percentage of home energy needs covered by solar,
    accounting for the limitation that excess power can't be stored.
    
    Parameters:
    month (int): Month (1-12), defaults to value in simulation_parameters
    day (int): Day of the month, defaults to value in simulation_parameters
    panel_area_sqm (float): Total solar panel area in square meters, defaults to value in simulation_parameters
    panel_efficiency (float): Panel efficiency as a decimal, defaults to value in simulation_parameters
    home_energy_kwh (float): Daily home energy consumption in kWh, defaults to value in simulation_parameters
    get_hourly_consumption: Function that returns hourly consumption data
    include_clouds (int): Whether to include cloud cover effects (0=no, 1=yes)
    
    Returns:
    tuple: (covered_energy_kwh, actual_percentage_covered, energy_surplus_kwh)
    """
    # Use parameters from simulation_parameters if not specified
    if month is None:
        month = params.MONTH
    if day is None:
        day = params.DAY
    if panel_area_sqm is None:
        panel_area_sqm = params.PANEL_AREA_SQM
    if panel_efficiency is None:
        panel_efficiency = params.PANEL_EFFICIENCY
    if home_energy_kwh is None:
        home_energy_kwh = params.HOME_ENERGY_KWH
    
    # Get solar panel generation data
    times, power_generation, _ = simulate_solar_panels(
        month, day, panel_area_sqm, panel_efficiency, include_clouds=include_clouds
    )
    
    # Convert 10-minute intervals to hourly for comparison
    hourly_generation = []
    for hour in range(24):
        # Get the indices for this hour (6 measurements per hour)
        start_idx = hour * 6
        end_idx = start_idx + 6
        
        # Calculate average power for this hour
        hour_avg = sum(power_generation[start_idx:end_idx]) / 6
        hourly_generation.append(hour_avg)
    
    # Get household consumption data for the month
    consumption_df = get_hourly_consumption(month=month, total_daily_kwh=home_energy_kwh)
    
    # Calculate average consumption by hour of day
    hourly_avg_consumption = consumption_df.groupby(consumption_df.index.hour)['total_kw'].mean() * 1000  # kW to W
    
    # Calculate usable solar energy and surplus for each hour
    usable_energy_wh = 0
    surplus_energy_wh = 0
    
    for hour in range(24):
        if hourly_generation[hour] <= hourly_avg_consumption.values[hour]:
            # All solar power is used
            usable_energy_wh += hourly_generation[hour] * 1  # W * hours = Wh (hourly data)
        else:
            # Some solar power is wasted (without storage)
            usable_energy_wh += hourly_avg_consumption.values[hour] * 1  # Use only what's needed
            surplus_energy_wh += (hourly_generation[hour] - hourly_avg_consumption.values[hour]) * 1  # Excess energy
    
    # Convert to kWh
    usable_energy_kwh = usable_energy_wh / 1000
    surplus_energy_kwh = surplus_energy_wh / 1000
    
    # Calculate actual percentage covered
    actual_percentage_covered = (usable_energy_kwh / home_energy_kwh) * 100
    
    return usable_energy_kwh, actual_percentage_covered, surplus_energy_kwh

def calculate_storage_coverage(month=None, day=None, panel_area_sqm=None, panel_efficiency=None, 
                              home_energy_kwh=None, get_hourly_consumption=None, 
                              storage_capacity_kwh=None, initial_charge_kwh=0, include_clouds=0):
    """
    Calculate the actual percentage of home energy needs covered by solar with battery storage,
    accounting for charging and discharging of the battery over time.
    
    Parameters:
    month (int): Month (1-12), defaults to value in simulation_parameters
    day (int): Day of the month, defaults to value in simulation_parameters
    panel_area_sqm (float): Total solar panel area in square meters, defaults to value in simulation_parameters
    panel_efficiency (float): Panel efficiency as a decimal, defaults to value in simulation_parameters
    home_energy_kwh (float): Daily home energy consumption in kWh, defaults to value in simulation_parameters
    get_hourly_consumption: Function that returns hourly consumption data
    storage_capacity_kwh (float): Battery storage capacity in kWh, defaults to value in simulation_parameters
    initial_charge_kwh (float): Initial charge of the battery in kWh
    include_clouds (int): Whether to include cloud cover effects (0=no, 1=yes)
    
    Returns:
    tuple: (directly_used_kwh, from_storage_kwh, total_percentage_covered, 
            wasted_energy_kwh, hourly_storage_state)
    """
    # Use parameters from simulation_parameters if not specified
    if month is None:
        month = params.MONTH
    if day is None:
        day = params.DAY
    if panel_area_sqm is None:
        panel_area_sqm = params.PANEL_AREA_SQM
    if panel_efficiency is None:
        panel_efficiency = params.PANEL_EFFICIENCY
    if home_energy_kwh is None:
        home_energy_kwh = params.HOME_ENERGY_KWH
    if storage_capacity_kwh is None:
        storage_capacity_kwh = params.STORAGE_CAPACITY_KWH
    
    # Get solar panel generation data
    times, power_generation, _ = simulate_solar_panels(
        month, day, panel_area_sqm, panel_efficiency, include_clouds=include_clouds
    )
    
    # Convert 10-minute intervals to hourly for comparison
    hourly_generation = []
    for hour in range(24):
        # Get the indices for this hour (6 measurements per hour)
        start_idx = hour * 6
        end_idx = start_idx + 6
        
        # Calculate average power for this hour
        hour_avg = sum(power_generation[start_idx:end_idx]) / 6
        hourly_generation.append(hour_avg)
    
    # Get household consumption data for the month
    consumption_df = get_hourly_consumption(month=month, total_daily_kwh=home_energy_kwh)
    
    # Calculate average consumption by hour of day
    hourly_avg_consumption = consumption_df.groupby(consumption_df.index.hour)['total_kw'].mean() * 1000  # kW to W
    
    # Calculate energy with storage
    directly_used_wh = 0
    from_storage_wh = 0
    wasted_energy_wh = 0
    
    # Initialize battery state
    battery_charge_wh = initial_charge_kwh * 1000  # Convert kWh to Wh
    max_capacity_wh = storage_capacity_kwh * 1000  # Convert kWh to Wh
    
    # Track battery state over time
    hourly_storage_state = []
    
    for hour in range(24):
        solar_power = hourly_generation[hour]
        consumption = hourly_avg_consumption.values[hour]
        
        if solar_power <= consumption:
            # All solar is used directly
            directly_used_wh += solar_power
            
            # Energy deficit needs to be supplied from battery
            deficit = consumption - solar_power
            
            # Check if battery has enough charge
            usable_charge = min(deficit, battery_charge_wh)
            from_storage_wh += usable_charge
            
            # Discharge the battery
            battery_charge_wh -= usable_charge
        else:
            # Direct consumption
            directly_used_wh += consumption
            
            # Excess power can charge the battery
            excess = solar_power - consumption
            
            # Calculate how much can be stored (respect battery capacity)
            can_be_stored = min(excess, max_capacity_wh - battery_charge_wh)
            battery_charge_wh += can_be_stored
            
            # Any remaining excess is wasted
            wasted_energy_wh += excess - can_be_stored
        
        # Record the battery state at the end of this hour
        hourly_storage_state.append(battery_charge_wh / 1000)  # Store in kWh
    
    # Convert to kWh
    directly_used_kwh = directly_used_wh / 1000
    from_storage_kwh = from_storage_wh / 1000
    wasted_energy_kwh = wasted_energy_wh / 1000
    
    # Calculate total percentage covered
    total_energy_supplied = directly_used_kwh + from_storage_kwh
    total_percentage_covered = (total_energy_supplied / home_energy_kwh) * 100
    
    return directly_used_kwh, from_storage_kwh, total_percentage_covered, wasted_energy_kwh, hourly_storage_state

def get_hourly_generation(month=None, day=None, panel_area_sqm=None, panel_efficiency=None, include_clouds=0):
    """
    Get hourly solar generation data from 10-minute interval data.
    
    Parameters:
    month (int): Month (1-12), defaults to value in simulation_parameters
    day (int): Day of the month, defaults to value in simulation_parameters
    panel_area_sqm (float): Total solar panel area in square meters, defaults to value in simulation_parameters
    panel_efficiency (float): Panel efficiency as a decimal, defaults to value in simulation_parameters
    include_clouds (int): Whether to include cloud cover effects (0=no, 1=yes)
    
    Returns:
    tuple: (hourly_times, hourly_generation)
    """
    # Use parameters from simulation_parameters if not specified
    if month is None:
        month = params.MONTH
    if day is None:
        day = params.DAY
    if panel_area_sqm is None:
        panel_area_sqm = params.PANEL_AREA_SQM
    if panel_efficiency is None:
        panel_efficiency = params.PANEL_EFFICIENCY
    
    # Get solar panel generation data
    times, power_generation, _ = simulate_solar_panels(
        month, day, panel_area_sqm, panel_efficiency, include_clouds=include_clouds
    )
    
    # Convert 10-minute intervals to hourly for comparison
    hourly_times = []
    hourly_generation = []
    
    for hour in range(24):
        # Get the indices for this hour (6 measurements per hour)
        start_idx = hour * 6
        end_idx = start_idx + 6
        
        # Calculate average power for this hour
        hour_avg = sum(power_generation[start_idx:end_idx]) / 6
        
        hourly_times.append(hour)
        hourly_generation.append(hour_avg)
    
    return hourly_times, hourly_generation
