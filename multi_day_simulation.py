import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
import simulation_parameters as params
from household_consumption import generate_atlanta_hourly_consumption
from solar_incidence import simulate_day
from solar_panels import simulate_solar_panels, get_hourly_generation

def run_multi_day_simulation(start_month=3, start_day=15, num_days=7, 
                            panel_area_sqm=50, panel_efficiency=0.20, 
                            storage_capacity_kwh=25, include_clouds=0,
                            battery_efficiency=0.90, output_file="simulation_results.csv",
                            system_age_years=0, panel_decay_rate_pct=0.5):
    """
    Run a multi-day simulation of a solar power system with battery storage.
    
    Parameters:
    start_month (int): Starting month (1-12)
    start_day (int): Starting day of the month
    num_days (int): Number of days to simulate
    panel_area_sqm (float): Solar panel area in square meters
    panel_efficiency (float): Panel efficiency as a decimal
    storage_capacity_kwh (float): Battery storage capacity in kWh
    include_clouds (int): Whether to include cloud effects (0=no, 1=yes)
    battery_efficiency (float): Round trip efficiency of the battery as a decimal (0.90 = 90%)
    output_file (str): Filename for output CSV
    system_age_years (float): Age of the solar panel system in years
    panel_decay_rate_pct (float): Annual decay rate of panel efficiency in percent (e.g., 0.5 for 0.5% per year)
    
    Returns:
    str: Path to the output CSV file
    """
    # Calculate the age-adjusted panel efficiency
    # Decay formula: new_efficiency = initial_efficiency * (1 - decay_rate)^age
    decay_factor = (1 - panel_decay_rate_pct/100) ** system_age_years
    age_adjusted_efficiency = panel_efficiency * decay_factor
    
    print(f"System age: {system_age_years} years")
    print(f"Annual decay rate: {panel_decay_rate_pct}%")
    print(f"Original panel efficiency: {panel_efficiency:.4f} ({panel_efficiency*100:.2f}%)")
    print(f"Age-adjusted efficiency: {age_adjusted_efficiency:.4f} ({age_adjusted_efficiency*100:.2f}%)")
    print(f"Total efficiency reduction: {(1-decay_factor)*100:.2f}%")
    
    # Calculate the one-way efficiency (square root of round trip efficiency)
    # We apply this for both charging and discharging
    one_way_efficiency = np.sqrt(battery_efficiency)
    
    # Initialize simulation start date
    start_date = datetime(2025, start_month, start_day)
    
    # Initialize data storage
    all_data = []
    
    # Initialize battery state
    battery_charge_kwh = 0  # Start with empty battery

    # Run simulation for each day
    for day_offset in range(num_days):
        # Calculate current date
        current_date = start_date + timedelta(days=day_offset)
        current_month = current_date.month
        current_day = current_date.day
        
        # Get solar generation data with age-adjusted efficiency
        _, hourly_generation = get_hourly_generation(
            month=current_month, 
            day=current_day, 
            panel_area_sqm=panel_area_sqm, 
            panel_efficiency=age_adjusted_efficiency,  # Use age-adjusted efficiency
            include_clouds=include_clouds
        )
        
        # Get solar generation data without clouds for the new column 
        # also using age-adjusted efficiency
        _, hourly_generation_nocloud = get_hourly_generation(
            month=current_month, 
            day=current_day, 
            panel_area_sqm=panel_area_sqm, 
            panel_efficiency=age_adjusted_efficiency,  # Use age-adjusted efficiency
            include_clouds=0  # Always use no clouds for this calculation
        )
        
        # Convert hourly generation from watts to kilowatts
        hourly_generation_kw = [w / 1000 for w in hourly_generation]
        hourly_generation_nocloud_kw = [w / 1000 for w in hourly_generation_nocloud]
        
        # Get household consumption for this day
        # Generate for the whole month but extract just the day we need
        consumption_df = generate_atlanta_hourly_consumption(month=current_month, days=31)
        day_consumption = consumption_df[
            (consumption_df.index.day == current_day) & 
            (consumption_df.index.month == current_month)
        ]
        
        # Process each hour of the day
        for hour in range(24):
            # Get timestamp for this hour
            timestamp = datetime(2025, current_month, current_day, hour)
            
            # Get power values for this hour
            solar_generation_kw = hourly_generation_kw[hour]
            solar_generation_nocloud_kw = hourly_generation_nocloud_kw[hour]
            
            # Find the consumption for this exact hour
            try:
                consumption_kw = day_consumption.loc[timestamp, 'total_kw']
            except KeyError:
                # If exact timestamp not found, use the average for this hour
                consumption_kw = day_consumption['total_kw'].mean()
            
            # Calculate energy balance for this hour
            if solar_generation_kw >= consumption_kw:
                # Excess solar power
                direct_solar_kw = consumption_kw
                from_battery_kw = 0
                unfulfilled_kw = 0
                
                # Calculate excess power that can be stored
                excess_kw = solar_generation_kw - consumption_kw
                
                # Apply charging efficiency when storing energy
                storable_excess_kw = excess_kw * one_way_efficiency
                storable_kw = min(storable_excess_kw, storage_capacity_kwh - battery_charge_kwh)
                
                # Update battery charge
                battery_charge_kwh += storable_kw
                
                # Calculate wasted energy (excess that couldn't be stored)
                # This includes both energy lost due to battery efficiency and capacity limits
                wasted_kw = excess_kw - (storable_kw / one_way_efficiency)
            else:
                # Solar not enough, need to use battery
                direct_solar_kw = solar_generation_kw
                energy_deficit_kw = consumption_kw - solar_generation_kw
                
                # Use battery to cover deficit if possible
                # Need to withdraw more due to discharge efficiency
                required_battery_kw = energy_deficit_kw / one_way_efficiency
                from_battery_raw_kw = min(required_battery_kw, battery_charge_kwh)
                battery_charge_kwh -= from_battery_raw_kw
                
                # The actual energy delivered from the battery after efficiency loss
                from_battery_kw = from_battery_raw_kw * one_way_efficiency
                
                # Any remaining deficit is unfulfilled
                unfulfilled_kw = energy_deficit_kw - from_battery_kw
                wasted_kw = 0
            
            # Record data for this hour
            hour_data = {
                'timestamp': timestamp,
                'solar_generation_kw': solar_generation_kw,
                'solar_generation_nocloud_kw': solar_generation_nocloud_kw,
                'household_consumption_kw': consumption_kw,
                'direct_solar_kw': direct_solar_kw,
                'from_battery_kw': from_battery_kw,
                'unfulfilled_kw': unfulfilled_kw,
                'wasted_kw': wasted_kw,
                'battery_charge_kwh': battery_charge_kwh
            }
            
            all_data.append(hour_data)
        
        # End of day processing - collect metrics but don't print
        day_df = pd.DataFrame(all_data)
        day_df = day_df[day_df['timestamp'].dt.date == current_date.date()]
    
    # Create DataFrame from all simulation data
    results_df = pd.DataFrame(all_data)
    
    # Ensure datetime column is properly formatted
    results_df['timestamp'] = pd.to_datetime(results_df['timestamp'])
    results_df.set_index('timestamp', inplace=True)
    
    # Add age-related metadata
    results_df.attrs['system_age_years'] = system_age_years
    results_df.attrs['panel_decay_rate_pct'] = panel_decay_rate_pct
    results_df.attrs['original_efficiency'] = panel_efficiency
    results_df.attrs['age_adjusted_efficiency'] = age_adjusted_efficiency
    
    # Save to CSV
    results_df.to_csv(output_file)
    print(f"Simulation complete. Results saved to {output_file}")
    
    return output_file

def calculate_summary_statistics(csv_file):
    """
    Calculate and print summary statistics from simulation results.
    
    Parameters:
    csv_file (str): Path to the CSV file with simulation results
    
    Returns:
    dict: Summary statistics
    """
    # Load simulation results
    df = pd.read_csv(csv_file, parse_dates=['timestamp'], index_col='timestamp')
    
    # Calculate daily statistics
    daily_stats = df.resample('D').sum()
    
    # Overall statistics
    total_consumption = df['household_consumption_kw'].sum()
    total_generation = df['solar_generation_kw'].sum()
    total_direct_solar = df['direct_solar_kw'].sum()
    total_from_battery = df['from_battery_kw'].sum()
    total_unfulfilled = df['unfulfilled_kw'].sum()
    total_wasted = df['wasted_kw'].sum()
    
    coverage_pct = ((total_direct_solar + total_from_battery) / total_consumption) * 100 if total_consumption > 0 else 0
    generation_ratio = (total_generation / total_consumption) * 100 if total_consumption > 0 else 0
    
    # Calculate battery usage statistics
    battery_cycles = total_from_battery / df['battery_charge_kwh'].max() if df['battery_charge_kwh'].max() > 0 else 0
    battery_max = df['battery_charge_kwh'].max()
    battery_min = df['battery_charge_kwh'].min()
    battery_avg = df['battery_charge_kwh'].mean()
    
    # Print summary
    print("\n===== SIMULATION SUMMARY =====")
    print(f"Simulation period: {df.index.min().strftime('%B %d')} to {df.index.max().strftime('%B %d, %Y')}")
    print(f"Total days: {len(daily_stats)}")
    print("\nENERGY METRICS:")
    print(f"Total household consumption: {total_consumption:.2f} kWh")
    print(f"Total solar generation: {total_generation:.2f} kWh")
    print(f"Total direct solar usage: {total_direct_solar:.2f} kWh")
    print(f"Total energy from battery: {total_from_battery:.2f} kWh")
    print(f"Total unfulfilled energy: {total_unfulfilled:.2f} kWh")
    print(f"Total wasted energy: {total_wasted:.2f} kWh")
    print(f"\nCOVERAGE METRICS:")
    print(f"Overall energy coverage: {coverage_pct:.1f}%")
    print(f"Generation to consumption ratio: {generation_ratio:.1f}%")
    print(f"\nBATTERY METRICS:")
    print(f"Equivalent battery cycles: {battery_cycles:.2f}")
    print(f"Maximum battery charge: {battery_max:.2f} kWh")
    print(f"Minimum battery charge: {battery_min:.2f} kWh")
    print(f"Average battery charge: {battery_avg:.2f} kWh")
    
    # Return summary statistics as a dictionary
    return {
        'total_consumption': total_consumption,
        'total_generation': total_generation,
        'total_direct_solar': total_direct_solar,
        'total_from_battery': total_from_battery,
        'total_unfulfilled': total_unfulfilled,
        'total_wasted': total_wasted,
        'coverage_pct': coverage_pct,
        'generation_ratio': generation_ratio,
        'battery_cycles': battery_cycles
    }

if __name__ == "__main__":
    # Default configuration
    start_month = 1  # January
    start_day = 1
    num_days = 60
    panel_area = 20  # m²
    panel_efficiency = .20  # 20%
    storage_capacity = 0  # kWh
    include_clouds = 1  # Default to include cloud simulation
    battery_efficiency = 1  # Default to 90% round trip efficiency
    system_age_years = 5  # Default to new system
    panel_decay_rate_pct = 0.5  # Default to 0.5% annual degradation
    
    # Configure simulation parameters
    print("\n===== SOLAR SYSTEM SIMULATION CONFIGURATION =====")
    print(f"Start date: {start_month}/{start_day}/2025")
    print(f"Duration: {num_days} days")
    print(f"Panel area: {panel_area} m²")
    print(f"Panel efficiency: {panel_efficiency * 100}%")
    print(f"Battery capacity: {storage_capacity} kWh")
    print(f"Battery efficiency: {battery_efficiency * 100}%")
    print(f"Cloud simulation: {'Enabled' if include_clouds else 'Disabled'}")
    print(f"System age: {system_age_years} years")
    print(f"Panel decay rate: {panel_decay_rate_pct}% per year")
    
    # Run the simulation
    output_file = run_multi_day_simulation(
        start_month=start_month,
        start_day=start_day,
        num_days=num_days,
        panel_area_sqm=panel_area,
        panel_efficiency=panel_efficiency,
        storage_capacity_kwh=storage_capacity,
        include_clouds=include_clouds,
        battery_efficiency=battery_efficiency,
        system_age_years=system_age_years,
        panel_decay_rate_pct=panel_decay_rate_pct
    )
    
    # Calculate and print summary statistics
    summary = calculate_summary_statistics(output_file)
