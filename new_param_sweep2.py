import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import gc
import os
import psutil  # For memory monitoring
from functools import lru_cache
from solar_panels import get_hourly_generation
from household_consumption import generate_atlanta_hourly_consumption, get_monthly_average_consumption

# Global cache for weather patterns (cloud patterns are deterministic for a given month/day)
WEATHER_CACHE = {}

# Initialize cache stats
CACHE_HITS = 0
CACHE_MISSES = 0

# Maximum size for simulation cache
MAX_CACHE_SIZE = 1000

def report_memory_usage():
    """Report current memory usage of the process"""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    return memory_info.rss / (1024 * 1024)  # Convert to MB

@lru_cache(maxsize=12)  # One per month
def get_consumption_cached(month, days=31):
    """Cached version of generate_atlanta_hourly_consumption"""
    return generate_atlanta_hourly_consumption(month=month, days=days)

@lru_cache(maxsize=366)  # One per day of year
def get_hourly_generation_cached(month, day, panel_area_sqm, panel_efficiency, include_clouds):
    """Cached version of get_hourly_generation with normalized panel parameters"""
    # Calculate with unit area and efficiency=1.0
    times, base_generation = get_hourly_generation(
        month=month, day=day, 
        panel_area_sqm=1.0,  # Unit area
        panel_efficiency=1.0,  # Unit efficiency
        include_clouds=include_clouds
    )
    
    # Scale by actual panel area and efficiency
    scaled_generation = [power * panel_area_sqm * panel_efficiency for power in base_generation]
    
    return times, scaled_generation

def simulate_day_energy(month, day, panel_area, panel_efficiency, 
                       storage_capacity, include_clouds, consumption_data,
                       initial_battery_charge=0, battery_efficiency=0.90):
    """
    Fast simulation of a single day's energy balance.
    Returns tuple of (direct_solar, from_battery, to_battery, unfulfilled, wasted, final_battery_charge)
    
    Parameters:
    battery_efficiency (float): Round trip efficiency of the battery as a decimal (0.90 = 90%)
    """
    # Calculate the one-way efficiency (square root of round trip efficiency)
    one_way_efficiency = np.sqrt(battery_efficiency)
    
    # Get hour-by-hour data
    _, generation = get_hourly_generation_cached(month, day, panel_area, panel_efficiency, include_clouds)
    
    # Convert generation from Watts to kilowatts
    generation_kw = [w / 1000 for w in generation]
    
    # Get consumption for this day
    day_consumption = consumption_data[
        (consumption_data.index.day == day) & 
        (consumption_data.index.month == month)
    ]
    
    # If we don't have data for this exact day, use the monthly average
    if len(day_consumption) == 0:
        # This is a fallback but shouldn't normally happen
        avg_consumption = get_monthly_average_consumption(month) / 24  # kWh per hour
        consumption_kw = [avg_consumption] * 24
    else:
        # Extract hourly consumption
        consumption_by_hour = {}
        for idx, row in day_consumption.iterrows():
            consumption_by_hour[idx.hour] = row['total_kw']
        
        # Fill any missing hours with average
        avg_consumption = day_consumption['total_kw'].mean()
        consumption_kw = [consumption_by_hour.get(hour, avg_consumption) for hour in range(24)]
    
    # Initialize metrics
    direct_solar_kwh = 0
    from_battery_kwh = 0
    to_battery_kwh = 0
    unfulfilled_kwh = 0
    wasted_kwh = 0
    
    # Use the battery charge passed in from the previous day
    battery_charge_kwh = initial_battery_charge
    
    # Process each hour
    for hour in range(24):
        solar_kw = generation_kw[hour]
        load_kw = consumption_kw[hour]
        
        # Calculate energy balance for this hour
        if solar_kw >= load_kw:
            # All load is supplied directly
            direct_solar_kwh += load_kw
            
            # Excess can charge battery
            excess = solar_kw - load_kw
            
            # Apply charging efficiency
            chargeable_excess = excess * one_way_efficiency
            
            # How much can go to battery?
            can_store = min(chargeable_excess, storage_capacity - battery_charge_kwh)
            battery_charge_kwh += can_store
            to_battery_kwh += can_store
            
            # Any remaining is wasted (including efficiency losses)
            wasted_kwh += excess - (can_store / one_way_efficiency)
        else:
            # Use all available solar directly
            direct_solar_kwh += solar_kw
            
            # Need to use battery for the deficit
            deficit = load_kw - solar_kw
            
            # Apply discharge efficiency - need to withdraw more
            required_battery = deficit / one_way_efficiency
            
            # How much can come from battery?
            withdrawable = min(required_battery, battery_charge_kwh)
            battery_charge_kwh -= withdrawable
            
            # The actual energy delivered from the battery after efficiency loss
            delivered = withdrawable * one_way_efficiency
            from_battery_kwh += delivered
            
            # Any remaining deficit is unfulfilled
            unfulfilled_kwh += deficit - delivered
    
    # Return the day's energy metrics and final battery state
    return (direct_solar_kwh, from_battery_kwh, to_battery_kwh, 
            unfulfilled_kwh, wasted_kwh, battery_charge_kwh)

def simulate_year_fast(start_month, start_day, num_days, panel_area, 
                     panel_efficiency, storage_capacity, include_clouds,
                     battery_efficiency=0.90):
    """
    Fast simulation of multiple days without creating huge DataFrames.
    Returns total energy metrics directly without storing hourly data.
    
    Parameters:
    battery_efficiency (float): Round trip efficiency of the battery as a decimal (0.90 = 90%)
    """
    # Get start date
    start_date = datetime(2025, start_month, start_day)
    
    # Pre-cache consumption data for all relevant months
    unique_months = set()
    for day_offset in range(num_days):
        current_date = start_date + timedelta(days=day_offset)
        unique_months.add(current_date.month)
    
    consumption_data = {month: get_consumption_cached(month) for month in unique_months}
    
    # Initialize totals
    total_direct_solar = 0
    total_from_battery = 0
    total_to_battery = 0
    total_unfulfilled = 0
    total_wasted = 0
    total_consumption = 0
    
    # Initialize battery state to fully charged (100%)
    battery_charge_kwh = storage_capacity
    
    # Simulate each day
    for day_offset in range(num_days):
        current_date = start_date + timedelta(days=day_offset)
        current_month = current_date.month
        current_day = current_date.day
        
        # Get consumption data for this day to calculate total consumption
        day_consumption = consumption_data[current_month][
            (consumption_data[current_month].index.day == current_day) & 
            (consumption_data[current_month].index.month == current_month)
        ]
        day_total_consumption = day_consumption['total_kw'].sum()
        total_consumption += day_total_consumption
        
        # Simulate this day - pass the current battery charge and efficiency
        (day_direct_solar, day_from_battery, day_to_battery,
         day_unfulfilled, day_wasted, battery_charge_kwh) = simulate_day_energy(
            current_month, current_day, panel_area, panel_efficiency,
            storage_capacity, include_clouds, consumption_data[current_month],
            initial_battery_charge=battery_charge_kwh,  # Pass the battery state from previous day
            battery_efficiency=battery_efficiency  # Pass battery efficiency
        )
        
        # Accumulate totals
        total_direct_solar += day_direct_solar
        total_from_battery += day_from_battery
        total_to_battery += day_to_battery
        total_unfulfilled += day_unfulfilled
        total_wasted += day_wasted
    
    # Return the overall energy metrics
    return {
        'total_consumption': total_consumption,
        'total_direct_solar': total_direct_solar,
        'total_from_battery': total_from_battery,
        'total_to_battery': total_to_battery,
        'total_unfulfilled': total_unfulfilled,
        'total_wasted': total_wasted,
        'energy_supplied': total_direct_solar + total_from_battery,
        'coverage_pct': ((total_direct_solar + total_from_battery) / total_consumption * 100) 
                        if total_consumption > 0 else 0
    }

# Dictionary-based cache with size limitation
class SimulationCache:
    def __init__(self, max_size=1000):
        self.cache = {}
        self.max_size = max_size
        self.hits = 0
        self.misses = 0
    
    def get(self, key):
        if key in self.cache:
            self.hits += 1
            return self.cache[key]
        self.misses += 1
        return None
    
    def put(self, key, value):
        # If we're at capacity, remove oldest 25% of entries
        if len(self.cache) >= self.max_size:
            # Simple eviction - just clear half the cache
            # In a more sophisticated version, we could track access times
            items = list(self.cache.items())
            self.cache = dict(items[len(items)//2:])
            gc.collect()  # Force garbage collection
        
        self.cache[key] = value
    
    def clear(self):
        self.cache.clear()
        gc.collect()

# Create our simulation cache
sim_cache = SimulationCache(MAX_CACHE_SIZE)

def run_single_simulation_fast(panel_area, storage_capacity, include_clouds, other_params):
    """
    Run a single simulation with given parameters using the optimized fast simulation.
    Returns detailed metrics for total, direct, and battery energy.
    """
    # Extract parameters
    start_month = other_params.get('start_month', 1)
    start_day = other_params.get('start_day', 1)
    num_days = other_params.get('num_days', 364)
    panel_efficiency = other_params.get('panel_efficiency', 0.21)
    battery_efficiency = other_params.get('battery_efficiency', 0.90)
    
    # Extract age and decay parameters
    system_age_years = other_params.get('system_age_years', 0)
    panel_decay_rate_pct = other_params.get('panel_decay_rate_pct', 0.5)
    
    # Calculate age-adjusted panel efficiency
    decay_factor = (1 - panel_decay_rate_pct/100) ** system_age_years
    age_adjusted_efficiency = panel_efficiency * decay_factor
    
    # Check cache first
    cache_key = (panel_area, storage_capacity, include_clouds, 
                 start_month, start_day, num_days, age_adjusted_efficiency, battery_efficiency,
                 system_age_years, panel_decay_rate_pct)
    
    cached_result = sim_cache.get(cache_key)
    if cached_result is not None:
        return panel_area, storage_capacity, include_clouds, cached_result
    
    # Run the fast simulation with age-adjusted efficiency
    results = simulate_year_fast(
        start_month=start_month,
        start_day=start_day,
        num_days=num_days,
        panel_area=panel_area,
        panel_efficiency=age_adjusted_efficiency,  # Use age-adjusted efficiency
        storage_capacity=storage_capacity,
        include_clouds=include_clouds,
        battery_efficiency=battery_efficiency
    )
    
    # Create a tuple with all metrics we want to track
    energy_metrics = {
        'total': results['energy_supplied'],
        'direct': results['total_direct_solar'],
        'battery': results['total_from_battery']
    }
    
    # Cache the results
    sim_cache.put(cache_key, energy_metrics)
    
    # Return simulation results
    return panel_area, storage_capacity, include_clouds, energy_metrics

def clear_caches():
    """Clear all caches to free memory"""
    sim_cache.clear()
    get_hourly_generation_cached.cache_clear()
    get_consumption_cached.cache_clear()
    gc.collect()  # Force garbage collection

def run_parameter_sweep_fast(
    panel_area_min=5, panel_area_max=50, panel_area_step=5,
    storage_capacity_min=0, storage_capacity_max=50, storage_capacity_step=5,
    start_month=1, start_day=1, num_days=364, 
    panel_efficiency=0.21, battery_efficiency=0.90, include_clouds=1,
    system_age_years=0, panel_decay_rate_pct=0.5,  # New parameters for system age
    output_prefix="sweep_results_optimized"
):
    """
    Run a parameter sweep of the solar simulation over different panel areas and storage capacities.
    Highly optimized version for large numbers of simulations.
    
    Parameters:
    battery_efficiency (float): Round trip efficiency of the battery as a decimal (0.90 = 90%)
    system_age_years (float): Age of the solar panel system in years
    panel_decay_rate_pct (float): Annual decay rate of panel efficiency in percent (e.g., 0.5 for 0.5% per year)
    """
    # Start timing
    start_time = time.time()
    
    # Generate parameter values
    panel_areas = np.arange(panel_area_min, panel_area_max + panel_area_step/2, panel_area_step)
    storage_capacities = np.arange(storage_capacity_min, storage_capacity_max + storage_capacity_step/2, storage_capacity_step)
    
    # Calculate age-adjusted panel efficiency
    decay_factor = (1 - panel_decay_rate_pct/100) ** system_age_years
    age_adjusted_efficiency = panel_efficiency * decay_factor
    
    # Create result matrices - now for each metric
    # With clouds
    results_cloud_total = np.zeros((len(panel_areas), len(storage_capacities)))
    results_cloud_direct = np.zeros((len(panel_areas), len(storage_capacities)))
    results_cloud_battery = np.zeros((len(panel_areas), len(storage_capacities)))
    
    # Without clouds
    results_nocloud_total = np.zeros((len(panel_areas), len(storage_capacities)))
    results_nocloud_direct = np.zeros((len(panel_areas), len(storage_capacities)))
    results_nocloud_battery = np.zeros((len(panel_areas), len(storage_capacities)))
    
    # Calculate total simulations
    total_combinations = len(panel_areas) * len(storage_capacities)
    total_simulations = total_combinations * 2 if include_clouds == 1 else total_combinations
    
    print(f"\n===== OPTIMIZED PARAMETER SWEEP CONFIGURATION =====")
    print(f"Panel area: {panel_area_min} to {panel_area_max} m² (step: {panel_area_step} m²)")
    print(f"Storage capacity: {storage_capacity_min} to {storage_capacity_max} kWh (step: {storage_capacity_step} kWh)")
    print(f"Start date: {start_month}/{start_day}/2025")
    print(f"Duration: {num_days} days")
    print(f"Original panel efficiency: {panel_efficiency * 100:.1f}%")
    print(f"System age: {system_age_years} years")
    print(f"Annual decay rate: {panel_decay_rate_pct}%")
    print(f"Age-adjusted efficiency: {age_adjusted_efficiency * 100:.1f}% ({(1-decay_factor)*100:.2f}% reduction)")
    print(f"Battery efficiency: {battery_efficiency * 100:.1f}%")
    print(f"Cloud simulation: {'Enabled' if include_clouds else 'Disabled'}")
    print(f"Total parameter combinations: {total_combinations}")
    print(f"Total simulations to run: {total_simulations}")
    print("=" * 55)
    
    # Define shared parameters
    other_params = {
        'start_month': start_month,
        'start_day': start_day,
        'num_days': num_days,
        'panel_efficiency': panel_efficiency,
        'battery_efficiency': battery_efficiency,
        'system_age_years': system_age_years,
        'panel_decay_rate_pct': panel_decay_rate_pct
    }
    
    # Progress tracking variables
    completed = 0
    last_update_time = time.time()
    update_interval = 5  # Only update progress every 5 seconds for large simulations
    
    # Pre-calculate baseline consumption for analysis
    # Get a representative simulation for baseline metrics
    baseline_results = simulate_year_fast(
        start_month=start_month,
        start_day=start_day,
        num_days=num_days,
        panel_area=10,  # Use a medium value
        panel_efficiency=age_adjusted_efficiency,  # Use age-adjusted efficiency
        storage_capacity=10,  # Use a medium value
        include_clouds=0,  # No clouds for baseline
        battery_efficiency=battery_efficiency
    )
    total_household_consumption = baseline_results['total_consumption']
    
    # With clouds (if enabled)
    if include_clouds == 1:
        for i, panel_area in enumerate(panel_areas):
            # Only clear caches and run gc occasionally to balance performance
            if i > 0 and i % 3 == 0:
                gc.collect()
                
            for j, storage_capacity in enumerate(storage_capacities):
                # Run with clouds
                _, _, _, energy_metrics = run_single_simulation_fast(
                    panel_area, storage_capacity, 1, other_params
                )
                
                # Store results for each metric
                results_cloud_total[i, j] = energy_metrics['total']
                results_cloud_direct[i, j] = energy_metrics['direct']
                results_cloud_battery[i, j] = energy_metrics['battery']
                
                # Update progress
                completed += 1
                
                # Update progress less frequently for better performance with large simulations
                current_time = time.time()
                if (current_time - last_update_time > update_interval or 
                    completed == 1 or completed == total_simulations):
                    last_update_time = current_time
                    
                    progress_pct = completed / total_simulations * 100
                    elapsed = current_time - start_time
                    remaining = (elapsed / completed) * (total_simulations - completed) if completed > 0 else 0
                    
                    # Format time estimates
                    elapsed_fmt = time.strftime("%H:%M:%S", time.gmtime(elapsed))
                    remaining_fmt = time.strftime("%H:%M:%S", time.gmtime(remaining))
                    
                    memory_mb = report_memory_usage()
                    
                    print(f"\rProgress: {completed}/{total_simulations} ({progress_pct:.1f}%) - "
                          f"Elapsed: {elapsed_fmt} - Remaining: {remaining_fmt} - "
                          f"Latest: Panel={panel_area}m², Storage={storage_capacity}kWh, "
                          f"Clouds=Yes, Total={energy_metrics['total']:.2f}kWh - "
                          f"Cache: {sim_cache.hits}/{sim_cache.hits+sim_cache.misses} hits - "
                          f"Memory: {memory_mb:.1f} MB", 
                          end='', flush=True)
    
    # Clear caches between cloud and no-cloud simulations
    clear_caches()
    
    # Without clouds (always run)
    for i, panel_area in enumerate(panel_areas):
        # Only clear caches and run gc occasionally to balance performance
        if i > 0 and i % 3 == 0:
            gc.collect()
            
        for j, storage_capacity in enumerate(storage_capacities):
            # Run without clouds
            _, _, _, energy_metrics = run_single_simulation_fast(
                panel_area, storage_capacity, 0, other_params
            )
            
            # Store results for each metric
            results_nocloud_total[i, j] = energy_metrics['total']
            results_nocloud_direct[i, j] = energy_metrics['direct']
            results_nocloud_battery[i, j] = energy_metrics['battery']
            
            # Update progress
            completed += 1
            
            # Update progress less frequently
            current_time = time.time()
            if (current_time - last_update_time > update_interval or 
                completed == total_simulations):
                last_update_time = current_time
                
                progress_pct = completed / total_simulations * 100
                elapsed = current_time - start_time
                remaining = (elapsed / completed) * (total_simulations - completed) if completed > 0 else 0
                
                # Format time estimates
                elapsed_fmt = time.strftime("%H:%M:%S", time.gmtime(elapsed))
                remaining_fmt = time.strftime("%H:%M:%S", time.gmtime(remaining))
                
                memory_mb = report_memory_usage()
                
                print(f"\rProgress: {completed}/{total_simulations} ({progress_pct:.1f}%) - "
                      f"Elapsed: {elapsed_fmt} - Remaining: {remaining_fmt} - "
                      f"Latest: Panel={panel_area}m², Storage={storage_capacity}kWh, "
                      f"Clouds=No, Total={energy_metrics['total']:.2f}kWh - "
                      f"Cache: {sim_cache.hits}/{sim_cache.hits+sim_cache.misses} hits - "
                      f"Memory: {memory_mb:.1f} MB", 
                      end='', flush=True)
    
    # If clouds weren't enabled, copy the results
    if include_clouds != 1:
        results_cloud_total = results_nocloud_total.copy()
        results_cloud_direct = results_nocloud_direct.copy()
        results_cloud_battery = results_nocloud_battery.copy()
    
    print("\nAll simulations completed!")
    
    # Create result DataFrames for each metric
    # With clouds
    df_cloud_total = pd.DataFrame(results_cloud_total, 
                            index=panel_areas, 
                            columns=storage_capacities)
    df_cloud_direct = pd.DataFrame(results_cloud_direct, 
                             index=panel_areas, 
                             columns=storage_capacities)
    df_cloud_battery = pd.DataFrame(results_cloud_battery, 
                              index=panel_areas, 
                              columns=storage_capacities)
    
    # Without clouds
    df_nocloud_total = pd.DataFrame(results_nocloud_total, 
                              index=panel_areas, 
                              columns=storage_capacities)
    df_nocloud_direct = pd.DataFrame(results_nocloud_direct, 
                               index=panel_areas, 
                               columns=storage_capacities)
    df_nocloud_battery = pd.DataFrame(results_nocloud_battery, 
                                index=panel_areas, 
                                columns=storage_capacities)
    
    # Set index and column names for all DataFrames
    for df in [df_cloud_total, df_cloud_direct, df_cloud_battery,
               df_nocloud_total, df_nocloud_direct, df_nocloud_battery]:
        df.index.name = "Panel Area (m²)"
        df.columns.name = "Storage Capacity (kWh)"
    
    # Add age-related metadata to file name
    if system_age_years > 0:
        output_prefix = f"{output_prefix}_age{system_age_years}yr_{panel_decay_rate_pct}pct"
        
    # Save to Excel file with six tabs
    excel_output = f"{output_prefix}_results.xlsx"
    
    # Use ExcelWriter to create a file with multiple sheets
    with pd.ExcelWriter(excel_output, engine='openpyxl') as writer:
        # With clouds
        df_cloud_total.to_excel(writer, sheet_name='Total Energy (with clouds)')
        df_cloud_direct.to_excel(writer, sheet_name='Direct Solar (with clouds)')
        df_cloud_battery.to_excel(writer, sheet_name='From Battery (with clouds)')
        
        # Without clouds
        df_nocloud_total.to_excel(writer, sheet_name='Total Energy (no clouds)')
        df_nocloud_direct.to_excel(writer, sheet_name='Direct Solar (no clouds)')
        df_nocloud_battery.to_excel(writer, sheet_name='From Battery (no clouds)')
        
        # Create a metadata sheet
        metadata = pd.DataFrame({
            'Parameter': [
                'System Age (years)', 
                'Panel Decay Rate (%/year)', 
                'Original Panel Efficiency (%)',
                'Age-adjusted Efficiency (%)',
                'Total Efficiency Reduction (%)'
            ],
            'Value': [
                system_age_years,
                panel_decay_rate_pct,
                panel_efficiency * 100,
                age_adjusted_efficiency * 100,
                (1-decay_factor) * 100
            ]
        })
        metadata.to_excel(writer, sheet_name='Simulation Metadata')
    
    print(f"Results saved to Excel file: {excel_output}")
    print(f"  - 'Total Energy (with clouds)': Total energy supplied with cloud effects")
    print(f"  - 'Direct Solar (with clouds)': Energy directly from solar panels with cloud effects")
    print(f"  - 'From Battery (with clouds)': Energy supplied from battery with cloud effects")
    print(f"  - 'Total Energy (no clouds)': Total energy supplied without cloud effects")
    print(f"  - 'Direct Solar (no clouds)': Energy directly from solar panels without cloud effects")
    print(f"  - 'From Battery (no clouds)': Energy supplied from battery without cloud effects")
    print(f"  - 'Simulation Metadata': Age and efficiency parameters")
    
    # Calculate elapsed time
    elapsed_time = time.time() - start_time
    hours = int(elapsed_time // 3600)
    minutes = int((elapsed_time % 3600) // 60)
    seconds = int(elapsed_time % 60)
    
    print(f"\nParameter sweep completed in {hours}h {minutes}m {seconds}s")
    
    # Print analysis
    print("\n===== PARAMETER SWEEP ANALYSIS =====")
    print(f"Total household energy consumption: {total_household_consumption:.2f} kWh")
    print(f"Simulation period: {num_days} days starting from {start_month}/{start_day}/2025")
    print(f"System age: {system_age_years} years")
    print(f"Panel decay rate: {panel_decay_rate_pct}% per year")
    print(f"Age-adjusted panel efficiency: {age_adjusted_efficiency*100:.2f}% (original: {panel_efficiency*100:.2f}%)")
    print(f"Total efficiency reduction due to age: {(1-decay_factor)*100:.2f}%")
    print(f"Battery efficiency: {battery_efficiency * 100:.1f}%")
    print("-" * 45)
    
    # Find optimal configurations
    max_energy_cloud = df_cloud_total.max().max()
    max_energy_nocloud = df_nocloud_total.max().max()
    
    # Get positions of maximum values
    max_pos_cloud = np.where(results_cloud_total == max_energy_cloud)
    max_panel_cloud = panel_areas[max_pos_cloud[0][0]]
    max_storage_cloud = storage_capacities[max_pos_cloud[1][0]]
    
    max_pos_nocloud = np.where(results_nocloud_total == max_energy_nocloud)
    max_panel_nocloud = panel_areas[max_pos_nocloud[0][0]]
    max_storage_nocloud = storage_capacities[max_pos_nocloud[1][0]]
    
    # Get the breakdown for the optimal configurations
    idx_cloud = (np.argmax(results_cloud_total.max(axis=1)), np.argmax(results_cloud_total.max(axis=0)))
    idx_nocloud = (np.argmax(results_nocloud_total.max(axis=1)), np.argmax(results_nocloud_total.max(axis=0)))
    
    # Get values for the optimal configuration
    max_direct_cloud = results_cloud_direct[max_pos_cloud[0][0]][max_pos_cloud[1][0]]
    max_battery_cloud = results_cloud_battery[max_pos_cloud[0][0]][max_pos_cloud[1][0]]
    
    max_direct_nocloud = results_nocloud_direct[max_pos_nocloud[0][0]][max_pos_nocloud[1][0]]
    max_battery_nocloud = results_nocloud_battery[max_pos_nocloud[0][0]][max_pos_nocloud[1][0]]
    
    print(f"WITH CLOUDS:")
    print(f"Maximum energy supplied: {max_energy_cloud:.2f} kWh ({(max_energy_cloud/total_household_consumption*100):.1f}% of total consumption)")
    print(f"  - Direct solar: {max_direct_cloud:.2f} kWh ({(max_direct_cloud/max_energy_cloud*100):.1f}% of supplied)")
    print(f"  - From battery: {max_battery_cloud:.2f} kWh ({(max_battery_cloud/max_energy_cloud*100):.1f}% of supplied)")
    print(f"Optimal configuration: Panel area = {max_panel_cloud} m², Storage capacity = {max_storage_cloud} kWh")
    
    print(f"\nWITHOUT CLOUDS:")
    print(f"Maximum energy supplied: {max_energy_nocloud:.2f} kWh ({(max_energy_nocloud/total_household_consumption*100):.1f}% of total consumption)")
    print(f"  - Direct solar: {max_direct_nocloud:.2f} kWh ({(max_direct_nocloud/max_energy_nocloud*100):.1f}% of supplied)")
    print(f"  - From battery: {max_battery_nocloud:.2f} kWh ({(max_battery_nocloud/max_energy_nocloud*100):.1f}% of supplied)")
    print(f"Optimal configuration: Panel area = {max_panel_nocloud} m², Storage capacity = {max_storage_nocloud} kWh")
    
    # Calculate impact of clouds
    if include_clouds == 1 and max_energy_nocloud > 0:
        cloud_impact = ((max_energy_nocloud - max_energy_cloud) / max_energy_nocloud) * 100
        print(f"\nCloud impact: {cloud_impact:.1f}% reduction in energy supply at optimal configurations")
    
    # Calculate impact of aging
    if system_age_years > 0:
        aging_impact = ((1 - decay_factor) * 100)
        print(f"Aging impact: {aging_impact:.1f}% reduction in energy supply due to {system_age_years} years of degradation")
    
    # Final memory cleanup
    clear_caches()
    
    return excel_output

# ===== MAIN EXECUTION =====

if __name__ == "__main__":
    # Example usage
    excel_results = run_parameter_sweep_fast(
        panel_area_min=155,
        panel_area_max=295,
        panel_area_step=10,
        storage_capacity_min=0,
        storage_capacity_max=200,
        storage_capacity_step=10,
        start_month=1,
        start_day=1,
        num_days=364,
        panel_efficiency=0.21,
        battery_efficiency=0.90,  # Default 90% efficiency
        include_clouds=1,
        system_age_years=10,  # 5 years of system age
        panel_decay_rate_pct=2,  # 0.5% annual degradation
        output_prefix="sweep_results_optimized"
    )
