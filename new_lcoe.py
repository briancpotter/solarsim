import pandas as pd
import numpy as np
import os

# ======= CONFIGURATION PARAMETERS =======
# You can modify these values directly without changing the function
# Input and output files
NEW_PANELS_CSV = 'solar_vals.csv'
OLD_PANELS_CSV = 'solar_vals_old.csv'
OUTPUT_CSV = 'lcoe_results_with_degradation.csv'
DEGRADATION_YEARS = 10  # Number of years between new and old panel data

# Cost parameters
SOLAR_COST_PER_KW = 1100  # $ per kW
STORAGE_COST_PER_KWH = 476  # $ per kWh
SOLAR_MAINTENANCE_PER_KW_PER_YEAR = 17  # $ per kW per year
STORAGE_MAINTENANCE_PER_KWH_PER_YEAR = 12  # $ per kWh per year

# Lifetime and financial parameters
SOLAR_LIFETIME_YEARS = 35  # years
STORAGE_LIFETIME_YEARS = 20  # years
DISCOUNT_RATE = 0.07  # 7%
# ======= END CONFIGURATION PARAMETERS =======

def calculate_lcoe_with_degradation(new_csv_file, old_csv_file, degradation_years, 
                                   solar_cost_per_kw, storage_cost_per_kwh, 
                                   solar_maintenance_per_kw_per_year, storage_maintenance_per_kwh_per_year,
                                   solar_lifetime_years, storage_lifetime_years, discount_rate=0.05):
    """
    Calculate Levelized Cost of Electricity (LCOE) for different combinations of 
    solar PV capacity and battery storage, accounting for solar panel degradation.
    
    Parameters:
    -----------
    new_csv_file : str
        Path to the CSV file containing energy generation data for new panels (year 0)
    old_csv_file : str
        Path to the CSV file containing energy generation data for degraded panels (after degradation_years)
    degradation_years : int
        Number of years between the two CSV files (used to calculate annual degradation rate)
    solar_cost_per_kw : float
        Capital cost of solar panels in $ per kilowatt
    storage_cost_per_kwh : float
        Capital cost of storage in $ per kilowatt-hour
    solar_maintenance_per_kw_per_year : float
        Annual maintenance costs of solar panels in $ per kilowatt
    storage_maintenance_per_kwh_per_year : float
        Annual maintenance costs of storage in $ per kilowatt-hour
    solar_lifetime_years : int
        Lifetime of solar panels in years
    storage_lifetime_years : int
        Lifetime of storage in years
    discount_rate : float, optional
        Annual discount rate for calculating present value, default=0.05 (5%)
    
    Returns:
    --------
    DataFrame with LCOE values for each solar-storage combination
    """
    # Read in the CSV data
    df_new = pd.read_csv(new_csv_file)
    df_old = pd.read_csv(old_csv_file)
    
    # Get column names
    all_columns = df_new.columns.tolist()
    
    # First column is solar capacity and rest are storage capacities
    solar_col = all_columns[0]
    storage_cols = all_columns[1:]
    
    # Convert storage column headers to float
    storage_capacities = [float(col) for col in storage_cols]
    
    # Get solar capacities from the first column, skipping any NaN values
    solar_capacities = df_new[solar_col].dropna().values
    
    # Create a new dataframe for LCOE results with the same dimensions as the energy data
    lcoe_results = pd.DataFrame(index=range(len(solar_capacities)), columns=storage_cols)
    
    # Calculate annual degradation rates for each combination of solar capacity and storage
    degradation_rates = {}
    
    for i, solar_capacity in enumerate(solar_capacities):
        # Find the row index in the original dataframes that match this solar capacity
        row_idx_new = df_new[df_new[solar_col] == solar_capacity].index[0]
        row_idx_old = df_old[df_old[solar_col] == solar_capacity].index[0]
        
        for storage_col in storage_cols:
            # Get the energy generation values for new and old panels
            energy_new = df_new.loc[row_idx_new, storage_col]
            energy_old = df_old.loc[row_idx_old, storage_col]
            
            # Skip if either energy value is NaN or zero
            if pd.isna(energy_new) or pd.isna(energy_old) or energy_new == 0 or energy_old == 0:
                # Set a default degradation rate if we can't calculate it
                degradation_rates[(solar_capacity, storage_col)] = 0.005  # 0.5% default
                continue
            
            # Calculate annual degradation rate using the formula: 
            # annual_rate = 1 - (old_value/new_value)^(1/degradation_years)
            # This gives us the percentage decrease per year
            try:
                # Make sure we're using the correct values and order
                if energy_new > energy_old and energy_new > 0:
                    annual_degradation_rate = 1 - (energy_old / energy_new) ** (1 / degradation_years)
                    
                    # Validate the degradation rate is reasonable (typically 0.3% to 0.8% per year for solar)
                    if annual_degradation_rate < 0 or annual_degradation_rate > 0.021:
                        # If outside reasonable range, use a typical value of 0.5%
                        annual_degradation_rate = 0.005
                else:
                    # If energy_old > energy_new, something is wrong - use default
                    annual_degradation_rate = 0.005
            except:
                # Fallback to a typical value if there's any calculation error
                annual_degradation_rate = 0.02
                
            degradation_rates[(solar_capacity, storage_col)] = annual_degradation_rate
    
    # Calculate LCOE for each combination of solar capacity and storage
    for i, solar_capacity in enumerate(solar_capacities):
        # Find the row index in the original dataframe that matches this solar capacity
        row_idx_new = df_new[df_new[solar_col] == solar_capacity].index[0]
        
        for j, storage_col in enumerate(storage_cols):
            # Get the initial annual energy generation in kWh for this combination
            initial_energy_kwh = df_new.loc[row_idx_new, storage_col]
            
            # Skip if energy value is NaN or zero
            if pd.isna(initial_energy_kwh) or initial_energy_kwh == 0:
                lcoe_results.iloc[i, j] = float('inf')
                continue
            
            # Get the degradation rate for this combination
            # Use the calculated degradation rate or a reasonable default (0.005 = 0.5%)
            degradation_rate = degradation_rates.get((solar_capacity, storage_col), 0.005)
            
            # Log the degradation rate for debugging purposes (keeping it in the results)
            # This can be removed in the final version
            print(f"Solar: {solar_capacity}, Storage: {storage_col}, Degradation Rate: {degradation_rate*100:.4f}%")
            
            # Calculate the capital costs
            solar_capital_cost = solar_capacity * solar_cost_per_kw
            storage_capacity = float(storage_col)
            storage_capital_cost = storage_capacity * storage_cost_per_kwh
            
            # Calculate annual maintenance costs
            solar_annual_maintenance = solar_capacity * solar_maintenance_per_kw_per_year
            storage_annual_maintenance = storage_capacity * storage_maintenance_per_kwh_per_year
            
            # Calculate present value of all future maintenance costs
            # For solar maintenance
            solar_maintenance_pv = 0
            for year in range(1, solar_lifetime_years + 1):
                solar_maintenance_pv += solar_annual_maintenance / ((1 + discount_rate) ** year)
            
            # For storage maintenance
            storage_maintenance_pv = 0
            for year in range(1, storage_lifetime_years + 1):
                storage_maintenance_pv += storage_annual_maintenance / ((1 + discount_rate) ** year)
            
            # Calculate the present value of all costs over the system lifetime
            # We need to consider the replacement costs if the lifetimes are different
            max_lifetime = max(solar_lifetime_years, storage_lifetime_years)
            
            # Initialize total costs with the initial capital costs
            total_costs_pv = solar_capital_cost + storage_capital_cost
            
            # Add maintenance costs
            total_costs_pv += solar_maintenance_pv + storage_maintenance_pv
            
            # Add replacement costs for components with shorter lifetimes
            years_running = 0
            while years_running < max_lifetime:
                years_running += solar_lifetime_years
                if years_running < max_lifetime:
                    # Add discounted replacement cost for solar panels
                    total_costs_pv += solar_capital_cost / ((1 + discount_rate) ** years_running)
            
            years_running = 0
            while years_running < max_lifetime:
                years_running += storage_lifetime_years
                if years_running < max_lifetime:
                    # Add discounted replacement cost for storage
                    total_costs_pv += storage_capital_cost / ((1 + discount_rate) ** years_running)
            
            # Calculate present value of all energy generated over the maximum lifetime,
            # taking into account the degradation of panels
            energy_pv = 0
            
            # For debugging, track total energy and degradation impact
            total_undiscounted_energy = 0
            
            for year in range(1, max_lifetime + 1):
                # Calculate the degraded energy output for this year
                # The degradation is compounded annually starting from year 2
                # Year 1 has the initial energy with no degradation
                degraded_energy = initial_energy_kwh * ((1 - degradation_rate) ** (year-1))
                
                # Track total undiscounted energy
                total_undiscounted_energy += degraded_energy
                
                # Add the present value of this year's energy
                energy_pv += degraded_energy / ((1 + discount_rate) ** year)
            
            # For debugging of the first few combinations only
            if i < 3 and j < 3:
                print(f"Initial energy: {initial_energy_kwh:.2f} kWh")
                print(f"Total undiscounted energy over {max_lifetime} years: {total_undiscounted_energy:.2f} kWh")
                print(f"Present value of energy: {energy_pv:.2f} kWh")
                print(f"Effective degradation over lifetime: {(1 - total_undiscounted_energy/(initial_energy_kwh*max_lifetime))*100:.2f}%")
                print("---")
            
            # Calculate LCOE
            if energy_pv > 0:
                lcoe = total_costs_pv / energy_pv
            else:
                lcoe = float('inf')  # Avoid division by zero
            
            # Store the result
            lcoe_results.iloc[i, j] = lcoe
    
    # Add the solar capacity column to the results
    lcoe_results.insert(0, solar_col, solar_capacities)
    
    return lcoe_results

# Main execution - this will run automatically when you press F5 in IDLE
if __name__ == "__main__":
    print("Starting LCOE calculation with solar panel degradation...")
    print(f"Using new panels data: {NEW_PANELS_CSV}")
    print(f"Using aged panels data: {OLD_PANELS_CSV}")
    
    # Check if input files exist
    if not os.path.exists(NEW_PANELS_CSV):
        print(f"ERROR: Could not find {NEW_PANELS_CSV}. Please make sure it's in the same directory as this script.")
        input("Press Enter to exit...")
        exit()
        
    if not os.path.exists(OLD_PANELS_CSV):
        print(f"ERROR: Could not find {OLD_PANELS_CSV}. Please make sure it's in the same directory as this script.")
        input("Press Enter to exit...")
        exit()
    
    # Calculate LCOE with degradation
    try:
        lcoe_results = calculate_lcoe_with_degradation(
            NEW_PANELS_CSV,
            OLD_PANELS_CSV,
            DEGRADATION_YEARS,
            SOLAR_COST_PER_KW,
            STORAGE_COST_PER_KWH,
            SOLAR_MAINTENANCE_PER_KW_PER_YEAR,
            STORAGE_MAINTENANCE_PER_KWH_PER_YEAR,
            SOLAR_LIFETIME_YEARS,
            STORAGE_LIFETIME_YEARS,
            DISCOUNT_RATE
        )
        
        # Save LCOE results to CSV
        lcoe_results.to_csv(OUTPUT_CSV, index=False)
        
        print(f"\nLCOE calculation complete! Results saved to '{OUTPUT_CSV}'")
        print(f"Used parameters:")
        print(f"  - Solar panel cost: ${SOLAR_COST_PER_KW}/kW")
        print(f"  - Storage cost: ${STORAGE_COST_PER_KWH}/kWh")
        print(f"  - Solar lifetime: {SOLAR_LIFETIME_YEARS} years")
        print(f"  - Storage lifetime: {STORAGE_LIFETIME_YEARS} years")
        print(f"  - Discount rate: {DISCOUNT_RATE*100:.1f}%")
        print(f"  - Degradation calculated over {DEGRADATION_YEARS} years")
        
    except Exception as e:
        print(f"ERROR: An error occurred during calculation: {str(e)}")
    
    # Keep the console window open when running in IDLE
    input("\nPress Enter to exit...")
