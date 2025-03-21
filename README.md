# Solar Energy Simulation System

This package contains a comprehensive set of Python scripts for simulating and analyzing solar energy systems with battery storage. The system allows for detailed modeling of energy generation, consumption, and storage across different timeframes while accounting for real-world factors like cloud cover, panel degradation, and battery efficiency.

## System Overview

This simulation framework enables you to:

- Model solar radiation based on time of day, date, geographic location, and weather conditions
- Simulate household energy consumption with realistic daily and seasonal patterns
- Calculate direct solar energy usage and battery storage dynamics
- Optimize system configurations by analyzing different combinations of panel sizes and battery capacities
- Calculate Levelized Cost of Electricity (LCOE) with consideration for system aging and component lifetimes
- Perform detailed multi-day simulations with hourly resolution

## Script Descriptions

### Core Simulation Modules

#### `simulation_parameters.py`
Defines default parameters used across all modules including:
- Default simulation date (Month, Day)
- Solar panel specifications (Area, Efficiency)
- Home energy consumption (kWh/day)
- Battery storage capacity
- Geographic location (Atlanta, GA by default)
- Utility functions for updating parameters

#### `solar_incidence.py`
Models solar radiation based on astronomical and atmospheric factors:
- Calculates solar incidence (W/m²) based on location, day of year, and hour
- Simulates realistic cloud patterns with monthly variations
- Accounts for atmospheric effects, air mass, and diffuse radiation
- Provides functions to simulate solar radiation for single days or extended periods

#### `household_consumption.py`
Generates realistic household energy consumption patterns:
- Models monthly, daily, and hourly variations in energy usage
- Adjusts for weekday/weekend patterns and seasonal differences
- Creates synthetic but realistic consumption data with appropriate variance
- Includes predefined monthly average consumption profiles for Atlanta

#### `solar_panels.py`
Simulates solar panel energy generation and usage:
- Calculates solar panel output based on incidence and panel specifications
- Simulates direct solar usage and battery storage dynamics
- Models actual energy coverage accounting for time-of-generation vs. consumption
- Provides functions for analyzing system performance with or without storage

### Advanced Analysis Scripts

#### `multi_day_simulation.py`
Performs detailed multi-day simulations with hourly resolution:
- Tracks energy flow between generation, consumption, battery, and grid
- Accounts for battery efficiency and state of charge
- Handles panel degradation based on system age
- Generates comprehensive CSV output with hourly data
- Calculates summary statistics for system performance

#### `new_param_sweep2.py`
Performs parameter sweeps to optimize system configuration:
- Tests multiple combinations of panel sizes and battery capacities
- Uses performance optimizations like caching for efficient large-scale analysis
- Accounts for panel aging and degradation
- Provides detailed analysis of optimal configurations with and without clouds
- Exports results to Excel with multiple data sheets

#### `new_lcoe.py`
Calculates Levelized Cost of Electricity (LCOE) for different system configurations:
- Accounts for capital costs, maintenance, and component lifetimes
- Models panel degradation effects on long-term energy production
- Applies financial parameters like discount rate
- Handles component replacement at different lifecycle intervals
- Creates CSV output with LCOE values for each configuration
- This takes in two CSV files as inputs, examples of which are provided

## Dependencies

The simulation system requires the following Python libraries:
- `pandas`: For data manipulation and analysis
- `numpy`: For numerical operations
- `matplotlib` (implied): For visualization (though not directly imported in the provided scripts)
- `scipy`: For signal processing in cloud pattern generation
- `openpyxl`: For Excel output in parameter sweeps
- `psutil`: For memory monitoring in large-scale simulations

## How to Use

### Basic Simulation

For a basic simulation of a solar energy system:

1. Modify `simulation_parameters.py` with your desired default settings
2. Run `multi_day_simulation.py` to perform a detailed hourly simulation
3. Review the output CSV file and summary statistics

Example:
```python
from multi_day_simulation import run_multi_day_simulation, calculate_summary_statistics

# Run a 30-day simulation with 25m² of panels and 10kWh battery
output_file = run_multi_day_simulation(
    start_month=3,        # March
    start_day=1,          # 1st day of month
    num_days=30,          # 30 days 
    panel_area_sqm=25,    # 25 m² of panels
    panel_efficiency=0.20, # 20% efficiency
    storage_capacity_kwh=10, # 10kWh battery
    include_clouds=1,     # Include cloud effects
    battery_efficiency=0.90, # 90% round-trip efficiency
    system_age_years=0    # New system
)

# Calculate statistics
summary = calculate_summary_statistics(output_file)
```

### Parameter Exploration

To explore multiple combinations of solar panel area and storage capacity

1. Run `new_param_sweep2.py` to test multiple panel/battery combinations
2. Review the Excel output to identify power produced under different configurations
3. Run `new_lcoe.py` with the simulation output in a CSV to see how costs change depending on configuration

Example:
```python
from new_param_sweep2 import run_parameter_sweep_fast

# Run a parameter sweep for various panel sizes and battery capacities
excel_file = run_parameter_sweep_fast(
    panel_area_min=10,    # Min panel area
    panel_area_max=50,    # Max panel area
    panel_area_step=5,    # Panel area step
    storage_capacity_min=0, # Min storage capacity
    storage_capacity_max=30, # Max storage capacity
    storage_capacity_step=5, # Storage step
    num_days=365,         # Full year simulation
    include_clouds=1      # Include cloud effects
)
```

## Data Flow

The system works through the following flow:

1. `solar_incidence.py` calculates solar radiation based on location and time
2. `household_consumption.py` generates realistic consumption patterns
3. `solar_panels.py` converts solar radiation to energy with panel parameters
4. Analysis scripts combine these elements to evaluate system performance:
   - `multi_day_simulation.py` for detailed time-series analysis
   - `new_param_sweep2.py` for configuration optimization
   - `new_lcoe.py` for economic analysis

## Key Features and Considerations

### Realistic Modeling
- **Geographic Specificity**: Calculations based on precise latitude/longitude
- **Weather Effects**: Cloud patterns modeled with seasonal variations
- **Consumption Patterns**: Household usage with realistic daily/seasonal patterns
- **Battery Dynamics**: Efficiency losses during charging and discharging

### System Aging
- Panel efficiency degradation over time (typically 0.5% per year)
- Distinct simulations for new vs. aged systems
- Aging factors incorporated into LCOE calculations

### Performance Optimization
- Caching mechanisms for repetitive calculations
- Memory management for large parameter sweeps
- Optimized algorithms for multi-day simulations

### Economic Analysis
- Capital and maintenance costs for panels and batteries
- Component replacement at different lifecycle points
- Present value calculations with discount rates
- Detailed LCOE output for different configurations

## Customization

The system is highly customizable through the parameter files and function arguments. Key customization points include:

- Geographic location (latitude/longitude)
- Panel specifications (area, efficiency)
- Battery capacity and efficiency
- System age and degradation rates
- Simulation timeframes (start date, duration)
- Economic parameters (costs, discount rate, lifetimes)

## Limitations

- The simulation uses synthetic cloud patterns rather than actual weather data
- Household consumption is based on statistical models rather than actual smart meter data
- The system does not model specific inverter characteristics or DC/AC conversion losses
- Grid interaction is simplified (energy is either used, stored, or wasted)
- This whole package was vibe-coded with Claude 3.7 Sonnet, and while I've done my best to verify the outputs I wouldn't swear to its accuracy. The solar panel degradation calculation in particular was added very late and I didn't spend much time checking to make sure it was producing sensible outputs. Use at your own risk!

