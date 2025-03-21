# Simulation parameters that will be shared across modules

# Default simulation date
MONTH = 3  # 3 = March
DAY = 15

# Solar panel parameters
PANEL_AREA_SQM = 25  # Total panel area in square meters
PANEL_EFFICIENCY = 0.20  # 20% efficiency

# Home energy consumption (kWh/day)
HOME_ENERGY_KWH = 50

# Battery storage capacity (kWh)
STORAGE_CAPACITY_KWH = 25

# Location parameters (Atlanta, GA)
LATITUDE = 33.749
LONGITUDE = -84.388

# Month names for display
MONTH_NAMES = ['', 'January', 'February', 'March', 'April', 'May', 'June', 
               'July', 'August', 'September', 'October', 'November', 'December']

def update_parameters(month=None, day=None, panel_area_sqm=None, panel_efficiency=None, 
                     home_energy_kwh=None, storage_capacity_kwh=None):
    """
    Update the simulation parameters.
    
    Parameters:
    month (int): Month (1-12)
    day (int): Day of the month
    panel_area_sqm (float): Total solar panel area in square meters
    panel_efficiency (float): Panel efficiency as a decimal
    home_energy_kwh (float): Daily home energy consumption in kWh
    storage_capacity_kwh (float): Battery storage capacity in kWh
    """
    global MONTH, DAY, PANEL_AREA_SQM, PANEL_EFFICIENCY, HOME_ENERGY_KWH, STORAGE_CAPACITY_KWH
    
    if month is not None:
        MONTH = month
    if day is not None:
        DAY = day
    if panel_area_sqm is not None:
        PANEL_AREA_SQM = panel_area_sqm
    if panel_efficiency is not None:
        PANEL_EFFICIENCY = panel_efficiency
    if home_energy_kwh is not None:
        HOME_ENERGY_KWH = home_energy_kwh
    if storage_capacity_kwh is not None:
        STORAGE_CAPACITY_KWH = storage_capacity_kwh
