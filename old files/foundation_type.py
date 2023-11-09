import pandas as pd


# these need to be connected to the tables

# Input variables
wind_speed_km_h = 22  # Wind load in km/h, from the shelter_info table: max_windspeed
ground_type = 8 

# All the information from the shelters.xlsx
shelter_height = 3
shelter_width = 5
shelter_depth = 6
surface_area_m2 = shelter_depth * shelter_height  # Surface area in square meters, now assumed. In further development this could be calculated for each shelter. 
shelter_weight = 100


soil_df = pd.read_csv('data/soil_data.csv', sep=';')


# print(soil_df)

consistency_df = pd.read_csv('data/soil_consistency.csv',sep = ";", encoding='latin')
# print(consistency_df)

UCS = soil_df['c:cohesion Kpa = KN/m2'][18] * 2 # Unconfined Compressive Strength, UCs of the soiltype in the location (ref: https://eng.libretexts.org/Bookshelves/Civil_Engineering/Book%3A_The_Delft_Sand_Clay_and_Rock_Cutting_Model_(Miedema)/02%3A_Basic_Soil_Mechanics/2.04%3A_Soil_Mechanical_Parameters)

print(UCS)

est_consistency = []
foundation_type = []
for index, row in consistency_df.iterrows():
    if row['UCS(kPa) (lower boundary)'] <= UCS < row['UCS(kPa) (upper boundary)']:
        est_consistency = (row['Estimated Consistency'])
        print(est_consistency)
        if est_consistency == 'Hard':
            foundation_type = 'no foundation possible, use weights or pad layer'
        elif est_consistency in ('Very Stiff', 'Stiff'):
            foundation_type.extend(['base plate nailed to soil', 'pegs', 'screw in ground anchor'])
        elif est_consistency in ('Medium', 'Soft'):
            foundation_type.extend(['screw in ground anchor', 'embedded base plate', 'concrete bucket foundation'])
        if est_consistency in ('Soft', 'Very Soft'):
            foundation_type.append('concrete pad foundation')

print(foundation_type)


# Constants
air_density = 1.225  # kg/m^3 (standard air density)
conversion_factor = 1000 / 3600  # Conversion from km/h to m/s

# Convert wind speed from km/h to m/s
wind_speed_m_s = wind_speed_km_h * conversion_factor

# Calculate wind pressure
wind_pressure = 0.5 * air_density * wind_speed_m_s ** 2

# Calculate wind load (force) on the surface
wind_load_force = wind_pressure * surface_area_m2

# Output the results
print(f"Wind Speed (m/s): {wind_speed_m_s}")
print(f"Wind Pressure (Pa): {wind_pressure} Pa")
print(f"Wind Load Force (N): {wind_load_force} N")


# Calculation of uplift force, simplified
# calculate the torque (momentum) caused by the windload: act like the building is a beam with one fixed side (B3 on the TU Delft structural mechanics formula sheet)
# 1/2ql**2
def uplift(wind_pressure, shelter_depth, shelter_height, shelter_width):
    q = wind_pressure * shelter_depth
    l = shelter_height
    torque = 1/2 * q*l**2
    torque

    # calculate the Fup and Fdown on the outsides of the shelter; T = R*F
    Fdown = torque/(1/2*shelter_width)
    Fup = -(Fdown)
    return Fup, Fdown

uplift_x = uplift(wind_pressure, shelter_depth, shelter_height, shelter_width)
uplift_y = uplift(wind_pressure, shelter_width,shelter_height,shelter_depth)

uplift_max = max(uplift_x + uplift_y)
uplift_min = min(uplift_x + uplift_y)

print(uplift_max, uplift_min)
anchor_needed = uplift_max - (shelter_weight*9.8)/(2)
if anchor_needed > 0:
    print('anchor needed:', anchor_needed, 'N')
else: 
    print('no anchor needed')