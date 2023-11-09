import pandas as pd
import math
from geopy.geocoders import Nominatim


def calculate_type_of_foundation(UCS):



    consistency_df = pd.read_csv(r'C:\Users\dimit\OneDrive - Delft University of Technology\Desktop\DESKTOP\2nd year\CORE\Structural\soil_consistency.csv',sep = ";", encoding='latin')

    for index, row in consistency_df.iterrows():
        if row['UCS(kPa) (lower boundary)'] <= UCS < row['UCS(kPa) (upper boundary)']:

            est_consistency = row['Estimated Consistency']

            if est_consistency == 'Hard':
                foundation_type = 'No foundation possible, use weights or pad layer'
            elif est_consistency in ('Very Stiff', 'Stiff'):
                foundation_type = 'Base plate nailed to soil / Pegs / Screw in ground anchor'
            elif est_consistency in ('Medium', 'Soft'):
                foundation_type = 'screw in ground anchor / Embedded base plate / Concrete bucket foundation'
            else: 
                foundation_type = ''

            if est_consistency in ('Soft', 'Very Soft'):
                foundation_type += ' Concrete pad foundation'

    return foundation_type


def uplift(wind_pressure, shelter_depth, shelter_height, shelter_width):
    q = wind_pressure * shelter_depth
    l = shelter_height
    torque = 1/2 * q*l**2
    torque

    # calculate the Fup and Fdown on the outsides of the shelter; T = R*F
    Fdown = torque/(1/2*shelter_width)
    Fup = -(Fdown)
    return Fup, Fdown


def calculate_uplift_decision(wind_pressure, shelter_depth, shelter_height, shelter_width, shelter_weight):
    uplift_x = uplift(wind_pressure, shelter_depth, shelter_height, shelter_width)
    uplift_y = uplift(wind_pressure, shelter_width,shelter_height,shelter_depth)

    uplift_max = max(uplift_x + uplift_y)
    uplift_min = min(uplift_x + uplift_y)

    # print(uplift_max, uplift_min)
    anchor_needed = uplift_max - (shelter_weight*9.8)/(2)
    if anchor_needed > 0:
        decision = 'anchor needed:' + anchor_needed + 'N'
    else: 
        decision = 'no anchor needed'

    return decision


def compute_structural_calculations(shelter_data, highest_wind_speed):

    #   #  COORDINATES   #    #

    # Initialize Nominatim API to find the coordinates 
    geolocator = Nominatim(user_agent="MyApp")
    location = geolocator.geocode("Antakya")


    latitude = location.latitude
    longitude = location.longitude


    #   #  COORDINATES   #    #






    #   #  ASKING FOR SOIL TYPE   #    #


    # Create a dictionary to map ESC-8 status letters (A to F) to ground_type and specification values
    esc8_mapping = {
        'A': ('Rock', 'Layers'),
        'B': ('hardpan, cemented sand or gravel', 'all'),
        'C': ('gravel or sand', 'Compact'),
        'D': ('Sand, coarse to medium', 'Loose'),
        'E': ('sand, fine, silty or with trace of clay', 'Firm'),
        'F': ('Silt', 'Loose')
    }


    # Provide instructions to the user
    print(f"Please check the website https://tadas.afad.gov.tr/list-station for the ESC-8 status, with latitude {latitude} and longitute {longitude}")  #Reference for Turkish soil type : https://tadas.afad.gov.tr/list-station
    print("After checking, enter the corresponding letter (A to F) in the terminal.")

    # Ask the user to enter the ESC-8 status letter
    esc8_status = input("Enter the ESC-8 status letter: ")

    # Check if the input letter is valid (A to F)
    if esc8_status.upper() in esc8_mapping:
        ground_type, specification = esc8_mapping[esc8_status.upper()]
        print(f"The corresponding Ground Type: {ground_type}")
        print(f"The corresponding Specification: {specification}")
    else:
        print("Invalid input. Please enter a letter from A to F.")   #Reference for the Ec8 : https://www.phd.eng.br/wp-content/uploads/2015/02/en.1998.1.2004.pdf 


    #   #  ASKING FOR SOIL TYPE   #    #






    #   #  GET THE PROPERTIES OF THE SOIL ACCORDING TO THE TYPE   #    #

    # Read the csv file
    df = pd.read_excel(r'C:\Users\dimit\OneDrive - Delft University of Technology\Desktop\DESKTOP\2nd year\CORE\Soil data\Soil data.xlsx')   # Reference for the bearing capacity of each soil : https://dicausa.com/soil-bearing-capacity/
    data = pd.DataFrame(df)

    

    # 
    # Get the Bearing capacity
    # 
    def get_numbers_by_ground_and_specification(data, ground_type, specification, column_name1, column_name2, column_name3, column_name4, column_name5, column_name6, column_name7, column_name8 ):
        # Filter the DataFrame based on the selected "Ground Type" and "Specification"
        filtered_data = data[(data['Ground Type'] == ground_type) & (data['Specification'] == specification)]

        if not filtered_data.empty:
            # If there are matching rows, extract the numbers from the specified columns
            number1 = filtered_data[column_name1].values[0]
            number2 = filtered_data[column_name2].values[0]
            number3 = filtered_data[column_name3].values[0]
            number4 = filtered_data[column_name4].values[0]
            number5 = filtered_data[column_name5].values[0]
            number6 = filtered_data[column_name6].values[0]
            number7 = filtered_data[column_name7].values[0]
            number8 = filtered_data[column_name8].values[0]
            return number1, number2, number3, number4, number5, number6, number7, number8  # Return a tuple containing both numbers
        else:
            # If there are no matching rows, return None or an appropriate message for both numbers
            return None, None, None, None, None, None, None, None

    # Example usage:
    # Assuming you have your data in a DataFrame named 'df'
    column_name1 = "Bearing capacity (kN/m2)"
    column_name2 = "w:density of soil Kg/m2"
    column_name3 = "Angle of repose"
    column_name4 = "Nc"
    column_name5 = "Nq"
    column_name6 = "Nγ"
    column_name7 = "γ : unit weight of soil KN/m3"
    column_name8 = "c:cohesion Kpa = KN/m2"                     # Reference : https://www.geotechdata.info/parameter/cohesion


    number1, number2, number3, number4, number5, number6, number7, number8 = get_numbers_by_ground_and_specification(df, ground_type, specification, column_name1, column_name2, column_name3, column_name4, column_name5, column_name6, column_name7, column_name8 )
    # print(number1, number2, number3, number4, number5, number6, number7, number8)


    #   #  GET THE PROPERTIES OF THE SOIL ACCORDING TO THE TYPE   #    #







    #   #  NUMBER OF FOOTINGS ACCORDING TO THE DIMENSIONS OF THE SHELTER   #    #


    def calculate_number_of_footings(row):
        length = row['length']
        width = row['width']
        
        if length <= 6:
            footing_L = 2
        elif 6 < length < 12:
            footing_L = 3
        else:
            footing_L = 4

        if width <= 6:
            footing_W = 2
        elif 6 < width < 12:
            footing_W = 3
        else:
            footing_W = 4

        return footing_L * footing_W
    shelter_data=shelter_data.copy()

    shelter_data['Number of Footings'] = shelter_data.apply(calculate_number_of_footings, axis=1)


    # #   #  NUMBER OF FOOTINGS ACCORDING TO THE DIMENSIONS OF THE SHELTER   #    #







    #   #   CALCULATE THE SIZE OF THE FOUNDATION   #    #



    # Define a function to calculate the length of the foundation based on shape

    def calculate_L(shape, area = None):
        if shape == "strip":
            return 1
        elif shape == "circle":
            return None
        elif shape == "square":
            return None
        elif shape == "rectangle":
            return 0.6
        else:
            print("Invalid shape input")
            return None

    # Get the shape input from the user
    shape = input("Enter the shape (strip, circle, square, or rectangle): ")

    # Call the function to calculate 'L' based on the input shape
    L = calculate_L(shape)                        # L: length of the foundation (m)
    shelter_data['Length of the foundation'] = L



    # Calculate the selfweight and the bearing capacity of soil

    g = 9.81
    shelter_data['package weight']= shelter_data['package weight'] * g  # g:Gravitational acceleration (m/s^2)               
    P = (shelter_data['package weight'] / 1000)               # P: Building load (kN)
    qu= round(number1, 3)                           # qu:Bearing capacity of soil (KN/m2)


    # Calculate the foundations

    shelter_data=shelter_data.copy()
    Area_footing = shelter_data['Footing area'] = P / qu      # Area of each foundation (m2)


    # Define a function to calculate the width of the foundation based on shape and L
    def calculate_foundation_width(row, shape, L):
        area = row['Footing area']  # Assuming you have a 'Footing area' column in your DataFrame
        if shape == "strip":
            return area / L
        elif shape == "circle":
            return 2 * math.sqrt(area / math.pi)
        elif shape == "square":
            return math.sqrt(area) 
        elif shape == "rectangle":
            return area / L
        else:
            return None

    # Apply the function to your DataFrame to calculate 'Width of the foundation'
    B =shelter_data['Width of the foundation'] = shelter_data.apply(calculate_foundation_width, args=(shape, L), axis=1)   # Width of the foundation (m)
    shelter_data['Width of the foundation'] = round(shelter_data['Width of the foundation'],3)
    
    if shape == "square":
        shelter_data['Length of the foundation'] = B

    #   #   CALCULATE THE SIZE OF THE FOUNDATION   #    #








    # #   #   MINIMUM DEPTH OF THE FOUNDATION   #    #
    
    w = number2                                                                        # w: density of soil (kg/m3)
    sin_θ = math.sin(math.radians(number3))                                            # sin_θ: angle of repose 
    D = shelter_data['Depth of the foundation'] = (P / w) * ((1 - sin_θ) / (1 + sin_θ)) ** 2         # D: Depth of the foundation (m)
    # If the Depth is less than 0.10m then assume depth 0.10

    # #   #   MINIMUM DEPTH OF THE FOUNDATION   #    #






    
    # #    #   BEARING CAPACITY EQUATION (TERZAGHI) CHECK   #     #

    # Assign different values to sc based on the input shape
    def calculate_sc_and_sgamma(shape, B, L):
        if shape == "strip":
            sc, sγ = 1, 1
        elif shape == "circle":
            sc, sγ = 1.3, 0.6
        elif shape == "square":
            sc, sγ = 1.3, 0.8
        elif shape == "rectangle":
            sc, sγ = (1 + 0.2 * B / L), (1 - 0.2 * B / L)
        else:
            print("Invalid shape input")
            return None
        return sc, sγ

    # Call the function to calculate 'sc' and 'sγ' based on the input shape
    sc, sγ = calculate_sc_and_sgamma(shape, B, L)


    # Define a function to calculate based on shape
    Nc = number4
    Nq = number5
    Nγ = number6
    c = number8
    γ = number7
    q=γ  * D
    qult = sc * c * Nc + q * Nq + sγ * 0.5 * γ * B *Nγ     #qult : Ultimate bearing capacity of soil (KN/m2)
    shelter_data['Depth of the foundation'] = shelter_data['Depth of the foundation'].apply(lambda x: 0.10 if x < 0.10 else x)
    
    UCS = c * 2
    shelter_data['Foundation_type'] = calculate_type_of_foundation(UCS)



    #   #   UPLIFT CALCULATION  #   #

    # Constants
    air_density = 1.225  # kg/m^3 (standard air density)
    conversion_factor = 1000 / 3600  # Conversion from km/h to m/s

    # Convert wind speed from km/h to m/s
    wind_speed_m_s = highest_wind_speed * conversion_factor

    # Calculate wind pressure
    wind_pressure = 0.5 * air_density * wind_speed_m_s ** 2

    # Calculate wind load (force) on the surface
    

    #   #   UPLIFT CALCULATION  #   #
    shelter_data['Uplift'] = shelter_data.apply(lambda row: calculate_uplift_decision(wind_pressure, row['length'], row['centre height'], row['width'], row['package weight']), axis =1)


    return shelter_data



    # #    #   BEARING CAPACITY EQUATION (TERZAGHI) CHECK   #     #
