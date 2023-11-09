import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cartopy.crs as ccrs
import geopandas as gpd
import shapely as shp

import math
import itertools
import rasterio

import osmnx as ox
import requests
import plotly.graph_objects as go
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import base64


class Locationgenerator: 

    def visualization_network_and_footprints(network, footprints, bbox = None, edge=False):
        if edge==False: 
        
            fig1, ax1 = ox.plot_footprints(footprints, color='#843700', bgcolor='#f4efec', show=False, close=False, figsize=(10,10))
        elif edge==True: 
            fig1, ax1 = ox.plot_footprints(footprints, color='#f4efec', edge_color='#843700', edge_linewidth=1,  bgcolor='#f4efec', show=False, close=False, figsize=(10,10))

        ox.plot_graph(network, ax=ax1,  bgcolor='w', node_size=0, edge_color= '#c48b73', show=True, close=False, figsize=(10,10), bbox=bbox)

    def visualization_network_and_places(network, locations):

        fig, ax = ox.plot_graph(network, show=False, close=False, bgcolor='#f4efec', node_size=0, edge_color= '#c48b73',figsize=(10,10))
        locations.plot(ax=ax, markersize = 20, color = "#843700" , alpha=1)

class PopulationMapGenerator:
    def __init__(self, data_file):
        self.df = pd.read_csv(data_file)
        self.min_lat = 36.169400
        self.max_lat = 36.203219
        self.min_lon = 36.112595
        self.max_lon = 36.162786

    def filter_data(self):
        self.df = self.df[(self.df['latitude'] >= self.min_lat) & (self.df['latitude'] <= self.max_lat) & 
                          (self.df['longitude'] >= self.min_lon) & (self.df['longitude'] <= self.max_lon)]

    def generate_population_map(self, output_file):
        fig = plt.figure(figsize=(12, 9))
        ax = plt.axes(projection=ccrs.PlateCarree())
        
        ax.coastlines()
        
        sites_lat_y = self.df['latitude'].tolist()
        sites_lon_x = self.df['longitude'].tolist()
        population = self.df['population_2020'].tolist()

        ax.scatter(sites_lon_x, sites_lat_y, s=population, alpha=0.3, c='#843700')


        plt.title('Population', fontsize=20)
        plt.savefig(output_file)
        plt.show()

    def get_population_geo_df(self):
        # Assuming that your population data columns are named 'latitude' and 'longitude'
        geometry = gpd.points_from_xy(self.df['longitude'], self.df['latitude'], crs='EPSG:4326')
        population_gdf = gpd.GeoDataFrame(self.df, geometry=geometry)
        return population_gdf

class SlopeFunctions: 
    

    def find_maximum_slope_among_points(coordinates, dem_file):

        def calculate_slope_between_points(lat1, lon1, lat2, lon2, dem_file):
                
            def calculate_slope_degrees(elevation1, elevation2, distance):
                # Calculate the vertical change in elevation
                delta_elevation = next(elevation2) - next(elevation1)

                # Calculate the slope angle in radians
                slope_radians = math.atan(delta_elevation / distance)

                # Convert the slope angle from radians to degrees
                slope_degrees = math.degrees(slope_radians)

                return slope_degrees
        

            def haversine(lat1, lon1, lat2, lon2):
                # Radius of the Earth in meters
                earth_radius = 6371000  # approximately 6,371 km

                # Convert latitude and longitude from degrees to radians
                lat1 = math.radians(lat1)
                lon1 = math.radians(lon1)
                lat2 = math.radians(lat2)
                lon2 = math.radians(lon2)

                # Haversine formula
                dlat = lat2 - lat1
                dlon = lon2 - lon1
                a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
                c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
                distance = earth_radius * c

                return distance
            
            with rasterio.open(dem_file) as src:
                # Calculate the vertical change in elevation and horizontal distance
                elevation1 = src.sample([(lon1, lat1)])
                elevation2 = src.sample([(lon2, lat2)])
                distance = haversine(lat1, lon1, lat2, lon2)

                # Calculate slope angle in degrees
                slope_degrees = calculate_slope_degrees(elevation1, elevation2, distance)

                return slope_degrees
    
        max_slope = 0  # Initialize with a minimum value

        for pair in itertools.combinations(coordinates, 2):
            lat1, lon1 = pair[0]
            lat2, lon2 = pair[1]

            slope_degrees = calculate_slope_between_points(lat1, lon1, lat2, lon2, dem_file)
            max_slope = max(max_slope, slope_degrees)

        return max_slope

class WeatherData: 
        
    def get_weather_API(coord):
        # These are the variables that are included in the URL that we use.
        # Other options are also possible, see open-meteo.com for more data.
        variables = ["temperature_2m_max", "temperature_2m_min", "rain_sum", "snowfall_sum", "windspeed_10m_max", "winddirection_10m_dominant"]

        # Create empty lists to store the means for each variable
        
        urllat = coord[0]
        urllon = coord[1]

        # Set URL for different datatypes at each location
        # open_meteo_url = f"http://archive-api.open-meteo.com/v1/archive?latitude={urllat}&longitude={urllon}&start_date=2022-01-01&end_date=2023-01-01&daily={','.join(variables)}&timezone=Africa/Cairo"
        open_meteo_url = f"http://archive-api.open-meteo.com/v1/archive?latitude={urllat}&longitude={urllon}&start_date=2018-01-01&end_date=2023-01-01&daily={','.join(variables)}&timezone=Africa%2FCairo" 
        # Request data from the URL
        response = requests.get(open_meteo_url)
        data = response.json()

        return data, variables
    
    def get_weather_data(coord, data, variables):
        weather_values = {var: [] for var in variables}
        # weather_values['lat'] = []  # List to store latitude values
        # weather_values['lon'] = []  # List to store longitude values
        weather_values['max_temp'] = []
        weather_values['min_temp'] = []
        weather_values['max_windspeed'] = []

        # weather_values['lat'].append(coord[0])
        # weather_values['lon'].append(coord[1])

        for var in variables:
            daily_data = data['daily']
            if var in daily_data:
                daily_var = daily_data[var]
                mean_value = np.mean(daily_var)
                weather_values[var].append(mean_value)
        
        def find_temperature_statistics(data, upper_bound = True):
            
            # If the temperature occurs more than 3 times per year on average, it is considered not tolerable as an extreme temperature (this is an estimate).  
            day_counts = len(data)
            min_occurrences = day_counts*3/365

            # Count the occurrences of each temperature
            temperatures_int = [int(x) for x in data]
            temperature_counts_around_range = pd.Series(temperatures_int).value_counts()

            # Filter temperatures that occur more than a certain number of days around the specified range
            filtered_temperatures_around_range = temperature_counts_around_range[temperature_counts_around_range > min_occurrences]

            if upper_bound == True: 
                return filtered_temperatures_around_range.index.max()
            else: 
                return filtered_temperatures_around_range.index.min()
            
        max_temp_data = data['daily']["temperature_2m_max"]    
        weather_values['max_temp'].append(find_temperature_statistics(max_temp_data, True))

        min_temp_data = data['daily']["temperature_2m_min"]
        weather_values['min_temp'].append(find_temperature_statistics(min_temp_data, False))

        max_windspeed_data = data['daily']["windspeed_10m_max"]    
        weather_values['max_windspeed'].append(find_temperature_statistics(max_windspeed_data, True))


        # print(weather_values)
        # Create a DataFrame to store the data
        df = pd.DataFrame(weather_values)

        return df
    

class Analysis: 

    def shelter_location_analysis():
        performance_mapping = {
                'GREEN': 1,
                'AMBER': 2,
                'RED': 0
        }
        df = pd.read_csv('data/location_information_long.csv')
        data=pd.DataFrame(df)

        def create_plot(Shelters):
            # Create a mapping of words to numeric values
                performance_mapping = {
                        'GREEN': 1,
                        'AMBER': 2,
                        'RED': 0
                }

                # Calculate the range for each dimension
                Shelter_names = [1, len(Shelters)]
                sepal_length_range = [(Shelters['Number of people'].min() - 10 * Shelters['Number of people'].min() / 100), (Shelters['Number of people'].max() + 10 * Shelters['Number of people'].max() / 100)]
                sepal_width_range = [(Shelters['TOTAL COST'].min() - 10 * Shelters['TOTAL COST'].min() / 100), (Shelters['TOTAL COST'].max() + 10 * Shelters['TOTAL COST'].max() / 100)]
                petal_length_range = [(Shelters['Working people'].min() - 10 * Shelters['Working people'].min() / 100), (Shelters['Working people'].max() + 10 * Shelters['Working people'].max() / 100)]
                Shelter_number_range = [(Shelters['Number of shelters'].min() - 10 * Shelters['Number of shelters'].min() / 100), (Shelters['Number of shelters'].max() + 10 * Shelters['Number of shelters'].max() / 100)]
                Life_span_range = [(Shelters['minimum lifespan'].min()- 10 * Shelters['minimum lifespan'].min() / 100), (Shelters['minimum lifespan'].max() + 10 * Shelters['minimum lifespan'].max() / 100)]
                # print(Shelter_names)

                dimensions = [
                
                    dict(range=Shelter_names, label='Shelter name', tickvals=Shelters['Shelter Number'], ticktext=Shelters['Variables'], values=Shelters['Shelter Number']),
                    dict(range=sepal_length_range,  label='Number of people', values=Shelters['Number of people']),
                    dict(range=sepal_width_range, label='Total Cost (€) ', values=Shelters['TOTAL COST']),
                    dict(range=petal_length_range, label='Working people', values=Shelters['Working people']),
                    dict(range=Shelter_number_range, label='Number of shelters', values=Shelters['Number of shelters']),
                    dict(range=[0,2],tickvals=[0, 1, 2],ticktext=['RED', 'AMBER', 'GREEN'], label='Flood Performance', values=Shelters['FLOOD PERFORMANCE'].replace(performance_mapping)),
                    dict(range=[0,2],tickvals=[0, 1, 2],ticktext=['RED', 'AMBER', 'GREEN'], label='Earthquake Performance', values=Shelters['EARTHQUAKE PERFORMANCE'].replace(performance_mapping)),
                    dict(range=Life_span_range,label='Minimum Life Span (years)', values=Shelters['minimum lifespan']),
                ]

                fig = go.Figure(data=go.Parcoords(
                    line=dict(
                        color=Shelters['Performance Numeric'],  # Use the numeric values
                        colorscale='Viridis',
                        showscale=True,
                        colorbar_title='Wind Performance'
                    ),
                    dimensions=dimensions  # Pass the list of dimensions here
                    
                ))

                # Adjust the size to fit all the labels
                fig.update_layout(
                    plot_bgcolor='white',
                    paper_bgcolor='white',
                    title='Parallel Coordinates Plot for Shelter Data'
                    
                )

                fig.show() #Reference:https://www.analyticsvidhya.com/blog/2021/11/visualize-data-using-parallel-coordinates-plot/


        # print(data.columns)
        #   #  BE ASKED ABOUT THE LOCATION  #   #

        # Create a loop to ask for the shelter location number
        while True:
            shelter_location = int(input(f"Enter the shelter location number (1 to {len(data)}): "))

            try:
                shelter_location = int(shelter_location)
                if 1 <= shelter_location <=len(data):
                    # Assuming 'shelter_location' is 1-based, subtract 1 to make it 0-based
                    shelter_location -= 1

                    # Extract the relevant row from the DataFrame
                    selected_row = data.iloc[shelter_location]


                    #    #     LOCATION DATA   #    #  #  

                    population = int(selected_row['population_2020']) #Population Reference:https://data.humdata.org/dataset/cod-ps-tur
                    area_location = int(selected_row['area'])
                

                    #    #     LOCATION DATA   #    #  #  



                    #    #     WEATHER DATA   #    #  #  Reference:https://open-meteo.com/
                    
                    highest_wind_speed = int(selected_row['windspeed_10m_max'])
                    max_temperature = int(selected_row['max_temp'])
                    min_temperature =int(selected_row['min_temp'])
                    snow_fall = int(selected_row['snowfall_sum'])
                    rain_fall = int(selected_row['rain_sum'])
                    wind_direction = int(selected_row['winddirection_10m_dominant'])
            

                    # #    #     WEATHER DATA   #    #  #  Reference:https://open-meteo.com/
                    shelter_location += 1
                    print(f"Population for Shelter Location {shelter_location}: {population}")

                    # Exit the loop if a valid input is provided
                    break
                else:
                    print("Please enter a number between 1 and 6.")
            except ValueError:
                print("Invalid input. Please enter a valid number.")


        # #   #  BE ASKED ABOUT THE LOCATION  #   #


        #    #     LOCATION DATA   #    #  #  


        family_size = 3.5 #https://www.globaldata.com/data-insights/macroeconomic/average-household-size-in-turkey-2096131/#:~:text=Size%20in%20Turkey-,Turkey%20had%20an%20average%20household%20size%20of%203.34%20people%20in,2021%2C%20between%202010%20and%202021


        #   #               SHELTER DATA           #    #

        # Read the Excel file for shelters
        df = pd.read_excel('data\Shelters.xlsx')

        # Remove the first row from every column
        df = df.iloc[1:, :]
        df.reset_index(drop=True, inplace=True)




        #   # WIND ANALYSIS #   #

        # # Assuming you've already loaded your data into the 'df' DataFrame
        # # Assuming 'WIND PROOF' indicates 'low' for wind speeds < 110 km/h
        low_wind_shelters = df[df['WIND PROOF'] == 'LOW']

        # # Assuming 'WIND PROOF' indicates 'middle' for wind speeds between 110 and 160 km/h
        middle_wind_shelters = df[df['WIND PROOF'] == 'MEDIUM']

        # # Assuming 'WIND PROOF' indicates 'high' for wind speeds > 160 km/h
        high_wind_shelters = df[df['WIND PROOF'] == 'HIGH']

        # Initialize empty lists for each category
        low_list = []
        medium_list = []
        high_list = []

        # Create a list for shelters with no 'WIND PROOF' value
        no_wind_proof_shelters = df[df['WIND PROOF'].isna()]

        # Check highest_wind_speed and categorize shelters accordingly
        if highest_wind_speed < 110:
            low_list = pd.concat([low_wind_shelters, middle_wind_shelters, high_wind_shelters, no_wind_proof_shelters])

        else:
            if highest_wind_speed >= 110 and highest_wind_speed <= 160:
                medium_list = pd.concat([middle_wind_shelters, high_wind_shelters])

            else:
                if highest_wind_speed > 160:
                    high_list = high_wind_shelters

        # Here we get the list with the appropriate shelters for this wind speed in the area
        result_shelter = []
        # Check highest_wind_speed and assign the appropriate list
        if highest_wind_speed < 110:
            result_shelter = low_list
        else:
            if highest_wind_speed >= 110 and highest_wind_speed <= 160:
                result_shelter = medium_list
            else:
                result_shelter = high_list  # or high_list, depending on your condition


        # Create a list with the appropriate shelters for this wind speed (green) and one with the acceptable (amber)
        # Check if result is not None (meaning you have a valid list)
        if result_shelter is not None:
            # Filter shelters from the result list based on 'WIND PERFORMANCE'
            green_wind_shelters = result_shelter[result_shelter['WIND PERFORMANCE'] == 'GREEN']
            amber_wind_shelters = result_shelter[result_shelter['WIND PERFORMANCE'] == 'AMBER']

            # Now you have two separate lists: green_wind_shelters and amber_wind_shelters
        else:
            print("Result list is None, please check the highest_wind_speed condition.")

        #   # WIND ANALYSIS #   #



        #   # TEMPERATURE ANALYSIS #   #
        # Create an empty list to store full rows where 'temperature low' values are greater than min_temperature
        # Filter green_wind_shelters based on the condition
        T_W_shelter_green = green_wind_shelters[(green_wind_shelters['temperature low'] <= min_temperature) & (green_wind_shelters['temperature high'] >= max_temperature)]
        T_W_shelter_amber = amber_wind_shelters[(amber_wind_shelters['temperature low'] <= min_temperature) & (amber_wind_shelters['temperature high'] >= max_temperature)]

        #   # TEMPERATURE ANALYSIS #   #




        #   # AMOUNT OF PEOPLE ___ FAMILY SIZE #   #

        T_W_shelter_green = T_W_shelter_green.copy()
        T_W_shelter_amber = T_W_shelter_amber.copy()
        T_W_shelter_green['Quantity'] = 1
        T_W_shelter_amber['Quantity'] = 1


        # Calculates the quantity required baesd on the number of persons per shelter
        def quantity(number_of_persons, family_size):
            """
        Calculates the quantity required baesd on the number of persons per shelter 
            """
            return math.ceil(family_size/float(number_of_persons))


        T_W_shelter_green['Quantity'] = T_W_shelter_green.apply(lambda row: quantity(row['amount of persons'], family_size), axis=1)

        T_W_shelter_amber['Quantity'] = T_W_shelter_amber.apply(lambda row: quantity(row['amount of persons'], family_size), axis=1)



        #   # AMOUNT OF PEOPLE ___ FAMILY SIZE #   #







        #   #  NUMBER OF SHELTERS PER LOCATION  #   #

        T_W_shelter_green = T_W_shelter_green.copy()
        T_W_shelter_amber = T_W_shelter_amber.copy()
        T_W_shelter_green['Number of shelters'] = 1
        T_W_shelter_amber['Number of shelters'] = 1


        # Step 2: Calculate the shelter area multiplied by quantity
        T_W_shelter_green['Shelter_Area'] = T_W_shelter_green['area'] * T_W_shelter_green['Quantity']
        T_W_shelter_amber['Shelter_Area'] = T_W_shelter_amber['area'] * T_W_shelter_amber['Quantity']

        # Step 4: Calculate the number of shelters per location
        T_W_shelter_green['Number of shelters'] =  (area_location - 0.1 * area_location) / T_W_shelter_green['Shelter_Area']
        T_W_shelter_green['Number of shelters'] = T_W_shelter_green['Number of shelters'].apply(lambda x: math.floor(x))
        T_W_shelter_amber['Number of shelters'] =  (area_location - 0.1 * area_location) / T_W_shelter_amber['Shelter_Area'] 
        T_W_shelter_amber['Number of shelters'] = T_W_shelter_amber['Number of shelters'].apply(lambda x: math.floor(x))


        #   #  NUMBER OF SHELTERS PER LOCATION  #   #






        #   #  NUMBER OF HOSTED PEOPLE PER LOCATION PER SHELTER  #   #

        T_W_shelter_green = T_W_shelter_green.copy()
        T_W_shelter_amber = T_W_shelter_amber.copy()
        T_W_shelter_green['Number of people'] = 1
        T_W_shelter_amber['Number of people'] = 1

        # Multiply the shelters with the family size to find how many people are going to be hosted
        T_W_shelter_green['Number of people'] =  T_W_shelter_green['Number of shelters'] * family_size
        T_W_shelter_green['Number of people'] = T_W_shelter_green['Number of people'].apply(lambda x: math.floor(x))

        T_W_shelter_amber['Number of people'] =  T_W_shelter_amber['Number of shelters'] * family_size
        T_W_shelter_amber['Number of people'] = T_W_shelter_amber['Number of people'].apply(lambda x: math.floor(x))


        #   #  NUMBER OF HOSTED PEOPLE PER LOCATION PER SHELTER  #   #





        #   #  NUMBER OF WORKING PEOPLE PER LOCATION TO ASSEMBLY IT ON TIME  #   #

        T_W_shelter_green = T_W_shelter_green.copy()
        T_W_shelter_amber = T_W_shelter_amber.copy()
        T_W_shelter_green['Working people'] = 1
        T_W_shelter_amber['Working people'] = 1


        T_W_shelter_green['Working people'] =  T_W_shelter_green['people needed for assembly'] * T_W_shelter_green['Number of shelters']
        T_W_shelter_green['Working people'] = T_W_shelter_green['Working people'].apply(lambda x: math.ceil(x))


        T_W_shelter_amber['Working people'] =  T_W_shelter_amber['people needed for assembly'] * T_W_shelter_amber['Number of shelters']
        T_W_shelter_amber['Working people'] = T_W_shelter_amber['Working people'].apply(lambda x: math.ceil(x))


        #   #  NUMBER OF WORKING PEOPLE PER LOCATION TO ASSEMBLY IT ON TIME  #   #






        #   #  TOTAL COST  #   #

        T_W_shelter_green = T_W_shelter_green.copy()
        T_W_shelter_green['TOTAL COST'] = 1
        T_W_shelter_amber = T_W_shelter_amber.copy()
        T_W_shelter_amber['TOTAL COST'] = 1

        T_W_shelter_green['TOTAL COST'] =  T_W_shelter_green['Number of shelters'] * T_W_shelter_green['cost']
        T_W_shelter_amber['TOTAL COST'] =  T_W_shelter_amber['Number of shelters'] * T_W_shelter_amber['cost']

        #   #  TOTAL COST  #   #




        # Combine the list
        Shelters = pd.concat([T_W_shelter_green, T_W_shelter_amber], ignore_index=True)
        # print(Shelters)





        #   #  VISUALIZATION - PARALLEL COORDINATES PLOT #   #

        
        # Map the 'WIND PERFORMANCE' column to numeric values
        Shelters['Performance Numeric'] = Shelters['WIND PERFORMANCE'].replace(performance_mapping)

        # Create a new column 'Shelter Number' with values ranging from 1 to the total number of shelters
        Shelters['Shelter Number'] = range(1, len(Shelters) + 1)

    
        create_plot(Shelters)
        return Shelters, highest_wind_speed, shelter_location

    if __name__ == '__main__':
        Shelters, _ = shelter_location_analysis()

class StructuralCalculation: 

    def compute_structural_calculations(shelter_data, highest_wind_speed, shelter_location):
        def calculate_uplift_decision(wind_pressure, shelter_depth, shelter_height, shelter_width, shelter_weight):
            def uplift(wind_pressure, shelter_depth, shelter_height, shelter_width):
                # Calculation of uplift force, simplified
                # calculate the torque (momentum) caused by the windload: act like the building is a beam with one fixed side (B3 on the TU Delft structural mechanics formula sheet)
                # 1/2ql**2
                q = wind_pressure * shelter_depth
                l = shelter_height
                torque = 1/2 * q*l**2

                # calculate the Fup and Fdown on the outsides of the shelter; T = R*F
                Fdown = torque/(1/2*shelter_width)
                Fup = -(Fdown)
                return Fup, Fdown
            
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


        #   #  COORDINATES   #    #

        # Initialize Nominatim API to find the coordinates 
        sportsfields = gpd.read_file("data/sportfields.json")

        # sportsfield_centers_gdf = gpd.read_file("data/sportsfield_centers_gdf", bbox=None, mask=None, rows=None, engine=None)
        shelter_locations_polygon = gpd.GeoDataFrame(sportsfields, geometry = sportsfields['geometry'])
        shelter_locations_polygon = shelter_locations_polygon.to_crs(epsg=4326)

        coords = shelter_locations_polygon['geometry'].centroid
        coords = shp.get_coordinates(coords)
        coords = coords.tolist()
 
        longitude = coords[shelter_location][0]
        latitude = coords[shelter_location][1]

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
        df = pd.read_excel('data\Soil data.xlsx')   # Reference for the bearing capacity of each soil : https://dicausa.com/soil-bearing-capacity/
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
        
        def calculate_type_of_foundation(UCS):

            consistency_df = pd.read_csv('data\soil_consistency.csv',sep = ";", encoding='latin')

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

class DecisionSupport:
    # Dash app initialization
    def run_dash(data):
        # Convert the data to a list of dictionaries
        def get_list(data):
            shelter_data = []
            for index, row in data.iterrows():
                L = f"{' * ' + str(round(row['Length of the foundation'],3)) if row['Length of the foundation'] is not None else ''}"
                size_foundation = f"**Size of the foundation:** {round(row['Width of the foundation'],3)} {L} m"
                shelter = {
                    'name': row['Variables'],
                    'description': f"""## Properties:  
                    \n**Length:** {row['length']}m  \n **Width:** {row['width']}m
                    \n\n**Amount of people:** {int(row['amount of persons'])}/shelter \n\n**Lowest temperature:** {row['temperature low']}°C  \n**Highest temperature:** {row['temperature high']}°C  \n**Minimum life span:** {row['minimum lifespan']}  
                    \n\n**Materials:** {row['local resources needed']}  
                    **Foundation type:** {row['Foundation_type']}   \n { size_foundation if row['Foundation_type'] != "No foundation possible, use weights or pad layer" else ''}    
                    \n**Uplift :** {row['Uplift']}  
                    \n**Cost:** {row['cost']} € 
                    [Link]({row['Link with extensive info']})"""  ,
                    'image_path': row['Image File Path'],
                }
                shelter_data.append(shelter)
            return shelter_data
        
        # Define a function to encode images
        def encode_image(image_path):
            with open(image_path, "rb") as image_file:
                encoded_string = base64.b64encode(image_file.read()).decode()
            return f"data:image/jpeg;base64,{encoded_string}"
        app = dash.Dash(__name__)

        # Layout of the app
        shelter_data = get_list(data)

        app.layout = html.Div([
            html.H1("Suitable shelters for the location", style={'text-align': 'center'}),
            dcc.Slider(
                id='Shelter-slider',
                min=0,
                max=len(shelter_data) - 1,
                step=1,
                value=0,
                marks={i: str(i) for i in range(len(shelter_data))},
                
            ),
            html.Div([
                html.Img(id='image-output', style={'display': 'block', 'margin': '70px', 'width': '40%','height': 'auto', 'border-style' : 'solid', }),
                dcc.Markdown(id='shelter-info', style={'border': '1px dashed black','border-radius': '20px', 'width': '350px', 'height': '400px'}),
            ],  style={'text-align': 'center', 'margin-top': '100px', 'display': 'flex', 'flex-direction': 'column', 'align-items': 'center', 'justify-content': 'center', 'height': '80vh'}),
        ])

        # Callback to update the displayed image and shelter info
        @app.callback(
            [Output('image-output', 'src'), Output('shelter-info', 'children')],
            Input('Shelter-slider', 'value')
        )
        def update_image_and_info(selected_value):
            selected_shelter = shelter_data[selected_value]
            selected_image_path = selected_shelter['image_path']
            
            shelter_info = f"**{selected_shelter['name']}**\n\n{selected_shelter['description']}"
            
            return encode_image(selected_image_path), shelter_info
        return app
        app.run_server(debug=True)

        

        