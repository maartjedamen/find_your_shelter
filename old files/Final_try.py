import pandas as pd
import math as math
import plotly.graph_objects as go

# Create a mapping of words to numeric values
performance_mapping = {
        'GREEN': 1,
        'AMBER': 2,
        'RED': 0
}


def create_plot(Shelters):

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


def shelter_location_analysis():

    df = pd.read_csv('data/location_information_long.csv')
    data=pd.DataFrame(df)

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

                print(f"Population for Shelter Location {shelter_location + 1}: {population}")

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
    return Shelters, highest_wind_speed

if __name__ == '__main__':
    Shelters, _ = shelter_location_analysis()
    