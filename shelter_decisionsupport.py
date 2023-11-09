from shelter import StructuralCalculation as sc 
from shelter import Analysis
from shelter import DecisionSupport as ds

if __name__ == '__main__':
    Shelters, highest_wind_speed, shelter_location = Analysis.shelter_location_analysis()
    shelter_data = sc.compute_structural_calculations(Shelters, highest_wind_speed, shelter_location)
    app = ds.run_dash(shelter_data)
    app.run_server(debug=False, threaded=True)
    exit(0)