# map_generation.py
# map_generation.py: This file generates a Google Map displaying multiple trip routes in different colors, based on GPS data stored in .txt files.

import os
import pandas as pd
import gmplot
import matplotlib.cm as cm
import numpy as np
from IPython.display import IFrame

def generate_multiple_trips_map(main_dir, output_file):
    gmap_initialized = False
    gmap = None

    trip_files = []

    for root, dirs, files in os.walk(main_dir):
        for file in files:
            if file.endswith('.txt'):  
                file_path = os.path.join(root, file)
                trip_files.append(file_path)

    num_trips = len(trip_files)
    colors = cm.rainbow(np.linspace(0, 1, num_trips))  


    for i, file_path in enumerate(trip_files):

        trip_df = pd.read_csv(file_path)

        latitudes = trip_df['Latitude'].tolist()
        longitudes = trip_df['Longitude'].tolist()



        if not gmap_initialized:
            
            latitude_center = latitudes[0]
            longitude_center = longitudes[0]
            gmap = gmplot.GoogleMapPlotter(latitude_center, longitude_center, 14, apikey='')
            gmap_initialized = True


        current_color = '#{:02x}{:02x}{:02x}'.format(
            int(colors[i][0] * 255), int(colors[i][1] * 255), int(colors[i][2] * 255)
        )


        gmap.plot(latitudes, longitudes, color=current_color, edge_width=7)

    gmap.draw(output_file)
    print(f"Map created successfully with each trip in a different color! Saved to {output_file}")


