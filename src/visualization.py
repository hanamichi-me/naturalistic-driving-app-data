# visualization.py
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
import os
import pandas as pd


def generate_js_variables(df_resampled, intersections_df, traffic_signal_df, api_key, car_url, intersectionMarkers_url, trafficSignal_url, car_icon_size, intersection_size, trafficSignal_size):
    # Convert speed to a string list for JavaScript
    speeds_js = df_resampled['speed_kmh'].apply(lambda x: str(x)).tolist()
    speeds_js_string = ", ".join(speeds_js)
    
    # Generate route coordinates for JavaScript
    route_coordinates_js = df_resampled.apply(
        lambda row: f"new google.maps.LatLng({row['Latitude']}, {row['Longitude']})", axis=1
    ).tolist()
    route_coordinates_js_length = len(route_coordinates_js)
    route_coordinates_js_str = "[" + ",\n".join(route_coordinates_js) + "]"
    
    # Intersections
    intersection_markers_js = intersections_df.apply(
        lambda row: f"new google.maps.LatLng({row['Y']}, {row['X']})", axis=1
    ).tolist()
    intersection_markers_js_str = "[" + ",\n".join(intersection_markers_js) + "]"
    
    # Traffic signals
    traffic_signal_markers_js = traffic_signal_df.apply(
        lambda row: f"new google.maps.LatLng({row['Y']}, {row['X']})", axis=1
    ).tolist()
    traffic_signal_markers_js_str = "[" + ",\n".join(traffic_signal_markers_js) + "]"

    api_key = api_key
    car_url = car_url
    intersectionMarkers_url = intersectionMarkers_url
    trafficSignal_url = trafficSignal_url
    car_icon_size = car_icon_size
    trafficSignal_size = trafficSignal_size
    intersection_size = intersection_size
    
    return {
        'speeds_js_string': speeds_js_string,
        'route_coordinates_js_str': route_coordinates_js_str,
        'route_coordinates_js_length': route_coordinates_js_length,
        'intersection_markers_js_str': intersection_markers_js_str,
        'traffic_signal_markers_js_str': traffic_signal_markers_js_str,
        'api_key': api_key,
        'car_url' : car_url,
        'intersectionMarkers_url': intersectionMarkers_url,
        'trafficSignal_url': trafficSignal_url,
        'car_icon_size': car_icon_size,
        'intersection_size' : intersection_size,
        'trafficSignal_size': trafficSignal_size
    }


def generate_html_template(js_variables):
    return f"""
    <!DOCTYPE html>
    <html>
      <head>
        <meta name="viewport" content="initial-scale=1.0, user-scalable=no" />
        <meta http-equiv="content-type" content="text/html; charset=UTF-8" />
        <title>Google Maps Car Route Simulation with Speedometer</title>
        <script src="https://maps.googleapis.com/maps/api/js?key={js_variables['api_key']}"></script>
        <style>
            body {{
                font-family: Arial, sans-serif;
            }}
            #map {{
                width: 60%;
                height: 500px;
                float: left;
            }}
            #speedometer {{
                width: 35%;
                height: 500px;
                float: right;
                display: flex;
                align-items: center;
                justify-content: center;
                flex-direction: column;
            }}
            #speed-display {{
                font-size: 48px;
                font-weight: bold;
                color: #FF0000;
            }}
            #index-display {{
                font-size: 36px;
                font-weight: bold;
                color: #0000FF;
            }}
            .speed-label {{
                font-size: 18px;
                color: #333;
            }}
            .index-label {{
                font-size: 18px;
                color: #333;
            }}
            #progress-bar {{
                width: 90%;
                margin-top: 20px;
            }}
        </style>
      </head>
      <body onload="initMap()">
        <h3>Car Route Simulation with Speedometer</h3>
        <div id="map"></div>
        <div id="speedometer">
            <div>
                <div class="speed-label">Current Speed (km/h)</div>
                <div id="speed-display">0</div>
            </div>
            <div>
                <div class="index-label">Current Index</div>
                <div id="index-display">0</div>
            </div>
            <input type="range" id="progress-bar" min="0" max="{js_variables['route_coordinates_js_length']}" value="0">
        </div>
    
        <script>
            var routeCoordinates = {js_variables['route_coordinates_js_str']};
            var intersectionMarkers = {js_variables['intersection_markers_js_str']};
            var Traffic_Signal_markers = {js_variables['traffic_signal_markers_js_str']};
            var calculatedSpeeds = [{js_variables['speeds_js_string']}];
            var map, marker, intervalId, index = 0;
    
            function haversine(lon1, lat1, lon2, lat2) {{
                var R = 6371.0;
                var dlon = lon2 - lon1;
                var dlat = lon2 - lon1;
                var a = Math.sin(dlat / 2) * Math.sin(dlat / 2) +
                        Math.cos(lat1) * Math.cos(lat2) * Math.sin(dlon / 2) * Math.sin(dlon / 2);
                var c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
                var distance = R * c;
                return distance;
            }}
    
            function updateSpeedometer(speed, index) {{
                document.getElementById('speed-display').textContent = speed.toFixed(2);
                document.getElementById('index-display').textContent = index;
                document.getElementById('progress-bar').value = index;
            }}
    
            function initMap() {{
                var mapOptions = {{
                    zoom: 19,
                    center: routeCoordinates[0],
                    tilt: 45
                }};
                map = new google.maps.Map(document.getElementById('map'), mapOptions);
    
                marker = new google.maps.Marker({{
                    position: routeCoordinates[0],
                    map: map,
                    icon: {{
                        url: "{js_variables['car_url']}",
                        scaledSize: new google.maps.Size({js_variables['car_icon_size'][0]}, {js_variables['car_icon_size'][1]})
                    }}
                }});
    
                intersectionMarkers.forEach(function(intersection) {{
                    new google.maps.Marker({{
                        position: intersection,
                        map: map,
                        icon: {{
                            url: "{js_variables['intersectionMarkers_url']}",
                            scaledSize: new google.maps.Size({js_variables['intersection_size'][0]}, {js_variables['intersection_size'][1]})
                        }}
                    }});
                }});
    
                Traffic_Signal_markers.forEach(function(signal) {{
                    new google.maps.Marker({{
                        position: signal,
                        map: map,
                        icon: {{
                            url: "{js_variables['trafficSignal_url']}",
                            scaledSize: new google.maps.Size({js_variables['trafficSignal_size'][0]}, {js_variables['trafficSignal_size'][1]})
                        }}
                    }});
                }});
    
                var routePath = new google.maps.Polyline({{
                    path: routeCoordinates,
                    geodesic: true,
                    strokeColor: '#FF0000',
                    strokeOpacity: 1.0,
                    strokeWeight: 2
                }});
                routePath.setMap(map);
    
                moveCar();
            }}
    
            function moveCar() {{
                if (index < routeCoordinates.length - 1) {{
                    var distance = haversine(routeCoordinates[index].lng(), routeCoordinates[index].lat(),
                                             routeCoordinates[index + 1].lng(), routeCoordinates[index + 1].lat());
                    var speed = calculatedSpeeds[index];
                    var timeInterval = (distance / speed) * 3600000;
                    timeInterval = timeInterval / 100;
    
                    index++;
                    marker.setPosition(routeCoordinates[index]);
                    updateSpeedometer(speed, index);
                    map.setCenter(routeCoordinates[index]);
    
                    clearInterval(intervalId);
                    intervalId = setInterval(moveCar, timeInterval);
                }}
            }}
    
            document.getElementById('progress-bar').addEventListener('input', function() {{
                clearInterval(intervalId);
                index = parseInt(this.value);
                marker.setPosition(routeCoordinates[index]);
                map.setCenter(routeCoordinates[index]);
                var speed = calculatedSpeeds[index];
                updateSpeedometer(speed, index);
            }});
    
            document.getElementById('progress-bar').addEventListener('change', function() {{
                moveCar();
            }});
        </script>
      </body>
    </html>
    """

def load_map_configurations(config):
    """
    Load intersection and traffic signal data, as well as Google Maps and icon configurations from the config file.
    
    Args:
        config (dict): Configuration dictionary loaded from params.yaml.
        
    Returns:
        tuple: Contains intersections_df, traffic_signal_df, and Google Maps configuration variables.
    """

    # Load intersection and traffic signal data
    intersections_df = pd.read_csv(config['data']['intersections'])
    traffic_signal_df = pd.read_csv(config['data']['traffic_signals'])

    # Intersections.csv without Traffic_Signal
    intersections_no_signal_df = intersections_df[~intersections_df['NODE_NAME'].isin(traffic_signal_df['NODE_NAME'])]
    intersections_no_signal_df.to_csv(os.path.join(config['paths']['main_dir2'],config['data']['intersections_no_signal']), index=False)

    # Google Maps and Icon configuration
    api_key = config['google_maps']['api_key']
    car_url = config['google_maps']['car_url']
    intersectionMarkers_url = config['google_maps']['intersectionMarkers_url']
    trafficSignal_url = config['google_maps']['trafficSignal_url']
    car_icon_size = config['google_maps']['car_icon_size']
    intersection_size = config['google_maps']['intersection_size']
    trafficSignal_size = config['google_maps']['trafficSignal_size']

    return (intersections_df, intersections_no_signal_df, traffic_signal_df, api_key, car_url, intersectionMarkers_url, 
            trafficSignal_url, car_icon_size, intersection_size, trafficSignal_size)
    

def generate_trip_map(df_resampled, output_path, config):
    """
    Generates and saves an unfiltered trip map using Google Maps API and traffic data.

    Args:
        df_resampled (pd.DataFrame): Resampled trip data.
        config (dict): Configuration dictionary loaded from params.yaml.
    
    Returns:
        str: Path to the generated HTML file.
    """
    (intersections_df, intersections_no_signal_df, traffic_signal_df, api_key, car_url, intersectionMarkers_url, 
     trafficSignal_url, car_icon_size, intersection_size, trafficSignal_size) = load_map_configurations(config)

    # Generate JavaScript variables based on the data
    js_variables = generate_js_variables(
        df_resampled, intersections_df, traffic_signal_df, api_key, 
        car_url, intersectionMarkers_url, trafficSignal_url, 
        car_icon_size, intersection_size, trafficSignal_size
    )

    # Generate the HTML template using the variables
    html_template = generate_html_template(js_variables)

    # Define the output directory and file path

    # output_path = os.path.join(config['paths']['main_dir3'], config['data']['unfiltered_trip_html'])

    # Save the HTML file
    with open(output_path, 'w') as f:
        f.write(html_template)

    return output_path



def plot_prediction_vs_distance(model, scaler, X_test, target_name, feature_name='directional_distance_to_intersection', 
                                distance_range=(-0.03, 0.03), steps=100, ax=None, save_path=None):
    
    if ax is None:
        fig, ax = plt.subplots()

    unique_distances = np.linspace(distance_range[0], distance_range[1], steps)
    y_plot_mean = []
    X_plot_list = []
    for distance in unique_distances:
        X_temp = X_test.copy()

        
        if feature_name in X_temp.columns:
            X_temp[feature_name] = distance
        else:
            X_temp = X_temp.assign(**{feature_name: distance})

        X_plot_list.append(X_temp)

    X_plot = pd.concat(X_plot_list)

    
    X_plot_scaled = scaler.transform(X_plot)
    X_plot_scaled = pd.DataFrame(X_plot_scaled, columns=X_plot.columns)

    
    y_plot = model.predict(X_plot_scaled)

    for i in range(len(unique_distances)):
        start_index = i * len(X_test)
        end_index = (i + 1) * len(X_test)
        y_plot_mean.append(np.mean(y_plot[start_index:end_index]))

    
    y_plot_mean_smooth = gaussian_filter1d(y_plot_mean, sigma=2)

    ax.plot(unique_distances * 1000, y_plot_mean_smooth, label=f'Smoothed Predicted {target_name}', color='blue')
    ax.plot(unique_distances * 1000, y_plot_mean, alpha=1, linestyle=':', label=f'Original Predicted {target_name}', color='darkorange',linewidth=2)
    ax.set_xlabel('Distance to Intersection')
    ax.set_ylabel(f'Predicted {target_name}')
    ax.set_title(f'Predicted {target_name} vs Distance to Intersection')
    ax.axvline(x=0, color='r', linestyle='--', label='Intersection')
    ax.legend()
    # Save the plot if a save path is provided
    if save_path:
        plt.savefig(save_path, dpi=80, bbox_inches='tight')
    # Show the plot
    # plt.show()



def plot_comparison(combined_df, target_name, save_path=None):
    """
    Plot the comparison of smoothed actual and predicted values along the cumulative distance.

    Args:
        combined_df (DataFrame): DataFrame with actual and predicted values and distances.
        target_name (str): The name of the target variable being compared (e.g., 'speed_kmh', 'acceleration').

    Returns:
        None
    """
    bins = np.linspace(combined_df['cumulative_distance'].min(), combined_df['cumulative_distance'].max(), 100)
    combined_df['distance_bin'] = pd.cut(combined_df['cumulative_distance'], bins)
    mean_actual_value = combined_df.groupby('distance_bin')[f'actual_{target_name}'].mean().reset_index(drop=True)
    mean_predicted_value = combined_df.groupby('distance_bin')[f'predicted_{target_name}'].mean().reset_index(drop=True)
    mean_distances = [interval.mid for interval in combined_df['distance_bin'].cat.categories]
    mean_actual_value_smooth = gaussian_filter1d(mean_actual_value, sigma=2)
    mean_predicted_value_smooth = gaussian_filter1d(mean_predicted_value, sigma=2)

    plt.figure(figsize=(12, 8))
    plt.plot(mean_distances, mean_actual_value_smooth, label=f'Smoothed Actual {target_name}', color='blue', linewidth=2)
    plt.plot(mean_distances, mean_predicted_value_smooth, linestyle='--', label=f'Smoothed Predicted {target_name}', color='red', linewidth=2)
    plt.xlabel('Distance between Intersections (meters)')
    plt.ylabel(f'{target_name.capitalize()}')
    plt.title(f'Smoothed Actual vs Predicted {target_name.capitalize()} between Intersections')
    plt.legend()
    if save_path:
        plt.savefig(save_path, dpi=80, bbox_inches='tight')
    plt.show()
