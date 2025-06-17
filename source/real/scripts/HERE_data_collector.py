import os
import requests
import json
import time

# Define your API key
API_KEY = 'YOURAPIKEYHERE'
REAL_FLOW_DATA_FOLDER = r"./realFlowData"

if not os.path.exists(REAL_FLOW_DATA_FOLDER):
    os.makedirs(REAL_FLOW_DATA_FOLDER)

istanbulCoords = {"north_latitude": 41.3785, "east_longitude": 29.4260, "south_latitude": 40.8028, "west_longitude": 28.5511}
lowerManhattan2 = {"north_latitude": 40.71988, "east_longitude": -73.99549, "south_latitude": 40.70111, "west_longitude": -74.02081}

selected = lowerManhattan2

url = f'https://data.traffic.hereapi.com/v7/flow?in=bbox:{selected["west_longitude"]},{selected["south_latitude"]},{selected["east_longitude"]},{selected["north_latitude"]}&locationReferencing=shape&apiKey={API_KEY}'

def fetch_and_save_data():
    try:
        # Make the request
        response = requests.get(url)

        # Check the response
        if response.status_code == 200:
            traffic_data = response.json()

            # Save as JSON with date and time 
            json_file = os.path.join(REAL_FLOW_DATA_FOLDER, f'traffic_data_{time.strftime("%Y%m%d_%H%M%S")}.json')
            with open(json_file, 'w') as f:
                json.dump(traffic_data, f)

            # Print success message
            print(f"Data saved to {json_file}")

        else:
            print(f"Error: {response.status_code}, {response.text}")

    except Exception as e:
        print(f"An error occurred: {e}")

# Keep the script running
while True:
    fetch_and_save_data()
    time.sleep(5 * 60)
