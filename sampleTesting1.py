import requests

# Replace 'your_api_key' with your actual API key from HERE.com
api_key = ''

# Base URL for the HERE Routing API
base_url = 'https://router.hereapi.com/v8/routes'

# Define start and end coordinates
start_latitude = 52.5160  # Example: Berlin, Germany
start_longitude = 13.3779
end_latitude = 52.5200  # Example: Berlin, Germany
end_longitude = 13.4050

# Parameters for the API request
params = {
    'transportMode': 'car',  # Mode of transportation: car, pedestrian, truck, etc.
    'origin': f'{start_latitude},{start_longitude}',  # Start coordinates
    'destination': f'{end_latitude},{end_longitude}',  # End coordinates
    'return': 'summary,polyline',  # Return route summary and polyline for map visualization
    'apiKey': api_key
}

# Make the request to the HERE Routing API
response = requests.get(base_url, params=params)

# Check if the request was successful
if response.status_code == 200:
    # Parse the JSON response
    route_data = response.json()
    
    # Extract relevant route information
    route = route_data['routes'][0]
    summary = route['sections'][0]['summary']
    
    print(f"Distance: {summary['length']} meters")
    print(f"Travel Time: {summary['duration']} seconds")
    print(f"Polyline: {route['sections'][0]['polyline']}")  # Polyline for map visualization
else:
    print(f"Error: {response.status_code}")
    print(response.text)  # Print the error message returned by the API
