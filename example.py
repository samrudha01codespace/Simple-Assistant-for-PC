import requests
import folium

# Function to fetch flight data
def fetch_flight_data():
    url = "https://opensky-network.org/api/states/all"
    response = requests.get(url)
    data = response.json()
    return data['states']

# Function to create a map with flight data
def create_flight_map(flights):
    m = folium.Map(location=[20, 0], zoom_start=2)
    for flight in flights:
        lat = flight[6]
        lon = flight[5]
        if lat and lon:
            folium.Marker([lat, lon], popup=flight[1]).add_to(m)
    return m

# Fetch flight data
flights = fetch_flight_data()

# Create and save the flight map
flight_map = create_flight_map(flights)

flight_map.open("flight_tracking_map.html")
