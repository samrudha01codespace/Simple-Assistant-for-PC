import requests

# Replace 'your_api_key' with your actual API key from HERE.com
api_key = ''

# Example endpoint for the HERE Geocoding API
base_url = 'https://geocode.search.hereapi.com/v1/geocode'

# Example address to geocode
address = 'Invalidenstraße 116, 10115 Berlin, Germany'

# Parameters for the API request
params = {
    'q': address,  # Address query
    'apiKey': api_key,  # Your API Key
    'lang': 'en-US',  # Optional: Response language
    'limit': 1  # Optional: Limit number of results
}

# Make the request to the HERE API
response = requests.get(base_url, params=params)

# Check if the request was successful
if response.status_code == 200:
    # Parse the JSON response
    data = response.json()
    # Print the formatted result
    print(data)
else:
    print(f"Error: {response.status_code}")
    print(response.text)  # Print the error message returned by the API
