import requests


def get_coordinates(place_name):
    """
    Fetch latitude and longitude for a given place name using Nominatim API
    """
    url = "https://nominatim.openstreetmap.org/search"

    params = {
        'q': place_name,
        'format': 'json',
        'limit': 1
    }

    headers = {
        'User-Agent': 'DistanceCalculator/1.0'
    }

    response = requests.get(url, params=params, headers=headers)
    data = response.json()

    if len(data) == 0:
        return None


    return {
        'name': data[0]['display_name'],
        'lat': float(data[0]['lat']),
        'lon': float(data[0]['lon'])
    }


def main():

    place = "Paris, France"

    coords = get_coordinates(place)

    if coords is None:
        print(f"Error: Could not find '{place}'")
        return

    print(coords)


if __name__ == "__main__":
    main()