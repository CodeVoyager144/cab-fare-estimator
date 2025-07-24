from flask import Flask, render_template, request
import requests
import datetime
import joblib
import pandas as pd
import os

app = Flask(__name__)

# ------------ API Keys ------------
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
OPENWEATHER_API_KEY = os.environ.get("OPENWEATHER_API_KEY")

# ----------- Helper Functions -----------

def get_route_data(origin, destination, google_api_key):
    url = 'https://maps.googleapis.com/maps/api/directions/json'
    params = {'origin': origin, 'destination': destination, 'key': google_api_key}
    resp = requests.get(url, params=params).json()
    if resp['status'] != 'OK':
        raise Exception(f"Google Maps API error: {resp['status']}")
    leg = resp['routes'][0]['legs'][0]
    route_polyline = resp['routes'][0]['overview_polyline']['points']
    return {
        'pickup_lat': leg['start_location']['lat'],
        'pickup_lon': leg['start_location']['lng'],
        'drop_lat': leg['end_location']['lat'],
        'drop_lon': leg['end_location']['lng'],
        'distance_km': leg['distance']['value'] / 1000,
        'duration_min': leg['duration']['value'] / 60,
        'route_polyline': route_polyline
    }

def get_weather(lat, lon, weather_api_key):
    url = 'https://api.openweathermap.org/data/2.5/weather'
    params = {'lat': lat, 'lon': lon, 'appid': weather_api_key, 'units': 'metric'}
    resp = requests.get(url, params=params).json()
    if resp.get('cod') != 200:
        raise Exception(f"OpenWeather API Error: {resp.get('message')}")
    weather_main = resp['weather'][0]['main']
    temperature = resp['main']['temp']
    return temperature, weather_main

def apply_weather_adjustment(fare, condition):
    if condition in ['Rain', 'Thunderstorm']:
        return fare * 1.25, "+25%"
    elif condition in ['Fog', 'Snow']:
        return fare * 1.15, "+15%"
    elif condition == 'Clouds':
        return fare * 1.05, "+5%"
    else:
        return fare, "+0%"

def generate_map_url(pickup_lat, pickup_lon, drop_lat, drop_lon, api_key, route_polyline):
    base_url = "https://maps.googleapis.com/maps/api/staticmap"
    size = "600x600"
    markers = (
        f"markers=color:green|label:A|{pickup_lat},{pickup_lon}&"
        f"markers=color:red|label:B|{drop_lat},{drop_lon}"
    )
    path = f"path=enc:{route_polyline}"
    url = f"{base_url}?size={size}&{markers}&{path}&key={api_key}"
    return url

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    error = None
    if request.method == "POST":
        origin = request.form.get("origin")
        destination = request.form.get("destination")
        try:
            # Route info
            route_info = get_route_data(origin, destination, GOOGLE_API_KEY)
            pickup_lat = route_info['pickup_lat']
            pickup_lon = route_info['pickup_lon']
            drop_lat = route_info['drop_lat']
            drop_lon = route_info['drop_lon']
            route_polyline = route_info['route_polyline']

            # Weather info
            temperature, weather_condition = get_weather(pickup_lat, pickup_lon, OPENWEATHER_API_KEY)

            # Time features
            now = datetime.datetime.now()
            hour = now.hour
            dayofweek = now.weekday()
            is_weekend = int(dayofweek >= 5)

            # Model input
            input_data = {
                'pickup_lat': pickup_lat,
                'pickup_lon': pickup_lon,
                'drop_lat': drop_lat,
                'drop_lon': drop_lon,
                'distance_km': route_info['distance_km'],
                'duration_min': route_info['duration_min'],
                'hour': hour,
                'dayofweek': dayofweek,
                'is_weekend': is_weekend,
                'temperature': temperature,
                'weather_condition': weather_condition
            }
            X = pd.DataFrame([input_data])

            # Predict base fare
            model = joblib.load("model/cab_fare_model.pkl")
            base_fare = model.predict(X)[0]

            # Adjustment
            adjusted_fare, adjustment_percent = apply_weather_adjustment(base_fare, weather_condition)

            # Map URL
            map_url = generate_map_url(pickup_lat, pickup_lon, drop_lat, drop_lon, GOOGLE_API_KEY, route_polyline)

            # Results for the template
            result = {
                'origin': origin,
                'destination': destination,
                'pickup_lat': pickup_lat,
                'pickup_lon': pickup_lon,
                'drop_lat': drop_lat,
                'drop_lon': drop_lon,
                'distance': f"{route_info['distance_km']:.2f}",
                'duration': f"{route_info['duration_min']:.2f}",
                'temperature': temperature,
                'weather': weather_condition,
                'base_fare': f"{base_fare:.2f}",
                'adjusted_fare': f"{adjusted_fare:.2f}",
                'adjustment_percent': adjustment_percent,
                'map_url': map_url
            }
        except Exception as e:
            error = str(e)
    return render_template("index.html", result=result, error=error)

if __name__ == "__main__":
    app.run(debug=True)
