{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "0b4a5e29-ec1b-4c58-993c-beb943c663fb",
   "metadata": {},
   "outputs": [
    {
     "name": "stdin",
     "output_type": "stream",
     "text": [
      "Enter origin location (e.g., Delhi):  Saket, Delhi\n",
      "Enter destination location (e.g., Noida):  Noida, Uttar Pradesh\n"
     ]
    }
   ],
   "source": [
    "origin = input(\"Enter origin location (e.g., Delhi): \")\n",
    "destination = input(\"Enter destination location (e.g., Noida): \")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "b3afcabf-023b-4615-9dc3-339e109371d4",
   "metadata": {},
   "outputs": [],
   "source": [
    "import requests\n",
    "\n",
    "def get_route_data(origin, destination, google_api_key):\n",
    "    url = 'https://maps.googleapis.com/maps/api/directions/json'\n",
    "    params = {\n",
    "        'origin': origin,\n",
    "        'destination': destination,\n",
    "        'key': google_api_key\n",
    "    }\n",
    "    response = requests.get(url, params=params).json()\n",
    "\n",
    "    if response['status'] != 'OK':\n",
    "        raise Exception(f\"Google Maps API error: {response['status']}\")\n",
    "\n",
    "    leg = response['routes'][0]['legs'][0]\n",
    "\n",
    "    return {\n",
    "        'pickup_lat': leg['start_location']['lat'],\n",
    "        'pickup_lon': leg['start_location']['lng'],\n",
    "        'drop_lat': leg['end_location']['lat'],\n",
    "        'drop_lon': leg['end_location']['lng'],\n",
    "        'distance_km': leg['distance']['value'] / 1000,\n",
    "        'duration_min': leg['duration']['value'] / 60\n",
    "    }\n",
    "\n",
    "# Example usage\n",
    "GOOGLE_API_KEY = 'AIzaSyAkhXYsZUEAA4lIne2r_UsQeMWNTXBhyZs'\n",
    "route_info = get_route_data(origin, destination, GOOGLE_API_KEY)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "993548a2-7627-4fba-bc07-0fd897d18139",
   "metadata": {},
   "outputs": [],
   "source": [
    "def get_weather(lat, lon, weather_api_key):\n",
    "    url = 'https://api.openweathermap.org/data/2.5/weather'\n",
    "    params = {\n",
    "        'lat': lat,\n",
    "        'lon': lon,\n",
    "        'appid': weather_api_key,\n",
    "        'units': 'metric'\n",
    "    }\n",
    "\n",
    "    response = requests.get(url, params=params).json()\n",
    "\n",
    "    if response.get('cod') != 200:\n",
    "        raise Exception(f\"OpenWeather API Error: {response.get('message')}\")\n",
    "\n",
    "    weather_main = response['weather'][0]['main']  # e.g., 'Clouds'\n",
    "    temperature = response['main']['temp']         # e.g., 30.5\n",
    "\n",
    "    return temperature, weather_main\n",
    "\n",
    "# Example usage\n",
    "OPENWEATHER_API_KEY = '07b2c1eb15dc9b8f5b77bef2febe3e87'\n",
    "temperature, weather_condition = get_weather(route_info['pickup_lat'], route_info['pickup_lon'], OPENWEATHER_API_KEY)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "095d73c5-a32b-4413-b240-ef4177d8a97b",
   "metadata": {},
   "outputs": [],
   "source": [
    "import datetime\n",
    "\n",
    "now = datetime.datetime.now()\n",
    "hour = now.hour\n",
    "dayofweek = now.weekday()\n",
    "is_weekend = 1 if dayofweek >= 5 else 0\n",
    "\n",
    "# Prepare feature vector for ML model\n",
    "features = [\n",
    "    route_info['pickup_lat'],\n",
    "    route_info['pickup_lon'],\n",
    "    route_info['drop_lat'],\n",
    "    route_info['drop_lon'],\n",
    "    route_info['distance_km'],\n",
    "    route_info['duration_min'],\n",
    "    hour,\n",
    "    dayofweek,\n",
    "    is_weekend,\n",
    "    temperature,\n",
    "    weather_condition  # Categorical, handled by model preprocessor\n",
    "]\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "id": "d27983b6-ffe9-499d-89fa-53666694c413",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "🎯 Base Fare: ₹402.48\n"
     ]
    }
   ],
   "source": [
    "import pandas as pd\n",
    "import joblib\n",
    "\n",
    "# Load the pipeline\n",
    "model = joblib.load(\"cab_fare_model.pkl\")\n",
    "\n",
    "# Create input as dictionary matching training feature names\n",
    "input_data = {\n",
    "    'pickup_lat': route_info['pickup_lat'],\n",
    "    'pickup_lon': route_info['pickup_lon'],\n",
    "    'drop_lat': route_info['drop_lat'],\n",
    "    'drop_lon': route_info['drop_lon'],\n",
    "    'distance_km': route_info['distance_km'],\n",
    "    'duration_min': route_info['duration_min'],\n",
    "    'hour': hour,\n",
    "    'dayofweek': dayofweek,\n",
    "    'is_weekend': is_weekend,\n",
    "    'temperature': temperature,\n",
    "    'weather_condition': weather_condition  # String is fine, let pipeline encode\n",
    "}\n",
    "\n",
    "# Convert to DataFrame with single row\n",
    "X = pd.DataFrame([input_data])\n",
    "\n",
    "# Predict fare using pipeline\n",
    "base_fare = model.predict(X)[0]\n",
    "\n",
    "print(f\"🎯 Base Fare: ₹{base_fare:.2f}\")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "id": "83429328-3fce-4adc-9b95-b9e068fa3f41",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Weather: Clouds\n",
      "Adjusted Fare: ₹422.60\n"
     ]
    }
   ],
   "source": [
    "def apply_weather_adjustment(fare, condition):\n",
    "    if condition in ['Rain', 'Thunderstorm']:\n",
    "        return fare * 1.25\n",
    "    elif condition in ['Fog', 'Snow']:  # Add 'Haze' here if you define it\n",
    "        return fare * 1.15\n",
    "    elif condition == 'Clouds':\n",
    "        return fare * 1.05\n",
    "    else:  # 'Clear', or anything else\n",
    "        return fare\n",
    "\n",
    "final_fare = apply_weather_adjustment(base_fare, weather_condition)\n",
    "print(f\"Weather: {weather_condition}\")\n",
    "print(f\"Adjusted Fare: ₹{final_fare:.2f}\")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "id": "f1a5c92b-5e0e-49ea-a2d4-dba8bce688c1",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "====================================\n",
      "From: Saket, Delhi  →  To: Noida, Uttar Pradesh\n",
      "Distance: 27.35 km\n",
      "Duration: 46.60 min\n",
      "Temp: 30.87°C | Weather: Clouds\n",
      "Base Fare: ₹402.48\n",
      "Final Fare (With Adjustment): ₹422.60\n",
      "====================================\n"
     ]
    }
   ],
   "source": [
    "print(\"====================================\")\n",
    "print(f\"From: {origin}  →  To: {destination}\")\n",
    "print(f\"Distance: {route_info['distance_km']:.2f} km\")\n",
    "print(f\"Duration: {route_info['duration_min']:.2f} min\")\n",
    "print(f\"Temp: {temperature}°C | Weather: {weather_condition}\")\n",
    "print(f\"Base Fare: ₹{base_fare:.2f}\")\n",
    "print(f\"Final Fare (With Adjustment): ₹{final_fare:.2f}\")\n",
    "print(\"====================================\")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "44a5091c-d551-47ca-a6b0-eb9e2e74c2e3",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.2"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
