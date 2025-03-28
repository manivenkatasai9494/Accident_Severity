from flask import Flask, render_template, request, jsonify
import joblib
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
import json
import requests
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut
import os
from dotenv import load_dotenv

app = Flask(__name__)

# OpenWeatherMap API Configuration
OPENWEATHER_API_KEY = "c8dda160bf9944492d159d19f2a0c82a"  # Replace with your actual API key
OPENWEATHER_BASE_URL = "http://api.openweathermap.org/data/2.5/weather"

# Add Google Places API configuration
GOOGLE_PLACES_API_KEY = "AIzaSyC9YdHb4Eo17MpYoJQMeORsDSSmwXpDJZ4"  # Replace with your actual API key
GOOGLE_PLACES_BASE_URL = "https://maps.googleapis.com/maps/api/place"

def get_coordinates_from_location(location):
    """Get coordinates from location name using geopy"""
    try:
        geolocator = Nominatim(user_agent="accident_prediction")
        location_data = geolocator.geocode(location)
        if location_data:
            return location_data.latitude, location_data.longitude
        return None, None
    except GeocoderTimedOut:
        return None, None

def get_weather_data(lat, lon):
    """Fetch weather data from OpenWeatherMap API"""
    try:
        params = {
            'lat': lat,
            'lon': lon,
            'appid': OPENWEATHER_API_KEY,
            'units': 'imperial'  # Use Fahrenheit for temperature
        }
        response = requests.get(OPENWEATHER_BASE_URL, params=params)
        data = response.json()
        
        if response.status_code == 200:
            # Get more detailed weather data
            weather_data = {
                'temperature': data['main']['temp'],
                'humidity': data['main']['humidity'],
                'wind_speed': data['wind']['speed'],
                'visibility': data['visibility'] / 1609.34,  # Convert meters to miles
                'precipitation': data.get('rain', {}).get('1h', 0) / 25.4,  # Convert mm to inches
                'pressure': data['main']['pressure'],
                'feels_like': data['main']['feels_like'],
                'weather_main': data['weather'][0]['main'],
                'weather_description': data['weather'][0]['description']
            }
            
            # Add additional weather conditions
            if 'snow' in data:
                weather_data['snow'] = data['snow'].get('1h', 0) / 25.4
            else:
                weather_data['snow'] = 0
                
            return weather_data
        return None
    except Exception as e:
        print(f"Error fetching weather data: {str(e)}")
        return None

# Load the model and scaler
try:
    # Try loading the joblib model first
    model = joblib.load('rf_model.joblib')
    print("Model loaded successfully from joblib")
    print(f"Model type: {type(model)}")
    print(f"Model attributes: {dir(model)}")
except Exception as e:
    print(f"Error loading joblib model: {str(e)}")
    try:
        # Try loading the pickle model as fallback
        with open('rf_model.pkl', 'rb') as f:
            model = pickle.load(f)
        print("Model loaded successfully from pickle")
        print(f"Model type: {type(model)}")
        print(f"Model attributes: {dir(model)}")
    except Exception as e:
        print(f"Error loading pickle model: {str(e)}")
        model = None

try:
    # Try loading the scaler
    scaler = joblib.load('scaler.joblib')
    print("Scaler loaded successfully")
except Exception as e:
    print(f"Error loading scaler: {str(e)}")
    scaler = None

# Store recent predictions (in-memory storage for demo)
recent_predictions = []

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'GET':
        return render_template('predict.html')
        
    if not model:
        return jsonify({'error': 'Model not loaded'}), 500
        
    try:
        data = request.get_json()
        location = data.get('location', '')
        
        # Get coordinates from location name
        lat, lon = get_coordinates_from_location(location)
        if not lat or not lon:
            return jsonify({'error': 'Could not find coordinates for the given location'}), 400
            
        # Get weather data from OpenWeatherMap
        weather_data = get_weather_data(lat, lon)
        if not weather_data:
            return jsonify({'error': 'Could not fetch weather data for the location'}), 400
        
        # Extract features from input and weather data
        features = [
            float(data.get('distance', 0)),
            weather_data['temperature'],
            weather_data['humidity'],
            weather_data['visibility'],
            weather_data['wind_speed'],
            weather_data['precipitation'],
            int(data.get('traffic_signal', False)),
            int(data.get('junction', False)),
            int(data.get('railway', False)),
            int(data.get('nautical_twilight', False)),
            lat,
            lon
        ]
        
        # Scale features
        features_scaled = scaler.transform([features])
        
        # Make prediction
        prediction = model.predict(features_scaled)[0]
        probabilities = model.predict_proba(features_scaled)[0]
        
        # Calculate base confidence
        base_confidence = max(probabilities) * 100
        
        # Adjust confidence based on conditions
        confidence_factors = []
        
        # Weather-based confidence adjustments
        if weather_data['visibility'] > 5:
            confidence_factors.append(1.1)  # 10% increase for excellent visibility
        elif weather_data['visibility'] > 3:
            confidence_factors.append(1.05)  # 5% increase for good visibility
            
        if 20 < weather_data['wind_speed'] < 30:
            confidence_factors.append(1.1)  # 10% increase for moderate wind
        elif weather_data['wind_speed'] < 20:
            confidence_factors.append(1.15)  # 15% increase for low wind
            
        if weather_data['precipitation'] == 0:
            confidence_factors.append(1.1)  # 10% increase for no precipitation
            
        if 32 < weather_data['temperature'] < 90:
            confidence_factors.append(1.05)  # 5% increase for moderate temperature
            
        # Traffic condition-based confidence adjustments
        if data.get('traffic_signal', False):
            confidence_factors.append(1.05)  # 5% increase for traffic signal
            
        if not data.get('junction', False):
            confidence_factors.append(1.05)  # 5% increase for no junction
            
        if not data.get('railway', False):
            confidence_factors.append(1.05)  # 5% increase for no railway
            
        if not data.get('nautical_twilight', False):
            confidence_factors.append(1.1)  # 10% increase for daylight conditions
            
        # Distance-based confidence adjustment
        if float(data.get('distance', 0)) < 1:
            confidence_factors.append(1.1)  # 10% increase for short distance
            
        # Calculate final confidence
        final_confidence = base_confidence
        for factor in confidence_factors:
            final_confidence *= factor
            
        # Cap confidence at 95%
        final_confidence = min(final_confidence, 95)
        
        # Map numeric severity to categories
        severity_mapping = {
            1: "1 - Very Low",
            2: "2 - Low",
            3: "3 - Moderate",
            4: "4 - High",
            5: "5 - Very High"
        }
        severity = severity_mapping.get(prediction, "Unknown")
        
        # Determine risk factors with more precise thresholds
        risk_factors = []
        
        # Weather-related risk factors with confidence levels
        if weather_data['visibility'] < 0.5:
            risk_factors.append("Extremely low visibility conditions (High Risk)")
        elif weather_data['visibility'] < 1:
            risk_factors.append("Low visibility conditions (Moderate Risk)")
            
        if weather_data['wind_speed'] > 30:
            risk_factors.append("Strong wind conditions (High Risk)")
        elif weather_data['wind_speed'] > 20:
            risk_factors.append("High wind speed (Moderate Risk)")
            
        if weather_data['precipitation'] > 0.5:
            risk_factors.append("Heavy precipitation (High Risk)")
        elif weather_data['precipitation'] > 0:
            risk_factors.append("Light precipitation (Low Risk)")
            
        if weather_data['snow'] > 0:
            risk_factors.append("Snow conditions (High Risk)")
            
        if weather_data['temperature'] < 32:
            risk_factors.append("Freezing conditions (High Risk)")
        elif weather_data['temperature'] > 90:
            risk_factors.append("Extreme heat conditions (High Risk)")
            
        # Traffic-related risk factors with confidence levels
        if data.get('traffic_signal', False):
            risk_factors.append("Traffic signal present (Low Risk)")
        if data.get('junction', False):
            risk_factors.append("Junction area (Moderate Risk)")
        if data.get('railway', False):
            risk_factors.append("Railway crossing (High Risk)")
        if data.get('nautical_twilight', False):
            risk_factors.append("Low light conditions (Moderate Risk)")
            
        # Additional risk factors based on weather conditions
        if weather_data['weather_main'] in ['Thunderstorm', 'Tornado']:
            risk_factors.append("Severe weather conditions (High Risk)")
        elif weather_data['weather_main'] in ['Rain', 'Drizzle']:
            risk_factors.append("Wet road conditions (Moderate Risk)")
        elif weather_data['weather_main'] in ['Snow', 'Sleet']:
            risk_factors.append("Slippery road conditions (High Risk)")
            
        # Generate more detailed recommendations based on severity and conditions
        recommendations = []
        if severity in ["4 - High", "5 - Very High"]:
            recommendations.extend([
                "Immediate emergency response required",
                "Maximum resources needed",
                "Alert nearby hospitals",
                "Prepare for potential road closure",
                "Deploy additional emergency vehicles",
                "Coordinate with local law enforcement"
            ])
        elif severity == "3 - Moderate":
            recommendations.extend([
                "Emergency response teams on standby",
                "Alert nearby hospitals",
                "Monitor traffic conditions",
                "Prepare for potential delays",
                "Increase police presence",
                "Set up traffic control points"
            ])
        else:
            recommendations.extend([
                "Basic monitoring required",
                "Standard traffic control measures",
                "Regular status updates",
                "Prepare for minor delays",
                "Maintain normal emergency response",
                "Monitor weather conditions"
            ])
            
        # Add weather-specific recommendations
        if weather_data['visibility'] < 1:
            recommendations.append("Deploy additional lighting and warning signs")
        if weather_data['wind_speed'] > 20:
            recommendations.append("Monitor for fallen debris and tree branches")
        if weather_data['precipitation'] > 0:
            recommendations.append("Deploy road maintenance crews")
        
        # Store prediction for dashboard
        prediction_data = {
            'timestamp': datetime.now(),
            'location': {
                'lat': lat,
                'lon': lon,
                'name': location
            },
            'severity': severity,
            'confidence': round(final_confidence, 2),
            'risk_factors': risk_factors,
            'recommendations': recommendations,
            'weather': weather_data,
            'traffic': {
                'distance': float(data.get('distance', 0)),
                'traffic_signal': bool(data.get('traffic_signal', False)),
                'junction': bool(data.get('junction', False)),
                'railway': bool(data.get('railway', False)),
                'nautical_twilight': bool(data.get('nautical_twilight', False))
            }
        }
        
        recent_predictions.append(prediction_data)
        if len(recent_predictions) > 100:  # Keep only last 100 predictions
            recent_predictions.pop(0)
            
        return jsonify({
            'severity': severity,
            'confidence': round(final_confidence, 2),
            'risk_factors': risk_factors,
            'recommendations': recommendations,
            'weather_data': weather_data,
            'features': features
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/dashboard')
def dashboard():
    # Get the latest predictions
    predictions = recent_predictions[-100:] if recent_predictions else []
    
    # Calculate statistics
    total_accidents = len(predictions)
    severity_counts = {'1 - Very Low': 0, '2 - Low': 0, '3 - Moderate': 0, '4 - High': 0, '5 - Very High': 0}
    confidences = []
    weather_impact = {'visibility': [], 'wind_speed': []}
    time_analysis = [0] * 24
    hotspots = {}
    
    for pred in predictions:
        severity_counts[pred['severity']] += 1
        confidences.append(pred['confidence'])
        weather_impact['visibility'].append(pred['weather']['visibility'])
        weather_impact['wind_speed'].append(pred['weather']['wind_speed'])
        time_analysis[pred['timestamp'].hour] += 1
        
        # Aggregate hotspots
        location_key = f"{pred['location']['lat']},{pred['location']['lon']}"
        if location_key not in hotspots:
            hotspots[location_key] = {
                'lat': pred['location']['lat'],
                'lon': pred['location']['lon'],
                'count': 0,
                'max_severity': '1 - Very Low'
            }
        hotspots[location_key]['count'] += 1
        if pred['severity'] in ['4 - High', '5 - Very High']:
            hotspots[location_key]['max_severity'] = pred['severity']
    
    # Convert hotspots to list and sort by count
    hotspots_list = list(hotspots.values())
    hotspots_list.sort(key=lambda x: x['count'], reverse=True)
    
    # Calculate average confidence
    avg_confidence = round(sum(confidences) / len(confidences), 2) if confidences else 0
    
    # Calculate weather impact averages
    avg_visibility = round(sum(weather_impact['visibility']) / len(weather_impact['visibility']), 2) if weather_impact['visibility'] else 0
    avg_wind_speed = round(sum(weather_impact['wind_speed']) / len(weather_impact['wind_speed']), 2) if weather_impact['wind_speed'] else 0
    
    return render_template('dashboard.html',
                         total_accidents=total_accidents,
                         severity_counts=severity_counts,
                         avg_confidence=avg_confidence,
                         avg_visibility=avg_visibility,
                         avg_wind_speed=avg_wind_speed,
                         time_analysis=time_analysis,
                         hotspots=hotspots_list[:10],
                         predictions=predictions)

@app.route('/api/dashboard/stats')
def dashboard_stats():
    if not recent_predictions:
        return jsonify({
            'total_accidents': 0,
            'severity_distribution': {'1 - Very Low': 0, '2 - Low': 0, '3 - Moderate': 0, '4 - High': 0, '5 - Very High': 0},
            'avg_confidence': 0,
            'weather_impact': {'visibility': 0, 'wind_speed': 0},
            'time_analysis': [0] * 24,
            'hotspots': []
        })

    # Calculate statistics
    total_accidents = len(recent_predictions)
    severity_counts = {'1 - Very Low': 0, '2 - Low': 0, '3 - Moderate': 0, '4 - High': 0, '5 - Very High': 0}
    confidences = []
    weather_impact = {'visibility': [], 'wind_speed': []}
    time_analysis = [0] * 24
    hotspots = {}

    for pred in recent_predictions:
        severity_counts[pred['severity']] += 1
        confidences.append(pred['confidence'])
        weather_impact['visibility'].append(pred['weather']['visibility'])
        weather_impact['wind_speed'].append(pred['weather']['wind_speed'])
        time_analysis[pred['timestamp'].hour] += 1

        # Aggregate hotspots
        location_key = f"{pred['location']['lat']},{pred['location']['lon']}"
        if location_key not in hotspots:
            hotspots[location_key] = {
                'lat': pred['location']['lat'],
                'lon': pred['location']['lon'],
                'count': 0,
                'max_severity': '1 - Very Low'
            }
        hotspots[location_key]['count'] += 1
        if pred['severity'] in ['4 - High', '5 - Very High']:
            hotspots[location_key]['max_severity'] = pred['severity']

    # Convert hotspots to list and sort by count
    hotspots_list = list(hotspots.values())
    hotspots_list.sort(key=lambda x: x['count'], reverse=True)

    return jsonify({
        'total_accidents': total_accidents,
        'severity_distribution': severity_counts,
        'avg_confidence': round(sum(confidences) / len(confidences), 2),
        'weather_impact': {
            'visibility': round(sum(weather_impact['visibility']) / len(weather_impact['visibility']), 2),
            'wind_speed': round(sum(weather_impact['wind_speed']) / len(weather_impact['wind_speed']), 2)
        },
        'time_analysis': time_analysis,
        'hotspots': hotspots_list[:10]  # Return top 10 hotspots
    })

@app.route('/emergency')
def emergency():
    """Emergency services dashboard"""
    # Get recent high severity accidents
    recent_emergencies = []
    for pred in recent_predictions:
        if pred['severity'] in ['4 - High', '5 - Very High']:
            recent_emergencies.append({
                'location': pred['location'],
                'severity': pred['severity'],
                'confidence': pred['confidence'],
                'risk_factors': pred['risk_factors'],
                'timestamp': pred['timestamp']
            })
    
    # Sort by severity and timestamp
    recent_emergencies.sort(key=lambda x: (x['severity'], x['timestamp']), reverse=True)
    
    return render_template('emergency.html', emergencies=recent_emergencies)

def get_nearby_hospitals_google(lat, lon, radius=30000):  # 30km radius in meters
    """Fetch nearby hospitals using Google Places API"""
    try:
        # First, search for hospitals
        search_url = f"{GOOGLE_PLACES_BASE_URL}/nearbysearch/json"
        params = {
            'location': f"{lat},{lon}",
            'radius': radius,
            'type': 'hospital',
            'key': GOOGLE_PLACES_API_KEY
        }
        
        response = requests.get(search_url, params=params)
        data = response.json()
        
        if data['status'] != 'OK':
            return []
            
        hospitals = []
        for place in data['results'][:10]:  # Get top 10 hospitals
            # Get detailed information for each hospital
            details_url = f"{GOOGLE_PLACES_BASE_URL}/details/json"
            details_params = {
                'place_id': place['place_id'],
                'fields': 'name,formatted_address,formatted_phone_number,website,opening_hours,rating,user_ratings_total',
                'key': GOOGLE_PLACES_API_KEY
            }
            
            details_response = requests.get(details_url, params=details_params)
            details_data = details_response.json()
            
            if details_data['status'] == 'OK':
                hospital = {
                    'name': place['name'],
                    'lat': place['geometry']['location']['lat'],
                    'lon': place['geometry']['location']['lng'],
                    'address': place['vicinity'],
                    'phone': details_data['result'].get('formatted_phone_number', 'N/A'),
                    'website': details_data['result'].get('website', 'N/A'),
                    'rating': place.get('rating', 'N/A'),
                    'reviews': place.get('user_ratings_total', 0),
                    'opening_hours': details_data['result'].get('opening_hours', {}).get('weekday_text', ['N/A']),
                    'distance': calculate_distance(lat, lon, 
                                               place['geometry']['location']['lat'],
                                               place['geometry']['location']['lng'])
                }
                hospitals.append(hospital)
        
        # Sort hospitals by distance
        hospitals.sort(key=lambda x: x['distance'])
        return hospitals
        
    except Exception as e:
        print(f"Error fetching hospitals from Google Places API: {str(e)}")
        return []

@app.route('/get_nearby_hospitals', methods=['POST'])
def get_nearby_hospitals():
    try:
        data = request.get_json()
        lat = float(data.get('lat'))
        lon = float(data.get('lon'))
        radius = float(data.get('radius', 30))  # Default 30km radius
        
        # Use Google Places API to get hospitals
        hospitals = get_nearby_hospitals_google(lat, lon, radius * 1000)  # Convert km to meters
        
        return jsonify({
            'success': True,
            'hospitals': hospitals,
            'radius': radius
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

def calculate_distance(lat1, lon1, lat2, lon2):
    """Calculate distance between two points in kilometers using Haversine formula"""
    R = 6371  # Earth's radius in kilometers
    
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    
    return R * c

@app.route('/get_coordinates', methods=['POST'])
def get_coordinates():
    try:
        data = request.get_json()
        location = data.get('location', '')
        
        if not location:
            return jsonify({'error': 'Location is required'}), 400
            
        # Get coordinates from location name
        lat, lon = get_coordinates_from_location(location)
        if not lat or not lon:
            return jsonify({'error': 'Could not find coordinates for the given location'}), 400
            
        return jsonify({
            'success': True,
            'lat': lat,
            'lon': lon
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

if __name__ == '__main__':
    app.run(debug=True)