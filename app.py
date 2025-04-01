from flask import Flask, render_template, request, jsonify, Response, session, redirect, url_for
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
import google.generativeai as genai
import uuid
from flask_sqlalchemy import SQLAlchemy
from functools import wraps

app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY', 'your-secret-key-here')

# OpenWeatherMap API Configuration
OPENWEATHER_API_KEY = "c8dda160bf9944492d159d19f2a0c82a"  # Replace with your actual API key
OPENWEATHER_BASE_URL = "http://api.openweathermap.org/data/2.5/weather"

# Add Google Places API configuration
GOOGLE_PLACES_API_KEY = "AIzaSyC9YdHb4Eo17MpYoJQMeORsDSSmwXpDJZ4"  # Replace with your actual API key
GOOGLE_PLACES_BASE_URL = "https://maps.googleapis.com/maps/api/place"

# Load environment variables
load_dotenv()

# Configure Gemini Pro
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
genai.configure(api_key=GOOGLE_API_KEY)
gemini_model = genai.GenerativeModel('gemini-pro')

# Chat history storage (in-memory for demo)
chat_history = {}

# In-memory storage for hospital requests (replace with database in production)
hospital_requests = {}

# Admin credentials (replace with proper authentication in production)
ADMIN_USERNAME = "admin"
ADMIN_PASSWORD = "admin123"

# Database configuration
app.config['SQLALCHEMY_DATABASE_URI'] = f"mysql+mysqlconnector://{os.getenv('MYSQL_USER')}:{os.getenv('MYSQL_PASSWORD')}@{os.getenv('MYSQL_HOST')}/{os.getenv('MYSQL_DB')}"
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# Login required decorator
def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not session.get('is_admin'):
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

def is_admin(username, password):
    return username == ADMIN_USERNAME and password == ADMIN_PASSWORD

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
def index():
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
        
        # Store prediction in database
        new_accident = Accident(
            location=location,
            severity=prediction,  # Store numeric severity
            features={
                'lat': lat,
                'lon': lon,
                'weather': weather_data,
                'traffic': {
                    'distance': float(data.get('distance', 0)),
                    'traffic_signal': bool(data.get('traffic_signal', False)),
                    'junction': bool(data.get('junction', False)),
                    'railway': bool(data.get('railway', False)),
                    'nautical_twilight': bool(data.get('nautical_twilight', False))
                },
                'risk_factors': risk_factors,
                'recommendations': recommendations,
                'confidence': round(final_confidence, 2)
            }
        )
        
        db.session.add(new_accident)
        db.session.commit()
        
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

def get_nearby_hospitals_google(lat, lon, radius):
    """Get nearby hospitals using Google Places API"""
    try:
        # First, try to get hospitals
        hospitals_url = f"{GOOGLE_PLACES_BASE_URL}/nearbysearch/json"
        params = {
            'location': f"{lat},{lon}",
            'radius': radius,
            'type': 'hospital',
            'key': GOOGLE_PLACES_API_KEY
        }
        
        print(f"Making Google Places API request with params: {params}")  # Debug print
        
        response = requests.get(hospitals_url, params=params)
        print(f"Google Places API response status: {response.status_code}")  # Debug print
        print(f"Google Places API response: {response.text}")  # Debug print
        
        if response.status_code != 200:
            print(f"Error response from Google Places API: {response.text}")  # Debug print
            return []
            
        data = response.json()
        
        if 'error_message' in data:
            print(f"Google Places API error: {data['error_message']}")  # Debug print
            return []
            
        if 'results' not in data:
            print("No results found in Google Places API response")  # Debug print
            return []
            
        hospitals = []
        for place in data['results']:
            try:
                # Get detailed information for each hospital
                place_id = place['place_id']
                details_url = f"{GOOGLE_PLACES_BASE_URL}/details/json"
                details_params = {
                    'place_id': place_id,
                    'fields': 'name,formatted_address,formatted_phone_number,website,opening_hours,rating,reviews,geometry',
                    'key': GOOGLE_PLACES_API_KEY
                }
                
                details_response = requests.get(details_url, params=details_params)
                if details_response.status_code == 200:
                    details_data = details_response.json()
                    if details_data.get('result'):
                        details = details_data['result']
                        
                        # Calculate distance
                        distance = calculate_distance(lat, lon, place['geometry']['location']['lat'], place['geometry']['location']['lng'])
                        
                        hospital = {
                            'id': place_id,
                            'name': details.get('name', 'Unknown Hospital'),
                            'address': details.get('formatted_address', 'Address not available'),
                            'phone': details.get('formatted_phone_number', 'Phone not available'),
                            'website': details.get('website', '#'),
                            'rating': details.get('rating', 'N/A'),
                            'reviews': details.get('reviews', []),
                            'opening_hours': details.get('opening_hours', {}).get('weekday_text', []),
                            'distance': distance,
                            'lat': place['geometry']['location']['lat'],
                            'lon': place['geometry']['location']['lng']
                        }
                        hospitals.append(hospital)
                        print(f"Added hospital: {hospital['name']} at distance {distance:.2f}km")  # Debug print
            except Exception as e:
                print(f"Error processing hospital details: {str(e)}")  # Debug print
                continue
        
        # Sort hospitals by distance
        hospitals.sort(key=lambda x: x['distance'])
        print(f"Total hospitals found: {len(hospitals)}")  # Debug print
        return hospitals
        
    except Exception as e:
        print(f"Error in get_nearby_hospitals_google: {str(e)}")  # Debug print
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

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        message = data.get('message', '')
        
        if not message:
            return jsonify({'error': 'Message is required'}), 400
            
        # Create a prompt for emergency advice
        prompt = f"""You are an emergency response assistant. A user has reported an emergency situation: "{message}"

Please provide immediate advice and steps to take. Include:
1. First aid measures if applicable
2. Emergency services to contact
3. Important safety precautions
4. What NOT to do
5. Follow-up actions

Keep the response clear, concise, and focused on immediate actions."""

        # Generate response using Gemini Pro
        response = gemini_model.generate_content(prompt)
        
        # Store in chat history
        if 'messages' not in chat_history:
            chat_history['messages'] = []
        chat_history['messages'].append({
            'role': 'user',
            'content': message
        })
        chat_history['messages'].append({
            'role': 'assistant',
            'content': response.text
        })
        
        return jsonify({
            'response': response.text,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        })
        
    except Exception as e:
        print(f"Chat Error: {str(e)}")
        return jsonify({'error': 'Failed to generate response'}), 500

@app.route('/chat_history', methods=['GET'])
def get_chat_history():
    return jsonify(chat_history.get('messages', []))

@app.route('/chat_widget')
def chat_widget():
    return render_template('chat_widget.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        if username == ADMIN_USERNAME and password == ADMIN_PASSWORD:
            session['is_admin'] = True
            return redirect(url_for('admin_dashboard'))
        else:
            return render_template('login.html', error='Invalid credentials')
    
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('is_admin', None)
    return redirect(url_for('index'))

@app.route('/admin')
@admin_required
def admin_dashboard():
    # Get statistics
    total_accidents = db.session.query(Accident).count()
    total_requests = db.session.query(HospitalRequest).count()
    pending_requests = db.session.query(HospitalRequest).filter_by(status='pending').count()
    
    # Get recent accidents
    recent_accidents = db.session.query(Accident).order_by(Accident.timestamp.desc()).limit(5).all()
    
    # Get recent requests
    recent_requests = db.session.query(HospitalRequest).order_by(HospitalRequest.timestamp.desc()).limit(5).all()
    
    return render_template('admin_dashboard.html',
                         total_accidents=total_accidents,
                         total_requests=total_requests,
                         pending_requests=pending_requests,
                         recent_accidents=recent_accidents,
                         recent_requests=recent_requests)

@app.route('/api/admin/requests')
@admin_required
def get_requests():
    requests = HospitalRequest.query.order_by(HospitalRequest.timestamp.desc()).all()
    return jsonify({
        'success': True,
        'requests': [{
            'id': req.id,
            'hospital_id': req.hospital_id,
            'hospital_name': req.hospital_name,
            'patient_name': req.patient_name,
            'emergency_type': req.emergency_type,
            'description': req.description,
            'contact_number': req.contact_number,
            'status': req.status,
            'timestamp': req.timestamp.strftime('%Y-%m-%d %H:%M:%S')
        } for req in requests]
    })

@app.route('/api/admin/requests/<request_id>/approve', methods=['POST'])
@admin_required
def approve_request(request_id):
    request = HospitalRequest.query.get_or_404(request_id)
    request.status = 'approved'
    db.session.commit()
    return jsonify({'success': True})

@app.route('/api/admin/requests/<request_id>/reject', methods=['POST'])
@admin_required
def reject_request(request_id):
    request = HospitalRequest.query.get_or_404(request_id)
    request.status = 'rejected'
    db.session.commit()
    return jsonify({'success': True})

@app.route('/api/emergency/request', methods=['POST'])
def submit_hospital_request():
    try:
        data = request.get_json()
        print("Received request data:", data)  # Debug print
        
        # Get hospital name from Google Places API if not provided
        if 'hospital_name' not in data:
            # Use Google Places API to get hospital details
            place_id = data['hospital_id']
            details_url = f"{GOOGLE_PLACES_BASE_URL}/details/json"
            details_params = {
                'place_id': place_id,
                'fields': 'name',
                'key': GOOGLE_PLACES_API_KEY
            }
            
            details_response = requests.get(details_url, params=details_params)
            if details_response.status_code == 200:
                details_data = details_response.json()
                if details_data.get('result'):
                    data['hospital_name'] = details_data['result'].get('name', 'Unknown Hospital')
                else:
                    data['hospital_name'] = 'Unknown Hospital'
            else:
                data['hospital_name'] = 'Unknown Hospital'
        
        # Create new hospital request
        new_request = HospitalRequest(
            hospital_id=data['hospital_id'],
            hospital_name=data['hospital_name'],
            patient_name=data['patient_name'],
            emergency_type=data['emergency_type'],
            description=data['description'],
            contact_number=data['contact_number'],
            status='pending'
        )
        
        # Save to database
        db.session.add(new_request)
        db.session.commit()
        
        print("Request saved successfully")  # Debug print
        return jsonify({
            'success': True,
            'request_id': new_request.id,
            'message': 'Request submitted successfully'
        })
        
    except Exception as e:
        print("Error in submit_hospital_request:", str(e))  # Debug print
        db.session.rollback()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/admin/stats')
@admin_required
def get_admin_stats():
    try:
        total_accidents = db.session.query(Accident).count()
        total_requests = db.session.query(HospitalRequest).count()
        pending_requests = db.session.query(HospitalRequest).filter_by(status='pending').count()
        
        return jsonify({
            'success': True,
            'total_accidents': total_accidents,
            'total_requests': total_requests,
            'pending_requests': pending_requests
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/admin/accidents')
@admin_required
def get_admin_accidents():
    try:
        accidents = db.session.query(Accident).order_by(Accident.timestamp.desc()).limit(10).all()
        return jsonify({
            'success': True,
            'accidents': [{
                'id': accident.id,
                'location': accident.location,
                'severity': accident.severity,
                'timestamp': accident.timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                'features': accident.features
            } for accident in accidents]
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/admin/accidents/<int:accident_id>')
@admin_required
def get_accident_details(accident_id):
    try:
        accident = Accident.query.get_or_404(accident_id)
        return jsonify({
            'success': True,
            'accident': {
                'id': accident.id,
                'location': accident.location,
                'severity': accident.severity,
                'timestamp': accident.timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                'features': accident.features
            }
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/emergency/request/<int:request_id>/status')
def get_request_status(request_id):
    try:
        request = HospitalRequest.query.get_or_404(request_id)
        return jsonify({
            'success': True,
            'status': request.status,
            'timestamp': request.timestamp.strftime('%Y-%m-%d %H:%M:%S')
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/emergency/requests/status', methods=['GET'])
def get_all_request_statuses():
    try:
        # Get all requests for the current user's session
        requests = HospitalRequest.query.filter_by(session_id=session.get('session_id')).all()
        
        # Create a dictionary of hospital_id -> request status
        request_statuses = {}
        for req in requests:
            request_statuses[req.hospital_id] = {
                'status': req.status,
                'timestamp': req.updated_at.strftime('%Y-%m-%d %H:%M:%S')
            }
        
        return jsonify({
            'success': True,
            'requests': request_statuses
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# Database Models
class Accident(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    location = db.Column(db.String(255), nullable=False)
    severity = db.Column(db.Integer, nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    features = db.Column(db.JSON)

class HospitalRequest(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    hospital_id = db.Column(db.String(255), nullable=False)
    hospital_name = db.Column(db.String(255), nullable=False)
    patient_name = db.Column(db.String(255), nullable=False)
    emergency_type = db.Column(db.String(50), nullable=False)
    description = db.Column(db.Text, nullable=False)
    contact_number = db.Column(db.String(20), nullable=False)
    status = db.Column(db.String(20), default='pending')
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)