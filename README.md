# Accident Severity Prediction System 🚨

A real-time accident severity prediction system with emergency response chatbot and interactive dashboard.

## Features 🌟

- Real-time accident severity prediction
- Emergency response chatbot (24/7)
- Interactive dashboard with visualizations
- Weather condition integration
- Nearby hospital finder
- Emergency services coordination
- Risk factor analysis
- Location-based predictions

## Prerequisites 📋

- Python 3.8 or higher
- Git
- Google Maps API key
- OpenWeatherMap API key
- Gemini Pro API key

## Installation 🛠️

1. Clone the repository:
```bash
git clone https://github.com/manivenkatasai9494/Accident_Severity
cd Accident_Severity
```

2. Create and activate virtual environment:
```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# Linux/Mac
python3 -m venv .venv
source .venv/bin/activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
Create a `.env` file in the root directory with:
```env
GOOGLE_API_KEY=your_gemini_api_key
WEATHER_API_KEY=your_openweathermap_api_key
```

## Dataset Setup 📊

1. Download the dataset:
   - Visit [US Accidents Dataset (March 2023)](https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents?resource=download)
   - Download the CSV file

2. Place the dataset:
   - Create a `data` directory in the project root
   - Place the downloaded CSV file in the `data` directory
   - Rename it to `US_Accidents_March23.csv`

## Running the Application 🚀

1. Start the Flask server:
```bash
python app.py
```

2. Access the application:
   - Open your browser
   - Visit `http://localhost:5000`

## Project Structure 📁

```
Accident_Severity_Prediction/
├── app.py                 # Main Flask application
├── requirements.txt       # Python dependencies
├── .env                  # Environment variables
├── data/                 # Dataset directory
├── static/              # Static files (CSS, JS)
├── templates/           # HTML templates
└── models/              # Trained ML models
```

## API Endpoints 🌐

- `/` - Home page
- `/predict` - Accident severity prediction
- `/dashboard` - Interactive dashboard
- `/emergency` - Emergency services
- `/chat` - Emergency response chatbot
- `/api/dashboard/stats` - Dashboard statistics

## Features in Detail 🔍

### 1. Accident Prediction
- Real-time severity prediction
- Weather condition integration
- Risk factor analysis
- Confidence scoring

### 2. Emergency Chatbot
- 24/7 emergency response
- First aid guidance
- Safety precautions
- Emergency contact information

### 3. Dashboard
- Real-time statistics
- Severity distribution
- Weather impact analysis
- Accident hotspots
- Time-based analysis

### 4. Emergency Services
- Nearby hospital finder
- Emergency contact numbers
- Quick response guidelines
- Location-based services

## Contributing 🤝

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

