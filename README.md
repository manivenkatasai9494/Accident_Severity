﻿# Accident Severity Prediction And Resonse System

A real-time accident severity prediction and emergency response coordination system that helps emergency services respond more effectively to accidents by predicting severity and coordinating with nearby hospitals.

## Features

- *Accident Severity Prediction*: Predicts accident severity (Low, Medium, High) based on various factors
- *Real-time Hospital Coordination*: Automatically notifies and coordinates with nearby hospitals
- *Emergency Response Dashboard*: Real-time status tracking and resource management
- *Interactive Maps*: Visual representation of accident locations and nearby hospitals
- *Real-time Status Updates*: Live tracking of hospital responses and emergency vehicle locations
- *Chat Widget*: Built-in communication system for emergency coordination
- *Admin Dashboard*: Comprehensive management interface for system administrators

## Prerequisites

- Python 3.8 or higher
- pip (Python package installer)
- Google Maps API key (for maps functionality)
- MySQL database

## Drive Link
[Project Drive Folder](https://drive.google.com/file/d/1DiBCoQd0acFUVbEWX6END1SWnT7zmbnH/view)

## Images
![Project Architecture](Screenshot (243).png)

## Installation

1. Clone the repository:
bash
git clone https://github.com/manivenkatasai9494/Accident_Severity.git
cd Accident_Severity


2. Create and activate a virtual environment (recommended):
bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate


3. Install required packages:
bash
pip install -r requirements.txt


4. Set up environment variables:
Create a .env file in the project root with the following variables:

GOOGLE_MAPS_API_KEY=your_api_key_here
DATABASE_URL=mysql://username:password@localhost/dbname
SECRET_KEY=your_secret_key_here


5. Initialize the database:
bash
python -c "from app import app, db; app.app_context().push(); db.create_all()"


## Running the Application

1. Start the Flask application:
bash
python app.py


2. Access the application:
- Main application: http://localhost:5000
- Admin dashboard: http://localhost:5000/admin (username: admin, password: admin123)

3. To access from other devices on your network:
- Find your computer's IP address:
  - Windows: Open CMD and type ipconfig
  - Linux/Mac: Open terminal and type ifconfig or ip addr
- Access the application using: http://<your-ip-address>:5000

## Project Structure


Accident_Severity/
├── app.py                 # Main application file
├── models.py             # Database models
├── requirements.txt      # Python dependencies
├── static/              # Static files (CSS, JS, images)
├── templates/           # HTML templates
│   ├── index.html      # Main page
│   ├── admin.html      # Admin dashboard
│   └── emergency.html  # Emergency services page
└── README.md           # Project documentation


## Key Features in Detail

### 1. Accident Severity Prediction
- Input factors:
  - Number of vehicles
  - Number of casualties
  - Time of day
  - Weather conditions
  - Road type
  - Speed limit
- Output: Severity level (Low, Medium, High)

### 2. Hospital Coordination
- Automatic hospital notification
- Real-time status tracking
- Response time estimation
- Resource availability monitoring

### 3. Emergency Dashboard
- Real-time accident monitoring
- Hospital availability status
- Emergency vehicle tracking
- Interactive maps
- Status updates

### 4. Admin Features
- User management
- System monitoring
- Emergency request management
- Hospital coordination
- Analytics and reporting

## API Endpoints

### Emergency Services
- POST /api/predict: Predict accident severity
- POST /api/emergency/request: Submit emergency request
- GET /api/emergency/status: Get emergency status

### Hospital Management
- POST /api/hospital/request: Submit hospital request
- GET /api/hospital/status: Get hospital status
- PUT /api/hospital/update: Update hospital status

### Admin
- GET /api/admin/requests: Get all emergency requests
- PUT /api/admin/request/status: Update request status
- GET /api/admin/analytics: Get system analytics

## Security Features

- User authentication
- Role-based access control
- Secure API endpoints
- Data encryption
- Session management

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## Troubleshooting

### Common Issues

1. Database Connection Error
   - Check database credentials in .env
   - Ensure MySQL service is running
   - Verify database exists

2. API Key Issues
   - Verify Google Maps API key is valid
   - Check API key permissions
   - Ensure key is properly set in .env

3. Port Already in Use
   - Check if another process is using port 5000
   - Use a different port by modifying app.py

### Getting Help

- Check the documentation
- Review existing issues
- Create a new issue with detailed information

## License

This project is licensed under the MIT License - see the LICENSE file for details.

