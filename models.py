import mysql.connector
from mysql.connector import Error
from datetime import datetime
from config import Config

def get_db_connection():
    try:
        connection = mysql.connector.connect(
            host=Config.MYSQL_HOST,
            user=Config.MYSQL_USER,
            password=Config.MYSQL_PASSWORD,
            database=Config.MYSQL_DB
        )
        return connection
    except Error as e:
        print(f"Error connecting to MySQL Database: {e}")
        return None

def init_db():
    connection = get_db_connection()
    if connection:
        try:
            cursor = connection.cursor()
            
            # Create users table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    username VARCHAR(50) UNIQUE NOT NULL,
                    password VARCHAR(255) NOT NULL,
                    email VARCHAR(100) UNIQUE NOT NULL,
                    role ENUM('user', 'admin') DEFAULT 'user',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create predictions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS predictions (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    user_id INT,
                    location VARCHAR(255),
                    latitude FLOAT,
                    longitude FLOAT,
                    severity VARCHAR(50),
                    confidence FLOAT,
                    weather_data JSON,
                    traffic_data JSON,
                    risk_factors JSON,
                    recommendations JSON,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users(id)
                )
            """)
            
            # Create hospital_requests table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS hospital_requests (
                    id VARCHAR(36) PRIMARY KEY,
                    user_id INT,
                    hospital_id VARCHAR(50),
                    patient_name VARCHAR(100),
                    emergency_type VARCHAR(50),
                    description TEXT,
                    contact_number VARCHAR(20),
                    status ENUM('pending', 'approved', 'rejected') DEFAULT 'pending',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users(id)
                )
            """)
            
            # Create default admin user if not exists
            cursor.execute("SELECT * FROM users WHERE username = 'admin'")
            if not cursor.fetchone():
                cursor.execute("""
                    INSERT INTO users (username, password, email, role)
                    VALUES ('admin', 'admin123', 'admin@example.com', 'admin')
                """)
            
            connection.commit()
            print("Database initialized successfully")
            
        except Error as e:
            print(f"Error initializing database: {e}")
        finally:
            cursor.close()
            connection.close()

class User:
    @staticmethod
    def create_user(username, password, email, role='user'):
        connection = get_db_connection()
        if connection:
            try:
                cursor = connection.cursor()
                cursor.execute("""
                    INSERT INTO users (username, password, email, role)
                    VALUES (%s, %s, %s, %s)
                """, (username, password, email, role))
                connection.commit()
                return True
            except Error as e:
                print(f"Error creating user: {e}")
                return False
            finally:
                cursor.close()
                connection.close()
        return False

    @staticmethod
    def authenticate(username, password):
        connection = get_db_connection()
        if connection:
            try:
                cursor = connection.cursor(dictionary=True)
                cursor.execute("""
                    SELECT * FROM users WHERE username = %s AND password = %s
                """, (username, password))
                user = cursor.fetchone()
                return user
            except Error as e:
                print(f"Error authenticating user: {e}")
                return None
            finally:
                cursor.close()
                connection.close()
        return None

class Prediction:
    @staticmethod
    def save_prediction(user_id, data):
        connection = get_db_connection()
        if connection:
            try:
                cursor = connection.cursor()
                cursor.execute("""
                    INSERT INTO predictions (
                        user_id, location, latitude, longitude, severity,
                        confidence, weather_data, traffic_data, risk_factors, recommendations
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    user_id,
                    data['location']['name'],
                    data['location']['lat'],
                    data['location']['lon'],
                    data['severity'],
                    data['confidence'],
                    str(data['weather']),
                    str(data['traffic']),
                    str(data['risk_factors']),
                    str(data['recommendations'])
                ))
                connection.commit()
                return True
            except Error as e:
                print(f"Error saving prediction: {e}")
                return False
            finally:
                cursor.close()
                connection.close()
        return False

    @staticmethod
    def get_user_predictions(user_id):
        connection = get_db_connection()
        if connection:
            try:
                cursor = connection.cursor(dictionary=True)
                cursor.execute("""
                    SELECT * FROM predictions WHERE user_id = %s ORDER BY created_at DESC
                """, (user_id,))
                predictions = cursor.fetchall()
                return predictions
            except Error as e:
                print(f"Error getting predictions: {e}")
                return []
            finally:
                cursor.close()
                connection.close()
        return []

class HospitalRequest:
    @staticmethod
    def create_request(user_id, data):
        connection = get_db_connection()
        if connection:
            try:
                cursor = connection.cursor()
                cursor.execute("""
                    INSERT INTO hospital_requests (
                        id, user_id, hospital_id, patient_name,
                        emergency_type, description, contact_number
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s)
                """, (
                    data['id'],
                    user_id,
                    data['hospital_id'],
                    data['patient_name'],
                    data['emergency_type'],
                    data['description'],
                    data['contact_number']
                ))
                connection.commit()
                return True
            except Error as e:
                print(f"Error creating hospital request: {e}")
                return False
            finally:
                cursor.close()
                connection.close()
        return False

    @staticmethod
    def get_user_requests(user_id):
        connection = get_db_connection()
        if connection:
            try:
                cursor = connection.cursor(dictionary=True)
                cursor.execute("""
                    SELECT * FROM hospital_requests WHERE user_id = %s ORDER BY created_at DESC
                """, (user_id,))
                requests = cursor.fetchall()
                return requests
            except Error as e:
                print(f"Error getting user requests: {e}")
                return []
            finally:
                cursor.close()
                connection.close()
        return []

    @staticmethod
    def get_all_requests():
        connection = get_db_connection()
        if connection:
            try:
                cursor = connection.cursor(dictionary=True)
                cursor.execute("""
                    SELECT r.*, u.username as user_name
                    FROM hospital_requests r
                    JOIN users u ON r.user_id = u.id
                    ORDER BY r.created_at DESC
                """)
                requests = cursor.fetchall()
                return requests
            except Error as e:
                print(f"Error getting all requests: {e}")
                return []
            finally:
                cursor.close()
                connection.close()
        return []

    @staticmethod
    def update_request_status(request_id, status):
        connection = get_db_connection()
        if connection:
            try:
                cursor = connection.cursor()
                cursor.execute("""
                    UPDATE hospital_requests SET status = %s WHERE id = %s
                """, (status, request_id))
                connection.commit()
                return True
            except Error as e:
                print(f"Error updating request status: {e}")
                return False
            finally:
                cursor.close()
                connection.close()
        return False 