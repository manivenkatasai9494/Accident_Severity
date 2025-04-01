import mysql.connector
from mysql.connector import Error
from config import Config

def create_database():
    try:
        # Connect to MySQL server without database
        connection = mysql.connector.connect(
            host=Config.MYSQL_HOST,
            user=Config.MYSQL_USER,
            password=Config.MYSQL_PASSWORD
        )
        
        if connection.is_connected():
            cursor = connection.cursor()
            
            # Create database if it doesn't exist
            cursor.execute(f"CREATE DATABASE IF NOT EXISTS {Config.MYSQL_DB}")
            print(f"Database '{Config.MYSQL_DB}' created successfully")
            
            cursor.close()
            connection.close()
            print("MySQL connection closed")
            
    except Error as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    create_database() 