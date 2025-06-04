import mysql.connector
from config import Config

def clear_history():
    try:
        # Connect to MySQL server
        connection = mysql.connector.connect(
            host=Config.MYSQL_HOST,
            user=Config.MYSQL_USER,
            password=Config.MYSQL_PASSWORD,
            database=Config.MYSQL_DB
        )
        
        if connection.is_connected():
            cursor = connection.cursor()
            
            # Disable foreign key checks temporarily
            cursor.execute("SET FOREIGN_KEY_CHECKS = 0")
            
            # Clear hospital_requests table
            cursor.execute("TRUNCATE TABLE hospital_requests")
            print("Cleared hospital requests history")
            
            # Clear predictions table
            cursor.execute("TRUNCATE TABLE predictions")
            print("Cleared predictions history")
            
            # Clear users table except admin
            cursor.execute("DELETE FROM users WHERE role != 'admin'")
            print("Cleared regular users")
            
            # Enable foreign key checks
            cursor.execute("SET FOREIGN_KEY_CHECKS = 1")
            
            connection.commit()
            print("All history has been cleared successfully!")
            
            cursor.close()
            connection.close()
            print("MySQL connection closed")
            
    except mysql.connector.Error as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    clear_history()