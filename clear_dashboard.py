import mysql.connector
from config import Config

def clear_dashboard_data():
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
            
            # Clear Accidents table
            cursor.execute("TRUNCATE TABLE accident")
            print("✓ Cleared all accident records")
            
            # Clear Hospital Requests table
            cursor.execute("TRUNCATE TABLE hospital_request")
            print("✓ Cleared all hospital requests")
            
            # Enable foreign key checks
            cursor.execute("SET FOREIGN_KEY_CHECKS = 1")
            
            connection.commit()
            print("\n✨ Admin dashboard data cleared successfully!")
            print("Your application will now look fresh and new.")
            
            cursor.close()
            connection.close()
            
    except mysql.connector.Error as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    clear_dashboard_data()