from pymongo import MongoClient
import sys
from pymongo.errors import ConnectionFailure

def check_mongodb_connection(host="mongodb", port=27017, db_name="TheSocialAnalysis"):
    """
    Check if MongoDB is accessible at the specified host and port
    Args:
        host (str): MongoDB host
        port (int): MongoDB port
        db_name (str): Database name to check
    """
    try:
        # Construct the MongoDB URI
        uri = f"mongodb://{host}:{port}/{db_name}"
        print(f"Attempting to connect to MongoDB at: {uri}")
        
        # Try to establish a connection
        client = MongoClient(uri, serverSelectionTimeoutMS=5000)
        
        # Force a connection attempt
        client.server_info()
        
        print("✅ Successfully connected to MongoDB")
        print("Server info:", client.server_info())
        
        # List all databases
        print("\nAvailable databases:")
        for db in client.list_databases():
            print(f"- {db['name']}")
            
        client.close()
        return True
        
    except ConnectionFailure as e:
        print("❌ Failed to connect to MongoDB")
        print(f"Error: {e}")
        return False
    except Exception as e:
        print("❌ An error occurred")
        print(f"Error: {e}")
        return False

if __name__ == "__main__":
    
    check_mongodb_connection()