"""
Simple MongoDB Connection Test
"""
import os
from dotenv import load_dotenv
from pymongo import MongoClient

load_dotenv()

print("\n" + "="*80)
print("MONGODB CONNECTION TEST")
print("="*80 + "\n")

# Get connection string
mongodb_uri = os.getenv('MONGODB_URI')
print(f"1. Reading MONGODB_URI from .env...")
print(f"   URI found: {mongodb_uri is not None}")
print(f"   URI length: {len(mongodb_uri) if mongodb_uri else 0}\n")

if mongodb_uri:
    # Show URI (masked password)
    masked_uri = mongodb_uri.replace('QkWXvFdwl3Id2O52', '***PASSWORD***')
    print(f"   URI format: {masked_uri}\n")
    
    print("2. Attempting connection...")
    try:
        client = MongoClient(mongodb_uri, serverSelectionTimeoutMS=5000)
        
        print("3. Testing connection...")
        # Force connection
        client.admin.command('ping')
        
        print("   [SUCCESS] Connected to MongoDB!\n")
        
        # List databases
        print("4. Available databases:")
        dbs = client.list_database_names()
        for db in dbs:
            print(f"   - {db}")
        
        print("\n5. Testing database access...")
        db = client['mental_health_chatbot']
        print(f"   Database: {db.name}")
        print(f"   Collections: {db.list_collection_names()}")
        
        client.close()
        print("\n[SUCCESS] All tests passed!")
        
    except Exception as e:
        print(f"   [ERROR] Connection failed: {e}")
        print(f"   Error type: {type(e).__name__}")
else:
    print("   [ERROR] MONGODB_URI not found in .env file")

print("\n" + "="*80)
