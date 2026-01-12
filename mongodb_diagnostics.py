"""
MongoDB Atlas Connection String Builder
Help diagnose connection issues
"""

print("\n" + "="*80)
print("MONGODB ATLAS CONNECTION DIAGNOSTICS")
print("="*80 + "\n")

print("Please verify these settings in your MongoDB Atlas dashboard:\n")

print("1. DATABASE ACCESS → Database Users")
print("   - Username: ayushbhagwatkar89_db_user")
print("   - Password: QkWXvFdwl3Id2O52")
print("   - Database User Privileges: 'Atlas admin' or 'Read and write to any database'")
print("   - Authentication Method: SCRAM\n")

print("2. NETWORK ACCESS → IP Access List")
print("   - Entry: 0.0.0.0/0 (Allow access from anywhere)")
print("   - OR: Add your current IP address\n")

print("3. DATABASE → Cluster")
print("   - Cluster name: cluster0")
print("   - Connection string format:")
print("   mongodb+srv://<username>:<password>@cluster0.opgn0zz.mongodb.net/\n")

print("="*80)
print("\nTROUBLESHOOTING STEPS:")
print("="*80 + "\n")

print("If authentication is still failing:")
print("1. Reset password in MongoDB Atlas:")
print("   - Go to Database Access")
print("   - Click 'Edit' on user ayushbhagwatkar89_db_user")
print("   - Click 'Edit Password'")
print("   - Use 'Autogenerate Secure Password' or set a new one")
print("   - Click 'Update User'")
print("   - Copy the new password\n")

print("2. Update .env file with new credentials\n")

print("3. Or try creating a NEW database user:")
print("   - Database Access → Add New Database User")
print("   - Username: chatbot_admin")
print("   - Password: (autogenerate)")
print("   - Database User Privileges: 'Atlas admin'")
print("   - Add User\n")

input("Press Enter after verifying/updating credentials in Atlas...")
