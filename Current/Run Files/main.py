import os
from dotenv import load_dotenv

# This tells Python to look for the .env file and load the variables
load_dotenv()

# Now you can access them using os.getenv
api_key = os.getenv("public_key")
secret_key = os.getenv("secret_key")

if api_key and secret_key:
    print("Keys loaded successfully!")
else:
    print("One or both keys not found.")