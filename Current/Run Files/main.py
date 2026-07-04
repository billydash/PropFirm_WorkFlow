def load_keys():
    """
    Load API keys from environment variables.
    Returns a tuple of (api_key, secret_key).
    """
    import os
    from dotenv import load_dotenv

    # Load environment variables from .env file
    load_dotenv()

    # Retrieve the keys
    api_key = os.getenv("public_key")
    secret_key = os.getenv("secret_key")

    return api_key, secret_key
print(load_keys())
