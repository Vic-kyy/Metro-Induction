import os
from supabase import create_client, Client
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

url: str = os.environ.get("SUPABASE_URL")
key: str = os.environ.get("SUPABASE_ANON_KEY")

supabase: Client = None

try:
    if url and key:
        supabase = create_client(url, key)
        print("✅ Supabase client initialized")
    else:
        print("⚠️ Supabase credentials missing from environment")
except Exception as e:
    print(f"❌ Failed to initialize Supabase client: {e}")

def get_supabase():
    return supabase

if __name__ == "__main__":
    # Simple connectivity test
    if supabase:
        try:
            # Try to fetch something simple or just check client
            print("Testing connection...")
            # We don't know the table names for sure yet, but we can try a health check
            # For now, just confirming client exists is a good first step
            print(f"Client instance: {supabase}")
        except Exception as e:
            print(f"Connection test failed: {e}")
