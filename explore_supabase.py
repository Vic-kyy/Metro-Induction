from supabase_client import get_supabase
import pandas as pd

supabase = get_supabase()

def test_fetch():
    if not supabase:
        print("Supabase client not initialized")
        return

    print("Fetching tables information...")
    # Since we don't have a direct 'list tables' in the client easily without RPC, 
    # let's try to fetch from the known tables in the code
    tables = ['train_status', 'system_config', 'stabling_tracks', 'kmrl_parts_cost', 'kmrl_relevant_parts_cost']
    
    for table in tables:
        try:
            print(f"\n--- Table: {table} ---")
            response = supabase.table(table).select("*", count="exact").limit(5).execute()
            print(f"Count: {response.count}")
            if response.data:
                df = pd.DataFrame(response.data)
                print(df.head())
            else:
                print("No data found")
        except Exception as e:
            print(f"Error fetching from {table}: {e}")

if __name__ == "__main__":
    test_fetch()
