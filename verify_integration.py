import requests
import time
import subprocess
import os

def test_endpoints():
    base_url = "http://localhost:5011"
    endpoints = [
        "/api/fleet",
        "/api/cleaning/summary",
        "/api/induction/master-plan",
        "/api/whatif/health",
        "/api/health"
    ]
    
    print("🚀 Starting verification...")
    
    # Start the app in the background
    process = subprocess.Popen(["python3", "app.py"], cwd="/Users/vic/Desktop/KMRLSIH 5")
    time.sleep(5) # Give it time to start
    
    success_count = 0
    for ep in endpoints:
        try:
            print(f"Testing {ep}...", end=" ", flush=True)
            url = f"{base_url}{ep}"
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                print("✅ OK")
                success_count += 1
            else:
                print(f"❌ Failed (Status: {response.status_code})")
        except Exception as e:
            print(f"❌ Error: {e}")
            
    # Cleanup
    process.terminate()
    
    print(f"\nVerification finished: {success_count}/{len(endpoints)} passed.")

if __name__ == "__main__":
    test_endpoints()
