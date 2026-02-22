from flask import Flask
from flask_cors import CORS

import overview
import fleet
import maintenance
import cleaning
import induction
import whatif
import feedback

# 1️⃣ Create app FIRST
app = Flask(__name__)
CORS(app)

# 2️⃣ Debug route to list all registered routes
@app.route("/__routes")
def list_routes():
    return {
        "routes": [str(r) for r in app.url_map.iter_rules()]
    }

# 3️⃣ Register all modules
modules = [
    ("Overview", overview.register),
    ("Fleet", fleet.register),
    ("Maintenance", maintenance.register),
    ("Cleaning", cleaning.register),
    ("Induction", induction.register),
    ("What-If", whatif.register),
    ("Feedback", feedback.register)
]

for name, register_fn in modules:
    try:
        register_fn(app)
        print(f"✅ {name} module registered")
    except Exception as e:
        print(f"❌ Failed to register {name} module: {e}")

try:
    whatif.load_models()
    whatif.test_db_connection()
except Exception as e:
    print(f"⚠️ What-If initialization error: {e}")

# 4️⃣ Run server
if __name__ == "__main__":
    app.run(debug=True, port=5011)