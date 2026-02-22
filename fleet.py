from flask import Flask, jsonify
from flask_cors import CORS
import joblib
import psycopg2
import pandas as pd
import numpy as np
import random
from psycopg2.extras import RealDictCursor
from datetime import datetime, timedelta
import os
from flask import render_template
import supabase_client

def register(app):

    # UI page
    @app.route("/fleet")
    def fleet_page():
        return render_template("fleet.html")

    # API
    @app.route("/api/fleet")
    def fleet_api():
        return get_fleet_data()

    @app.route("/api/health")
    def fleet_health_api():
        return jsonify({"status": "healthy"})

    @app.route("/api/stats")
    def stats_api():
        return jsonify({"ok": True})

# Database connection parameters
DB_CONN = "postgresql://postgres:RkoGkPLWxh4vavX3@db.trwsfdhxzwzkjandsmvz.supabase.co:5432/postgres"

# Load ML model & scaler
model = None
scaler = None

def load_ml_models():
    global model, scaler
    try:
        model_paths = [
            "realistic_kochi_metro_rf_model.pkl",
            "models/realistic_kochi_metro_rf_model.pkl"
        ]
        scaler_paths = [
            "realistic_kochi_metro_scaler.pkl",
            "models/realistic_kochi_metro_scaler.pkl"
        ]
        
        model_loaded = False
        scaler_loaded = False
        
        for model_path in model_paths:
            if os.path.exists(model_path):
                model = joblib.load(model_path)
                model_loaded = True
                print(f"✅ ML model loaded from: {model_path}")
                break
                
        for scaler_path in scaler_paths:
            if os.path.exists(scaler_path):
                scaler = joblib.load(scaler_path)
                scaler_loaded = True
                print(f"✅ Scaler loaded from: {scaler_path}")
                break
        
        if model_loaded and scaler_loaded:
            print("🤖 TRAINED RANDOM FOREST MODEL LOADED SUCCESSFULLY!")
            if hasattr(model, 'n_features_in_'):
                print(f"   - Model expects {model.n_features_in_} features")
        else:
            print("⚠️ ML model files not found. Using rule-based calculation.")
            
    except Exception as e:
        print(f"⚠️ Error loading ML models: {e}")
        model = None
        scaler = None

load_ml_models()

def calculate_health_score(train_data):
    """Calculate health score based on multiple factors (fallback method)"""
    try:
        mtbf = float(train_data.get('mtbf', 2500))
        brake_wear = int(train_data.get('brake_wear', 50))
        mileage = int(train_data.get('mileage_km', 10000))
        motor_temp = float(train_data.get('motor_temp', 60))
        door_failures = int(train_data.get('door_failures', 2))
        incident_reports = int(train_data.get('incident_reports', 0))
        operating_hours = float(train_data.get('operating_hours', 8))
        
        health = 100
        
        if mtbf < 2000:
            health -= 20
        elif mtbf < 2500:
            health -= 10
        
        if brake_wear > 60:
            health -= 15
        elif brake_wear > 40:
            health -= 8
        
        if mileage > 20000:
            health -= 12
        elif mileage > 15000:
            health -= 6
        
        if motor_temp > 70:
            health -= 15
        elif motor_temp > 65:
            health -= 8
        
        health -= door_failures * 3
        health -= incident_reports * 5
        
        if operating_hours > 12:
            health -= 10
        elif operating_hours > 10:
            health -= 5
        
        health += np.random.randint(-5, 6)
        health = max(30, min(100, health))
        
        return int(health)
        
    except Exception as e:
        print(f"Error calculating health score: {e}")
        return np.random.randint(60, 85)

def predict_with_ml(df):
    """Use ML model for prediction"""
    try:
        if model is None or scaler is None:
            return None
            
        print(f"🤖 Using trained Random Forest model for {len(df)} trains")
        
        expected_features = [
            'mtbf', 'brake_wear', 'energy_kwh', 'fault_code', 'mileage_km', 
            'motor_temp', 'trip_count', 'energy_cost', 'hvac_status', 
            'door_failures', 'available_bays', 'depot_location', 'depot_position',
            'battery_current', 'battery_voltage', 'cleaning_status', 'operating_hours',
            'door_cycle_count', 'incident_reports', 'maintenance_cost', 'compliance_status',
            'mileage_balancing', 'work_order_status', 'passenger_capacity', 'passengers_onboard',
            'depot_accessibility', 'fitness_certificate', 'pending_maintenance',
            'standby_requirement', 'total_operating_cost', 'operating_cost_per_km',
            'advertising_commitments', 'operating_cost_per_hour', 'stabling_geometry_score',
            'occupancy_ratio', 'energy_per_km'
        ]
        
        X = pd.DataFrame()
        
        for feature in expected_features:
            if feature in df.columns:
                if feature == 'fault_code':
                    X[feature] = df[feature].fillna('').apply(lambda x: 0 if x == '' or pd.isna(x) else hash(str(x)) % 5 + 1)
                elif feature in ['depot_location', 'depot_position', 'cleaning_status', 'work_order_status', 
                               'depot_accessibility', 'advertising_commitments']:
                    X[feature] = df[feature].fillna('Unknown').apply(lambda x: hash(str(x)) % 10)
                elif feature in ['hvac_status', 'compliance_status', 'fitness_certificate', 
                               'pending_maintenance', 'standby_requirement']:
                    X[feature] = pd.to_numeric(df[feature], errors='coerce').fillna(0).astype(int)
                else:
                    X[feature] = pd.to_numeric(df[feature], errors='coerce').fillna(0)
            else:
                defaults = {
                    'stabling_geometry_score': 0.8, 'occupancy_ratio': 0.5, 'energy_per_km': 0.04
                }
                X[feature] = defaults.get(feature, 0)
        
        if 'occupancy_ratio' not in df.columns:
            X['occupancy_ratio'] = X['passengers_onboard'] / (X['passenger_capacity'] + 1e-3)
        if 'energy_per_km' not in df.columns:
            X['energy_per_km'] = X['energy_kwh'] / (X['mileage_km'] + 1e-3)
        
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        X_scaled = X.copy()
        X_scaled[numeric_cols] = scaler.transform(X[numeric_cols])
        
        readiness_predictions = model.predict(X_scaled)
        
        min_health, max_health = readiness_predictions.min(), readiness_predictions.max()
        health_predictions = 30 + ((readiness_predictions - min_health) / 
                                 (max_health - min_health + 1e-6)) * 65
        
        health_predictions = np.clip(health_predictions, 30, 95).astype(int)
        
        print(f"✅ ML predictions: Health range {health_predictions.min()}-{health_predictions.max()}%")
        print(f"   Average health: {health_predictions.mean():.1f}%")
        
        return health_predictions
        
    except Exception as e:
        print(f"❌ ML prediction error: {e}")
        import traceback
        traceback.print_exc()
        return None

def determine_status(health, additional_factors=None):
    """Determine train status based on health score"""
    if health >= 80:
        base_status = "Active"
    elif health >= 65:
        base_status = "Alert"  
    elif health >= 45:
        base_status = "Alert"
    else:
        base_status = "Maintenance"
    
    if additional_factors and health < 80:
        if additional_factors.get('pending_maintenance', False) and base_status == "Alert":
            base_status = "Maintenance"
        
        work_order = additional_factors.get('work_order_status', 'Completed')
        if work_order in ['Pending', 'Overdue'] and base_status == "Alert":
            base_status = "Maintenance"
    
    if health >= 85 and base_status == "Maintenance":
        base_status = "Active"
    elif health >= 75 and base_status == "Maintenance":
        base_status = "Alert"
    
    return base_status

def format_next_maintenance(retirement_date, health):
    """Calculate next maintenance date"""
    try:
        if isinstance(retirement_date, str):
            retirement = datetime.strptime(retirement_date, '%Y-%m-%d').date()
        else:
            retirement = retirement_date
            
        if health < 50:
            days_until = np.random.randint(1, 7)
        elif health < 70:
            days_until = np.random.randint(7, 21)
        else:
            days_until = np.random.randint(21, 60)
            
        next_maintenance = datetime.now().date() + timedelta(days=days_until)
        return next_maintenance.strftime('%Y-%m-%d')
        
    except:
        return (datetime.now().date() + timedelta(days=np.random.randint(7, 30))).strftime('%Y-%m-%d')

def home():
    """Serve the fleet.html page"""
    return '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Kochi Metro Fleet API</title>
        <style>
            body { font-family: Arial, sans-serif; max-width: 800px; margin: 50px auto; padding: 20px; }
            h1 { color: #2563EB; }
            .endpoint { background: #f3f4f6; padding: 15px; margin: 10px 0; border-radius: 8px; }
            code { background: #1f2937; color: #10b981; padding: 2px 6px; border-radius: 4px; }
        </style>
    </head>
    <body>
        <h1>Kochi Metro Fleet Management API</h1>
        <p>Flask backend running on port 5001</p>
        
        <h2>Available Endpoints:</h2>
        
        <div class="endpoint">
            <h3>GET /fleet</h3>
            <p>Returns all fleet data with ML-powered health predictions</p>
            <code>http://localhost:5011/fleet</code>
        </div>
        
        <div class="endpoint">
            <h3>GET /health</h3>
            <p>API health check and system status</p>
            <code>http://localhost:5011/health</code>
        </div>
        
        <div class="endpoint">
            <h3>GET /stats</h3>
            <p>Fleet statistics and analytics</p>
            <code>http://localhost:5011/stats</code>
        </div>
        
        <p><strong>Frontend:</strong> Access the fleet management interface by opening <code>fleet.html</code> in your browser.</p>
    </body>
    </html>
    '''


def get_fleet_data():
    try:
        supabase = supabase_client.get_supabase()
        if supabase:
            try:
                response = supabase.table('train_status').select("*").order("train_number").execute()
                if response.data:
                    fleet_data = response.data
                    df = pd.DataFrame(fleet_data)
                    print(f"📊 Loaded {len(df)} trains from Supabase")
                    return process_fleet_response(df)
            except Exception as e:
                print(f"⚠️ Supabase fetch error: {e}")

        # Connect to database (Fallback)
        conn = psycopg2.connect(DB_CONN)
        cur = conn.cursor(cursor_factory=RealDictCursor)
        
        cur.execute("""
            SELECT * FROM train_status 
            ORDER BY train_number
        """)
        
        fleet_data = cur.fetchall()
        
        if not fleet_data:
            raise Exception("No data found")
        
        df = pd.DataFrame(fleet_data)
        print(f"📊 Loaded {len(df)} trains from database (Fallback)")
        return process_fleet_response(df)

    except Exception as e:
        print(f"❌ Database error: {e}")
        sample_data = [
            {'id': 'KMR-1001', 'status': 'Active', 'health': 88, 'mileage': 12450, 'nextMaintenance': '2024-06-15'},
            {'id': 'KMR-1002', 'status': 'Maintenance', 'health': 35, 'mileage': 18230, 'nextMaintenance': '2024-05-22'},
            {'id': 'KMR-1003', 'status': 'Alert', 'health': 65, 'mileage': 9870, 'nextMaintenance': '2024-07-10'},
            {'id': 'KMR-1004', 'status': 'Active', 'health': 92, 'mileage': 8340, 'nextMaintenance': '2024-07-20'},
            {'id': 'KMR-1005', 'status': 'Alert', 'health': 68, 'mileage': 15600, 'nextMaintenance': '2024-06-05'}
        ]
        return jsonify(sample_data)

def process_fleet_response(df):
    ml_predictions = predict_with_ml(df)
    
    if ml_predictions is not None:
        df['health'] = ml_predictions
        print("🤖 Using ML model predictions")
    else:
        df['health'] = df.apply(lambda row: calculate_health_score(row), axis=1)
        print("📋 Using rule-based health calculation")
    
    df['status'] = df.apply(lambda row: determine_status(
        row['health'], 
        {
            'pending_maintenance': row.get('pending_maintenance', False),
            'work_order_status': row.get('work_order_status', 'Completed'),
            'mileage_km': row.get('mileage_km', 0),
            'brake_wear': row.get('brake_wear', 0)
        }
    ), axis=1)
    
    status_counts = df['status'].value_counts()
    print(f"📊 Status Distribution:")
    for status, count in status_counts.items():
        print(f"   {status}: {count} trains")
    
    df['nextMaintenance'] = df.apply(lambda row: format_next_maintenance(row['retirement_date'], row['health']), axis=1)
    
    result = []
    for _, row in df.iterrows():
        result.append({
            'id': row['train_number'],
            'status': row['status'],
            'health': int(row['health']),
            'mileage': int(row['mileage_km']) if row['mileage_km'] else 0,
            'nextMaintenance': row['nextMaintenance']
        })
    
    return jsonify(result)

def fleet_health_check():
    """Health check endpoint"""
    try:
        conn = psycopg2.connect(DB_CONN)
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM train_status")
        train_count = cur.fetchone()[0]
        cur.close()
        conn.close()
        
        return jsonify({
            'status': 'healthy',
            'trains_in_db': train_count,
            'ml_model_loaded': model is not None,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

def get_stats():
    """Get detailed fleet statistics"""
    try:
        conn = psycopg2.connect(DB_CONN)
        cur = conn.cursor(cursor_factory=RealDictCursor)
        
        cur.execute("""
            SELECT 
                COUNT(*) as total_trains,
                AVG(mileage_km) as avg_mileage,
                AVG(mtbf) as avg_mtbf,
                AVG(brake_wear) as avg_brake_wear,
                COUNT(CASE WHEN brake_wear > 60 THEN 1 END) as high_brake_wear_count,
                COUNT(CASE WHEN door_failures > 3 THEN 1 END) as high_door_failures_count
            FROM train_status
        """)
        
        stats = cur.fetchone()
        cur.close()
        conn.close()
        
        return jsonify(dict(stats))
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500
