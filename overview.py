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
    @app.route("/")
    def overview():
        return render_template("overview.html")

    @app.route('/api/fleet', methods=['GET'])
    def get_fleet_route():
        return get_fleet()

    @app.route('/api/health')
    def overview_health_route():
        return overview_health_check()

    @app.route('/api/simulation-status')
    def simulation_status_route():
        return simulation_status()

    @app.route('/api/toggle-simulation', methods=['POST'])
    def toggle_simulation_route():
        return toggle_simulation()

    @app.route('/api/stats')
    def get_stats_route():
        return get_stats()

# Database connection parameters
DB_CONN = (
    "postgresql://postgres:RkoGkPLWxh4vavX3"
    "@db.trwsfdhxzwzkjandsmvz.supabase.co:5432/postgres"
    "?sslmode=require"
)

# Global variables
model = None
scaler = None
simulation_active = True

def load_ml_models():
    global model, scaler
    try:
        # Try to load your trained models first
        model_paths = [
            "realistic_kochi_metro_rf_model.pkl",  # Your trained model
            "models/realistic_kochi_metro_rf_model.pkl"  # Backup location
        ]
        scaler_paths = [
            "realistic_kochi_metro_scaler.pkl",  # Your trained scaler
            "models/realistic_kochi_metro_scaler.pkl"  # Backup location
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
            
            # Debug: Check model features
            if hasattr(model, 'n_features_in_'):
                print(f"   - Model expects {model.n_features_in_} features")
            if hasattr(model, 'feature_names_in_'):
                print(f"   - Model type: {type(model).__name__}")
                print(f"   - Feature count: {len(model.feature_names_in_) if model.feature_names_in_ is not None else 'Unknown'}")
        else:
            print("⚠️ Your trained ML model files not found.")
            print("   Expected files: realistic_kochi_metro_rf_model.pkl, realistic_kochi_metro_scaler.pkl")
            print("   System will use rule-based health calculation.")
            
    except Exception as e:
        print(f"⚠️ Error loading your trained ML models: {e}")
        print("   Using fallback health calculation.")
        model = None
        scaler = None

# Load models on startup
load_ml_models()

def simulate_operational_changes(df, time_elapsed_minutes=1):
    """
    Simulate real operational changes based on time elapsed
    time_elapsed_minutes: simulation time step (1 minute = realistic refresh interval)
    """
    current_time = datetime.now()
    hour = current_time.hour
    day_of_week = current_time.weekday()  # 0=Monday, 6=Sunday
    
    print(f"🚇 Simulating {time_elapsed_minutes} minutes of operation (Hour: {hour:02d}:xx)")
    
    changes_log = {
        'mileage_added': 0,
        'brake_wear_increase': 0,
        'new_faults': 0,
        'repairs_completed': 0,
        'temperature_changes': 0,
        'maintenance_scheduled': 0
    }
    
    # Operating schedule: 5:30 AM to 11:00 PM
    is_operating_hours = 5.5 <= hour <= 23.0
    
    # Peak hours: 7-9 AM, 5-8 PM on weekdays
    is_peak_hours = (
        (7 <= hour <= 9 or 17 <= hour <= 20) and 
        day_of_week < 5  # Monday to Friday
    )
    
    for idx, row in df.iterrows():
        train_id = row.get('train_number', f'KMR-{1001+idx}')
        current_status = row.get('status', 'Active')
        
        # INITIALIZE distance_added at the start of each iteration
        distance_added = 0
        
        # Only active trains operate and accumulate changes
        if current_status == 'Active' and is_operating_hours:
            
            # 1. MILEAGE ACCUMULATION
            if is_peak_hours:
                # Peak hours: 2-4 trips per hour, 25-45 km per trip
                trips_per_minute = random.uniform(0.05, 0.08)  # 3-5 trips/hour
                km_per_trip = random.uniform(25, 45)
            else:
                # Off-peak: 1-2 trips per hour, 20-35 km per trip  
                trips_per_minute = random.uniform(0.02, 0.04)  # 1-2.5 trips/hour
                km_per_trip = random.uniform(20, 35)
            
            distance_added = trips_per_minute * time_elapsed_minutes * km_per_trip
            if distance_added > 0:
                df.at[idx, 'mileage_km'] = row['mileage_km'] + distance_added
                changes_log['mileage_added'] += distance_added
            
            # 2. BRAKE WEAR (increases with usage and emergency stops)
            brake_wear_per_km = random.uniform(0.008, 0.015)  # 0.8-1.5% per 100km
            emergency_brake_chance = 0.02 if is_peak_hours else 0.005
            
            brake_increase = distance_added * brake_wear_per_km
            if random.random() < emergency_brake_chance * time_elapsed_minutes:
                brake_increase += random.uniform(0.5, 2.0)  # Emergency brake wear
            
            if brake_increase > 0:
                current_brake_wear = min(100, row['brake_wear'] + brake_increase)
                df.at[idx, 'brake_wear'] = current_brake_wear
                changes_log['brake_wear_increase'] += brake_increase
            
            # 3. MOTOR TEMPERATURE (varies by load, weather, and operation)
            base_temp = 58
            
            # Temperature factors
            if is_peak_hours:
                load_temp = random.uniform(8, 18)  # High passenger load
            else:
                load_temp = random.uniform(-2, 8)   # Lower load
            
            # Weather simulation (simplified)
            if 11 <= hour <= 16:  # Afternoon heat
                weather_temp = random.uniform(5, 12)
            else:
                weather_temp = random.uniform(-5, 5)
            
            # Operational stress
            if distance_added > 20:  # Heavy operation
                stress_temp = random.uniform(3, 8)
            else:
                stress_temp = random.uniform(-2, 3)
            
            new_temp = base_temp + load_temp + weather_temp + stress_temp
            new_temp = max(45, min(95, new_temp))  # Realistic bounds
            
            if abs(new_temp - row['motor_temp']) > 1:
                df.at[idx, 'motor_temp'] = new_temp
                changes_log['temperature_changes'] += 1
            
            # 4. DOOR OPERATIONS & FAILURES
            # Doors cycle every station stop (8-15 stops per trip)
            if distance_added > 10:  # Significant distance covered
                stops_per_km = 0.4  # Roughly every 2.5km
                door_cycles = int(distance_added * stops_per_km * 2)  # 2 doors per stop
                df.at[idx, 'door_cycle_count'] = row['door_cycle_count'] + door_cycles
                
                # Door failure probability increases with cycles and age
                failure_rate = 0.0001 + (row['door_cycle_count'] / 50000) * 0.0005
                if random.random() < failure_rate * door_cycles:
                    df.at[idx, 'door_failures'] = row['door_failures'] + 1
                    changes_log['new_faults'] += 1
            
            # 5. ENERGY CONSUMPTION (realistic variation)
            passenger_load_factor = random.uniform(0.3, 0.9) if is_peak_hours else random.uniform(0.1, 0.5)
            base_energy_per_km = 18  # kWh per km base
            load_energy = passenger_load_factor * 8  # Additional energy for passengers
            
            if distance_added > 0:
                energy_consumed = (base_energy_per_km + load_energy) * distance_added
                df.at[idx, 'energy_kwh'] = row['energy_kwh'] + energy_consumed
                
                # Update passenger count based on time
                capacity = row['passenger_capacity']
                if is_peak_hours:
                    df.at[idx, 'passengers_onboard'] = int(capacity * passenger_load_factor)
                else:
                    df.at[idx, 'passengers_onboard'] = int(capacity * passenger_load_factor * 0.6)
        
        # 6. MAINTENANCE & REPAIRS (happens regardless of operation)
        
        # Work order completion
        if row.get('work_order_status') in ['Pending', 'In Progress']:
            completion_chance = 0.008 * time_elapsed_minutes  # ~1% per 2 hours
            if random.random() < completion_chance:
                df.at[idx, 'work_order_status'] = 'Completed'
                changes_log['repairs_completed'] += 1
                
                # Completing work orders might improve health
                if row.get('pending_maintenance'):
                    df.at[idx, 'pending_maintenance'] = False
        
        # Scheduled maintenance triggers
        needs_maintenance = (
            row['brake_wear'] > 78 or 
            row['door_failures'] > 10 or 
            row['mileage_km'] > 24000 or
            row['motor_temp'] > 85
        )
        
        if needs_maintenance and not row.get('pending_maintenance', False):
            schedule_chance = 0.02 * time_elapsed_minutes  # 2% per hour for critical trains
            if random.random() < schedule_chance:
                df.at[idx, 'pending_maintenance'] = True
                df.at[idx, 'work_order_status'] = 'Scheduled'
                changes_log['maintenance_scheduled'] += 1
        
        # 7. RANDOM FAULTS & INCIDENTS
        
        # New fault codes
        if not row.get('fault_code') and current_status == 'Active':
            fault_chance = 0.0005 * time_elapsed_minutes  # Very low base rate
            if row['motor_temp'] > 75:
                fault_chance *= 3  # Higher chance when hot
            if row['brake_wear'] > 70:
                fault_chance *= 2  # Higher chance with worn brakes
                
            if random.random() < fault_chance:
                fault_codes = ['F101', 'F205', 'F312', 'F089', 'F156', 'F267']
                df.at[idx, 'fault_code'] = random.choice(fault_codes)
                changes_log['new_faults'] += 1
        
        # Clear minor faults occasionally
        elif row.get('fault_code') and current_status != 'Maintenance':
            clear_chance = 0.001 * time_elapsed_minutes
            if random.random() < clear_chance:
                df.at[idx, 'fault_code'] = None
        
        # 8. OPERATING COSTS UPDATE
        if distance_added > 0:
            cost_per_km = row.get('operating_cost_per_km', 2.0)
            additional_cost = distance_added * cost_per_km
            df.at[idx, 'total_operating_cost'] = row['total_operating_cost'] + additional_cost
    
    # Log significant changes
    if changes_log['mileage_added'] > 10:
        print(f"   📊 Fleet traveled {changes_log['mileage_added']:.1f} km total")
    if changes_log['brake_wear_increase'] > 1:
        print(f"   🔧 Brake wear increased by {changes_log['brake_wear_increase']:.2f}% total")  
    if changes_log['new_faults'] > 0:
        print(f"   ⚠️ {changes_log['new_faults']} new faults detected")
    if changes_log['repairs_completed'] > 0:
        print(f"   ✅ {changes_log['repairs_completed']} work orders completed")
    if changes_log['maintenance_scheduled'] > 0:
        print(f"   🛠️ {changes_log['maintenance_scheduled']} trains scheduled for maintenance")
    
    return df
def calculate_health_score(train_data):
    """Calculate health score based on multiple factors (fallback method)"""
    try:
        # Key factors affecting train health
        mtbf = float(train_data.get('mtbf', 2500))
        brake_wear = int(train_data.get('brake_wear', 50))
        mileage = int(train_data.get('mileage_km', 10000))
        motor_temp = float(train_data.get('motor_temp', 60))
        door_failures = int(train_data.get('door_failures', 2))
        incident_reports = int(train_data.get('incident_reports', 0))
        operating_hours = float(train_data.get('operating_hours', 8))
        
        # Base health score
        health = 100
        
        # Reduce health based on various factors
        # MTBF impact (lower MTBF = lower health)
        if mtbf < 2000:
            health -= 20
        elif mtbf < 2500:
            health -= 10
        
        # Brake wear impact (higher wear = lower health)
        if brake_wear > 60:
            health -= 15
        elif brake_wear > 40:
            health -= 8
        
        # Mileage impact (higher mileage = lower health)
        if mileage > 20000:
            health -= 12
        elif mileage > 15000:
            health -= 6
        
        # Motor temperature impact (higher temp = lower health)
        if motor_temp > 70:
            health -= 15
        elif motor_temp > 65:
            health -= 8
        
        # Door failures impact
        health -= door_failures * 3
        
        # Incident reports impact
        health -= incident_reports * 5
        
        # Operating hours impact (overworking)
        if operating_hours > 12:
            health -= 10
        elif operating_hours > 10:
            health -= 5
        
        # Add some randomness for realism
        health += np.random.randint(-5, 6)
        
        # Ensure health is within bounds
        health = max(30, min(100, health))
        
        return int(health)
        
    except Exception as e:
        print(f"Error calculating health score: {e}")
        return np.random.randint(60, 85)

def predict_with_ml(df):
    """Use ML model for prediction - Updated for your trained Random Forest model"""
    try:
        if model is None or scaler is None:
            return None
            
        print(f"🤖 Using trained Random Forest model for {len(df)} trains")
        
        # Your model's exact features (based on your training code)
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
        
        # Prepare feature matrix with actual database values
        X = pd.DataFrame()
        
        # Handle each feature according to your training setup
        for feature in expected_features:
            if feature in df.columns:
                if feature == 'fault_code':
                    # Encode fault codes (0=no fault, 1-5=different faults)
                    X[feature] = df[feature].fillna('').apply(lambda x: 0 if x == '' or pd.isna(x) else hash(str(x)) % 5 + 1)
                elif feature in ['depot_location', 'depot_position', 'cleaning_status', 'work_order_status', 
                               'depot_accessibility', 'advertising_commitments']:
                    # Encode categorical features
                    X[feature] = df[feature].fillna('Unknown').apply(lambda x: hash(str(x)) % 10)
                elif feature in ['hvac_status', 'compliance_status', 'fitness_certificate', 
                               'pending_maintenance', 'standby_requirement']:
                    # Convert boolean to int
                    X[feature] = pd.to_numeric(df[feature], errors='coerce').fillna(0).astype(int)
                else:
                    # Numeric features - use actual values from database
                    X[feature] = pd.to_numeric(df[feature], errors='coerce').fillna(0)
                    
                print(f"   ✅ Using database value for: {feature}")
            else:
                # This should rarely happen now since we fetch all columns
                print(f"   ❌ Missing column in database: {feature}")
                # Provide realistic defaults only if absolutely necessary
                defaults = {
                    'stabling_geometry_score': 0.8, 'occupancy_ratio': 0.5, 'energy_per_km': 0.04
                }
                X[feature] = defaults.get(feature, 0)
        
        # Calculate derived features (as in your training)
        if 'occupancy_ratio' not in df.columns:
            X['occupancy_ratio'] = X['passengers_onboard'] / (X['passenger_capacity'] + 1e-3)
        if 'energy_per_km' not in df.columns:
            X['energy_per_km'] = X['energy_kwh'] / (X['mileage_km'] + 1e-3)
        
        # Get numeric columns for scaling
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        
        # Scale numeric features only
        X_scaled = X.copy()
        X_scaled[numeric_cols] = scaler.transform(X[numeric_cols])
        
        # Make predictions (your model predicts readiness_score)
        readiness_predictions = model.predict(X_scaled)
        
        # Convert readiness score to health percentage
        # Your model outputs readiness scores, convert to health percentage (30-95%)
        min_health, max_health = readiness_predictions.min(), readiness_predictions.max()
        health_predictions = 30 + ((readiness_predictions - min_health) / 
                                 (max_health - min_health + 1e-6)) * 65
        
        # Ensure bounds and convert to int
        health_predictions = np.clip(health_predictions, 30, 95).astype(int)
        
        print(f"✅ ML predictions completed: Health range {health_predictions.min()}-{health_predictions.max()}%")
        print(f"   Average health: {health_predictions.mean():.1f}%")
        
        return health_predictions
        
    except Exception as e:
        print(f"❌ ML prediction error: {e}")
        import traceback
        traceback.print_exc()
        return None

def determine_status(health, additional_factors=None):
    """Determine train status based on health score and additional factors"""
    
    # PRIMARY RULE: Health score determines base status
    if health >= 80:
        base_status = "Active"
    elif health >= 40:
        base_status = "Alert"  
      # Still operational but needs attention
    else:
        base_status = "Maintenance"  # Too low to operate
    
    # SECONDARY RULES: Only apply to medium/low health trains
    if additional_factors and health < 80:  # Only affect non-excellent trains
        
        # Pending maintenance can downgrade Alert to Maintenance
        if additional_factors.get('pending_maintenance', False) and base_status == "Alert":
            base_status = "Maintenance"
        
        # Critical work orders can downgrade Alert to Maintenance  
        work_order = additional_factors.get('work_order_status', 'Completed')
        if work_order in ['Pending', 'Overdue'] and base_status == "Alert":
            base_status = "Maintenance"
    
    # SAFETY CHECK: Never put high-health trains in maintenance
    if health >= 85 and base_status == "Maintenance":
        base_status = "Active"
    elif health >= 75 and base_status == "Maintenance":
        base_status = "Alert"
    
    return base_status

def format_next_maintenance(retirement_date, health):
    """Calculate next maintenance date based on health and current date"""
    try:
        if isinstance(retirement_date, str):
            retirement = datetime.strptime(retirement_date, '%Y-%m-%d').date()
        else:
            retirement = retirement_date
            
        # Calculate maintenance frequency based on health
        if health < 50:
            days_until = np.random.randint(1, 7)  # Urgent maintenance
        elif health < 70:
            days_until = np.random.randint(7, 21)  # Regular maintenance
        else:
            days_until = np.random.randint(21, 60)  # Scheduled maintenance
            
        next_maintenance = datetime.now().date() + timedelta(days=days_until)
        return next_maintenance.strftime('%Y-%m-%d')
        
    except:
        return (datetime.now().date() + timedelta(days=np.random.randint(7, 30))).strftime('%Y-%m-%d')

def get_fleet():
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
        
        # Query train data with ALL columns needed for ML model
        cur.execute("""
            SELECT * FROM train_status 
            ORDER BY train_number
        """)
        
        fleet_data = cur.fetchall()
        
        if not fleet_data:
            raise Exception("No data found")
        
        # Convert to DataFrame
        df = pd.DataFrame(fleet_data)
        print(f"📊 Loaded {len(df)} trains from database (Fallback)")
        return process_fleet_response(df)

    except Exception as e:
        print(f"❌ Database error: {e}")
        # Return sample data on database error
        sample_data = [
            {'id': 'KMR-1001', 'status': 'Active', 'health': 88, 'mileage': 12450, 'nextMaintenance': '2024-06-15'},
            {'id': 'KMR-1002', 'status': 'Maintenance', 'health': 35, 'mileage': 18230, 'nextMaintenance': '2024-05-22'},
            {'id': 'KMR-1003', 'status': 'Alert', 'health': 65, 'mileage': 9870, 'nextMaintenance': '2024-07-10'},
            {'id': 'KMR-1004', 'status': 'Active', 'health': 92, 'mileage': 8340, 'nextMaintenance': '2024-07-20'},
            {'id': 'KMR-1005', 'status': 'Alert', 'health': 68, 'mileage': 15600, 'nextMaintenance': '2024-06-05'}
        ]
        return jsonify(sample_data)

def process_fleet_response(df):
    global simulation_active
    
    # Simulate operational changes if simulation is active
    if simulation_active:
        df = simulate_operational_changes(df, time_elapsed_minutes=1)
    
    # Apply ML predictions to the data
    ml_predictions = predict_with_ml(df)
    
    if ml_predictions is not None:
        df['health'] = ml_predictions
        print("🤖 Using ML model predictions")
    else:
        # Fallback to rule-based health calculation
        df['health'] = df.apply(lambda row: calculate_health_score(row), axis=1)
        print("📋 Using rule-based health calculation")
    
    # Determine status with additional factors for realism
    df['status'] = df.apply(lambda row: determine_status(
        row['health'], 
        {
            'pending_maintenance': row.get('pending_maintenance', False),
            'work_order_status': row.get('work_order_status', 'Completed'),
            'mileage_km': row.get('mileage_km', 0),
            'brake_wear': row.get('brake_wear', 0)
        }
    ), axis=1)
    
    # Debug: Log status distribution
    status_counts = df['status'].value_counts()
    print(f"📊 Status Distribution:")
    for status, count in status_counts.items():
        print(f"   {status}: {count} trains")
    
    df['nextMaintenance'] = df.apply(lambda row: format_next_maintenance(row['retirement_date'], row['health']), axis=1)
    
    # Prepare response data
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

def overview_health_check():
    """System health check endpoint"""
    try:
        supabase = supabase_client.get_supabase()
        db_connected = False
        if supabase:
            try:
                # Check supabase connection
                supabase.table('train_status').select("train_id", count="exact").limit(1).execute()
                db_connected = True
            except:
                pass
        
        return jsonify({
            "status": "healthy",
            "database": "connected" if db_connected else "fallback_mode",
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500
        
def simulation_status():
    """Get current simulation status and statistics"""
    try:
        conn = psycopg2.connect(DB_CONN)
        cur = conn.cursor(cursor_factory=RealDictCursor)
        
        cur.execute("""
            SELECT 
                train_number,
                mileage_km,
                brake_wear,
                motor_temp,
                door_failures,
                energy_kwh,
                passengers_onboard,
                passenger_capacity,
                work_order_status,
                pending_maintenance,
                fault_code
            FROM train_status 
            ORDER BY train_number
        """)
        
        trains = cur.fetchall()
        
        # Calculate fleet statistics
        stats = {
            'total_trains': len(trains),
            'total_mileage': sum(t['mileage_km'] for t in trains),
            'avg_brake_wear': sum(t['brake_wear'] for t in trains) / len(trains) if trains else 0,
            'avg_motor_temp': sum(t['motor_temp'] for t in trains) / len(trains) if trains else 0,
            'total_door_failures': sum(t['door_failures'] for t in trains),
            'total_passengers': sum(t['passengers_onboard'] for t in trains),
            'trains_with_faults': sum(1 for t in trains if t['fault_code']),
            'trains_pending_maintenance': sum(1 for t in trains if t['pending_maintenance']),
            'current_time': datetime.now().isoformat(),
            'simulation_active': simulation_active
        }
        
        cur.close()
        conn.close()
        
        return jsonify(stats)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def toggle_simulation():
    """Toggle dynamic simulation on/off"""
    global simulation_active
    simulation_active = not simulation_active
    
    return jsonify({
        'simulation_active': simulation_active,
        'message': f"Dynamic simulation {'enabled' if simulation_active else 'disabled'}"
    })

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

if __name__ == "__main__":
    print("🚀 Starting Kochi Metro Fleet Management System...")
    print(f"   - Database: {'✅ Connected' if DB_CONN else '❌ Not configured'}")
    print(f"   - ML Model: {'✅ Loaded' if model else '❌ Not available'}")
    print("   - Web interface: React frontend (port 3000)")
    print("   - API endpoints: /api/fleet, /api/health, /api/stats, /api/simulation-status, /api/toggle-simulation")
    print("\n🌐 Starting Flask server on port 5011...")
    app.run(debug=True, host='0.0.0.0', port=5011)