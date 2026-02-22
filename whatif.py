from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
import logging
import psycopg2
from psycopg2.extras import RealDictCursor
import json
import warnings
from flask import render_template
import pickle
import supabase_client


def whatif_page():
    return render_template("kochi_metro_whatif.html")

def whatif_health():
    return jsonify({"status": "ok"})


def whatif_predict_api():
    """
    API endpoint for What-If prediction
    """
    try:
        if not db_connection_status:
            return jsonify({
                "error": "Database not connected",
                "message": "Supabase DB unavailable"
            }), 503

        data = request.get_json()
        if not data:
            return jsonify({"error": "No input data provided"}), 400

        train_number = data.get("train_number")
        if not train_number:
            return jsonify({"error": "train_number is required"}), 400

        base_data = get_train_by_number(train_number)
        if not base_data:
            return jsonify({
                "error": "Train not found",
                "train_number": train_number
            }), 404

        final_data = base_data.copy()
        final_data.update(data)

        result = predict_with_model(final_data)

        return jsonify({
            "train_number": train_number,
            "prediction": result["prediction"],
            "category": result["category"],
            "category_class": result["category_class"],
            "prediction_method": result["prediction_method"],
            "timestamp": datetime.now().isoformat()
        })

    except Exception as e:
        logger.error(f"What-if prediction failed: {str(e)}")
        return jsonify({
            "error": "Prediction failed",
            "message": str(e)
        }), 500

def register(app):
    app.add_url_rule(
        "/whatif",
        view_func=whatif_page,
        methods=["GET"]
    )

    app.add_url_rule(
        "/api/whatif/health",
        view_func=whatif_health,
        methods=["GET"]
    )

    app.add_url_rule(
        "/api/whatif/predict",
        view_func=whatif_predict_api,
        methods=["POST"]
    )

# Suppress sklearn version warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Direct Supabase PostgreSQL connection
DB_CONN = "postgresql://postgres:RkoGkPLWxh4vavX3@db.trwsfdhxzwzkjandsmvz.supabase.co:5432/postgres"

# Global variables
model = None
scaler = None
feature_columns = None
model_feature_names = None
db_connection_status = False

def load_models():
    """Load the trained model and scaler"""
    global model, scaler, feature_columns, model_feature_names
    
    try:
        model_paths = [
            ('realistic_kochi_metro_rf_model.pkl', 'realistic_kochi_metro_scaler.pkl'),
            ('models/realistic_kochi_metro_rf_model.pkl', 'models/realistic_kochi_metro_scaler.pkl'),
        ]
        
        for model_path, scaler_path in model_paths:
            if os.path.exists(model_path) and os.path.exists(scaler_path):
                model = joblib.load(model_path)
                scaler = joblib.load(scaler_path)
                
                # Get actual feature names from the scaler or model
                if hasattr(scaler, 'feature_names_in_'):
                    model_feature_names = list(scaler.feature_names_in_)
                    logger.info(f"Found {len(model_feature_names)} feature names from scaler")
                else:
                    # Fallback to defined feature columns
                    model_feature_names = [
                        'mtbf', 'brake_wear', 'energy_kwh', 'fault_code', 'mileage_km', 
                        'motor_temp', 'trip_count', 'energy_cost', 'hvac_status', 
                        'door_failures', 'available_bays', 'depot_location', 'depot_position',
                        'battery_current', 'battery_voltage', 'cleaning_status', 'operating_hours',
                        'door_cycle_count', 'incident_reports', 'maintenance_cost', 
                        'compliance_status', 'work_order_status', 'passenger_capacity',
                        'passengers_onboard', 'depot_accessibility', 'fitness_certificate',
                        'pending_maintenance', 'standby_requirement', 'total_operating_cost',
                        'operating_cost_per_km', 'advertising_commitments', 
                        'operating_cost_per_hour', 'stabling_geometry_score',
                        'occupancy_ratio', 'energy_per_km', 'mileage_balancing'
                    ]
                
                feature_columns = model_feature_names.copy()
                
                logger.info(f"Models loaded successfully from {model_path}")
                logger.info(f"Model type: {type(model).__name__}")
                logger.info(f"Expected features: {len(feature_columns)}")
                if hasattr(model, 'n_estimators'):
                    logger.info(f"Random Forest with {model.n_estimators} estimators")
                
                return True
        
        logger.warning("Model files not found - ML predictions not available")
        return False
        
    except Exception as e:
        logger.error(f"Error loading models: {str(e)}")
        return False

def test_db_connection():
    """Test database connection"""
    global db_connection_status
    try:
        supabase = supabase_client.get_supabase()
        if supabase:
            # Simple select to test connection
            supabase.table('train_status').select("train_id").limit(1).execute()
            db_connection_status = True
            logger.info("✅ Database connection successful via Supabase client")
            return True
            
        conn = psycopg2.connect(DB_CONN)
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM train_status")
        count = cur.fetchone()[0]
        cur.close()
        conn.close()
        db_connection_status = True
        logger.info(f"Database connected successfully - {count} trains found (Fallback)")
        return True
    except Exception as e:
        db_connection_status = False
        logger.error(f"Database connection failed: {str(e)}")
        return False

def get_trains_from_db():
    """Fetch all train data from database - NO FALLBACK DATA"""
    try:
        supabase = supabase_client.get_supabase()
        if supabase:
            try:
                response = supabase.table('train_status').select("*").order("train_number").execute()
                if response.data:
                    logger.info(f"Retrieved {len(response.data)} trains from Supabase")
                    return response.data
            except Exception as e:
                logger.error(f"Supabase error in get_trains_from_db: {e}")

        if not db_connection_status:
            logger.error("Database not connected - cannot fetch trains")
            return []
            
        conn = psycopg2.connect(DB_CONN)
        cur = conn.cursor(cursor_factory=RealDictCursor)
        
        cur.execute("SELECT * FROM train_status ORDER BY train_number")
        
        trains = cur.fetchall()
        cur.close()
        conn.close()
        
        if trains:
            logger.info(f"Retrieved {len(trains)} trains from Fallback database")
            return [dict(train) for train in trains]
        else:
            logger.warning("No trains found in database")
            return []
    except Exception as e:
        logger.error(f"Error fetching trains from database: {str(e)}")
        return []

def get_train_by_number(train_number):
    """Fetch specific train data from database - NO FALLBACK DATA"""
    try:
        supabase = supabase_client.get_supabase()
        if supabase:
            try:
                response = supabase.table('train_status').select("*").eq("train_number", train_number).execute()
                if response.data:
                    return response.data[0]
            except Exception as e:
                logger.error(f"Supabase error in get_train_by_number: {e}")

        if not db_connection_status:
            logger.error("Database not connected - cannot fetch train")
            return None
            
        conn = psycopg2.connect(DB_CONN)
        cur = conn.cursor(cursor_factory=RealDictCursor)
        
        cur.execute("SELECT * FROM train_status WHERE train_number = %s", (train_number,))
        
        train = cur.fetchone()
        cur.close()
        conn.close()
        
        if train:
            logger.info(f"Retrieved train {train_number} from database")
            return dict(train)
        else:
            logger.warning(f"Train {train_number} not found in database")
            return None
            
    except Exception as e:
        logger.error(f"Error fetching train {train_number}: {str(e)}")
        return None

def validate_input_data(data):
    """Validate essential input data"""
    required_fields = ['train_number']
    for field in required_fields:
        if field not in data or data[field] is None or data[field] == '':
            raise ValueError(f"Missing required field: {field}")
    return True

def preprocess_data(data):
    """Preprocess input data for ML model"""
    try:
        processed_data = data.copy()
        
        # Label encoders (same as training)
        label_encoders = {
            'fault_code': {'': 0, 'F01': 1, 'F02': 2, 'F03': 3, 'F04': 4, 'F05': 5},
            'depot_location': {'Aluva': 0, 'Edappally': 1, 'Mannuthy': 2},
            'depot_position': {'A1': 0, 'B2': 1, 'C3': 2, 'D4': 3},
            'cleaning_status': {'Clean': 0, 'In-progress': 1, 'Pending': 2},
            'work_order_status': {'Closed': 0, 'Open': 1},
            'depot_accessibility': {'Difficult': 0, 'Easy': 1, 'Moderate': 2},
            'advertising_commitments': {'High': 0, 'Low': 1, 'Medium': 2}
        }
        
        # Apply label encoding
        for column, encoding_map in label_encoders.items():
            if column in processed_data and processed_data[column] is not None:
                processed_data[column] = encoding_map.get(str(processed_data[column]), 0)
        
        # Convert boolean columns
        bool_columns = ['hvac_status', 'compliance_status', 'fitness_certificate',
                       'pending_maintenance', 'standby_requirement']
        for col in bool_columns:
            if col in processed_data:
                if isinstance(processed_data[col], bool):
                    processed_data[col] = int(processed_data[col])
                elif str(processed_data[col]).lower() in ['true', '1', 'yes']:
                    processed_data[col] = 1
                else:
                    processed_data[col] = 0
        
        # Fill missing values with sensible defaults only for ML prediction
        defaults = {
            'trip_count': 25,
            'energy_cost': 500,
            'hvac_status': 1,
            'battery_current': 250,
            'battery_voltage': 700,
            'compliance_status': 1,
            'standby_requirement': 0,
            'door_failures': 0,
            'operating_hours': 8760,
            'door_cycle_count': 50000,
            'incident_reports': 0,
            'maintenance_cost': 50000,
            'total_operating_cost': 200000,
            'operating_cost_per_km': 8.0,
            'operating_cost_per_hour': 25.0,
            'passenger_capacity': 600,
            'passengers_onboard': 300
        }
        
        for key, default_val in defaults.items():
            if key not in processed_data or processed_data[key] is None:
                processed_data[key] = default_val
        
        # Calculate derived features only if missing
        if 'occupancy_ratio' not in processed_data or processed_data['occupancy_ratio'] is None:
            passengers = processed_data.get('passengers_onboard', 300)
            capacity = processed_data.get('passenger_capacity', 600)
            processed_data['occupancy_ratio'] = passengers / max(capacity, 1)
        
        if 'energy_per_km' not in processed_data or processed_data['energy_per_km'] is None:
            energy = processed_data.get('energy_kwh', 500)
            mileage = processed_data.get('mileage_km', 25000)
            processed_data['energy_per_km'] = energy / max(mileage, 1)
        
        if 'mileage_balancing' not in processed_data or processed_data['mileage_balancing'] is None:
            brake_wear = processed_data.get('brake_wear', 30)
            mileage_km = processed_data.get('mileage_km', 25000)
            processed_data['mileage_balancing'] = 1 - (brake_wear/100 * 0.5 + mileage_km/50000 * 0.5)
        
        return processed_data
        
    except Exception as e:
        logger.error(f"Error in preprocessing: {str(e)}")
        raise

def predict_with_model(data):
    """Make prediction using the trained model - ONLY when model is available"""
    if not model or not scaler:
        raise ValueError("ML model not loaded - cannot make ML prediction")
    
    try:
        # Validate input
        validate_input_data(data)
        
        # Preprocess data
        processed_data = preprocess_data(data)
        
        # Create DataFrame with all expected features
        df_data = {}
        
        # Fill all required features
        for feature_name in feature_columns:
            if feature_name in processed_data:
                df_data[feature_name] = processed_data[feature_name]
            else:
                # Provide sensible defaults for missing features
                defaults = {
                    'stabling_geometry_score': 0.7,
                    'advertising_commitments': 1,
                    'depot_location': 0,
                    'depot_position': 0,
                    'cleaning_status': 0,
                    'compliance_status': 1,
                    'depot_accessibility': 1,
                    'work_order_status': 0,
                    'fault_code': 0,
                    'hvac_status': 1,
                    'fitness_certificate': 1,
                    'pending_maintenance': 0,
                    'standby_requirement': 0
                }
                df_data[feature_name] = defaults.get(feature_name, 0)
        
        # Create DataFrame and select features in exact order
        df = pd.DataFrame([df_data])
        X = df[feature_columns]
        
        logger.debug(f"Input shape: {X.shape}")
        
        # Convert to numpy array to avoid sklearn feature name validation issues
        X_array = X.values
        X_scaled = scaler.transform(X_array)
        
        # Make prediction
        prediction = model.predict(X_scaled)[0]
        
        # Get feature importance
        feature_contributions = []
        if hasattr(model, 'feature_importances_'):
            feature_importances = model.feature_importances_
            
            for i, (feature, importance) in enumerate(zip(feature_columns, feature_importances)):
                contribution = {
                    'feature': feature,
                    'importance': float(importance),
                    'value': float(X.iloc[0, i])
                }
                feature_contributions.append(contribution)
            
            feature_contributions.sort(key=lambda x: x['importance'], reverse=True)
        
        # Determine category
        if prediction >= 7.5:
            category, category_class = 'Revenue Service Ready', 'revenue-service'
        elif prediction >= 6.0:
            category, category_class = 'Standby Status', 'standby'
        else:
            category, category_class = 'IBL Maintenance Required', 'maintenance'
        
        logger.info(f"ML prediction successful: {prediction:.2f} -> {category}")
        
        return {
            'prediction': float(prediction),
            'category': category,
            'category_class': category_class,
            'feature_contributions': feature_contributions[:10],
            'prediction_method': 'ML_MODEL'
        }
        
    except Exception as e:
        logger.error(f"Error in predict_with_model: {str(e)}")
        raise
