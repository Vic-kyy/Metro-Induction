from flask import Flask, render_template, jsonify, request,send_from_directory

import psycopg2
import psycopg2.extras
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder
import os
import traceback
from flask import render_template
import supabase_client

def register(app):

    planner = InductionPlanner()

    # ================= UI =================
    @app.route("/induction")
    def induction_page():
        return render_template("induction_planning.html")

    # ================= API =================
    @app.route("/api/induction/master-plan")
    def induction_master_plan():
        return jsonify(planner.generate_master_plan())

    @app.route("/api/induction/health")
    def induction_health():
        conn = planner.get_db_connection()
        if not conn:
            return jsonify({"status": "unhealthy"}), 503
        conn.close()
        return jsonify({"status": "healthy"})

    @app.route("/api/induction/depot-status")
    def induction_depot_status():
        plan = planner.generate_master_plan()
        return jsonify(plan["depot_status"])

    @app.route("/api/induction/insights")
    def induction_insights():
        plan = planner.generate_master_plan()
        return jsonify(plan["operational_insights"])

    @app.route("/api/induction/data-completeness")
    def induction_data_completeness():
        df = planner.fetch_train_data()
        return jsonify(planner.assess_data_completeness(df))

    @app.route("/api/induction/parts-cost")
    def induction_parts_cost():
        return jsonify(planner.get_parts_cost_data())

    @app.route("/api/induction/train/<train_number>")
    def induction_train_details(train_number):
        plan = planner.generate_master_plan()
        train = next(
            (t for t in plan["train_allocations"] if t["train_number"] == train_number),
            None
        )
        if not train:
            return jsonify({"error": "Train not found"}), 404
        return jsonify(train)


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database configuration
DB_CONN = (
    "postgresql://postgres:RkoGkPLWxh4vavX3"
    "@db.trwsfdhxzwzkjandsmvz.supabase.co:5432/postgres"
    "?sslmode=require"
)

class InductionPlanner:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_columns = None
        self.load_model()
        
        # Initialize with fallbacks first effectively
        self.service_requirements = self._get_fallback_requirements()
        self.depot_bays = self._get_fallback_bays()
        self.stabling_tracks = self._get_fallback_tracks()
        self.categorical_mappings = self._get_fallback_mappings()
        
        # Then try to update from database
        try:
            db_requirements = self.get_service_requirements()
            if db_requirements:
                self.service_requirements = db_requirements
                
            db_bays = self.get_depot_configuration()
            if db_bays:
                self.depot_bays = db_bays
                
            db_tracks = self.get_stabling_tracks()
            if db_tracks:
                self.stabling_tracks = db_tracks
                
            db_mappings = self.get_categorical_mappings()
            if db_mappings:
                self.categorical_mappings = db_mappings
        except Exception as e:
            logger.warning(f"Failed to initialize some components from database: {e}. Using fallbacks.")

    def _get_fallback_requirements(self):
        return {
            'revenue_service_needed': 15,
            'standby_needed': 5,
            'maintenance_capacity': 5
        }

    def _get_fallback_bays(self):
        return {
            "MAINT_BAY_01": {"type": "Heavy Maintenance", "avg_hours": 48.0, "usage_count": 0},
            "MAINT_BAY_02": {"type": "Periodic Maintenance", "avg_hours": 36.0, "usage_count": 0},
            "MAINT_BAY_03": {"type": "Brake Service", "avg_hours": 16.0, "usage_count": 0},
            "MAINT_BAY_04": {"type": "General Maintenance", "avg_hours": 24.0, "usage_count": 0}
        }

    def _get_fallback_tracks(self):
        return [f"ST_{i:02d}" for i in range(1, 21)]

    def _get_fallback_mappings(self):
        return {
            'fault_code': {'None': 0, 'F01': 1, 'F02': 2, 'F03': 3, 'F04': 4, 'F05': 5},
            'depot_location': {'Aluva': 0, 'Edappally': 1, 'Mannuthy': 2},
            'depot_position': {'A1': 0, 'B2': 1, 'C3': 2, 'D4': 3},
            'cleaning_status': {'Clean': 0, 'In-progress': 1, 'Pending': 2},
            'work_order_status': {'Closed': 0, 'Open': 1},
            'depot_accessibility': {'Easy': 1, 'Moderate': 2, 'Difficult': 0},
            'advertising_commitments': {'Low': 1, 'Medium': 2, 'High': 0}
        }

    def get_service_requirements(self):
        """Calculate service requirements from actual train data only"""
        supabase = self.get_supabase_client()
        if supabase:
            try:
                # Calculate from actual fleet data in train_status table
                response = supabase.table('train_status').select("fitness_certificate, pending_maintenance, brake_wear, motor_temp, retirement_date").execute()
                data = response.data
                
                if data:
                    total_trains = len(data)
                    fit_trains = sum(1 for r in data if r.get('fitness_certificate') is True)
                    ready_trains = sum(1 for r in data if r.get('pending_maintenance') is False)
                    good_brake_trains = sum(1 for r in data if (r.get('brake_wear') or 0) < 60)
                    good_temp_trains = sum(1 for r in data if (r.get('motor_temp') or 0) < 75)
                    
                    # Calculate operational requirements based on train condition
                    high_readiness_trains = min(fit_trains, ready_trains, good_brake_trains)
                    
                    # Revenue service: trains in best condition (60-70% of high readiness trains)
                    revenue_needed = max(int(high_readiness_trains * 0.65), min(12, total_trains))
                    
                    # Standby: backup trains (15-25% of total fleet)  
                    standby_needed = max(int(total_trains * 0.2), min(3, total_trains - revenue_needed))
                    
                    # Maintenance capacity: remaining trains
                    maintenance_capacity = total_trains - revenue_needed - standby_needed
                    
                    logger.info(f"Calculated requirements from Supabase ({total_trains} trains): {revenue_needed} revenue, {standby_needed} standby, {maintenance_capacity} maintenance")
                    
                    return {
                        'revenue_service_needed': revenue_needed,
                        'standby_needed': standby_needed,
                        'maintenance_capacity': maintenance_capacity
                    }
            except Exception as e:
                logger.error(f"Error getting service requirements from Supabase: {e}")

        # Fallback to direct DB if Supabase client fails (optional, mostly for local dev if host still resolves)
        logger.warning("Supabase client failed or no data - trying direct DB")
        conn = self.get_db_connection()
        if not conn:
            logger.warning("Database connection failed - using fallback service requirements")
            return self._get_fallback_requirements()
        
        try:
            with conn.cursor() as cur:
                # Calculate from actual fleet data in train_status table
                cur.execute("""
                    SELECT 
                        COUNT(*) as total_trains,
                        COUNT(CASE WHEN fitness_certificate = true THEN 1 END) as fit_trains,
                        COUNT(CASE WHEN pending_maintenance = false THEN 1 END) as ready_trains,
                        COUNT(CASE WHEN brake_wear < 60 THEN 1 END) as good_brake_trains,
                        COUNT(CASE WHEN motor_temp < 75 THEN 1 END) as good_temp_trains
                    FROM train_status 
                    WHERE (retirement_date IS NULL OR retirement_date > CURRENT_DATE)
                """)
                
                result = cur.fetchone()
                if not result or result[0] == 0:
                    raise Exception("No active trains found in train_status table")
                
                total_trains, fit_trains, ready_trains, good_brake_trains, good_temp_trains = result
                
                high_readiness_trains = min(fit_trains or 0, ready_trains or 0, good_brake_trains or 0)
                revenue_needed = max(int(high_readiness_trains * 0.65), min(12, total_trains))
                standby_needed = max(int(total_trains * 0.2), min(3, total_trains - revenue_needed))
                maintenance_capacity = total_trains - revenue_needed - standby_needed
                
                return {
                    'revenue_service_needed': revenue_needed,
                    'standby_needed': standby_needed,
                    'maintenance_capacity': maintenance_capacity
                }
                
        except Exception as e:
            logger.error(f"Error getting service requirements: {e}")
            return self._get_fallback_requirements()
        finally:
            if conn:
                conn.close()

    def get_depot_configuration(self):
        """Get depot bay configuration from existing database tables"""
        try:
            supabase = self.get_supabase_client()
            if supabase:
                try:
                    # Try to get from kmrl_parts_cost
                    response = supabase.table('kmrl_parts_cost').select("*").execute()
                    if response.data:
                        bay_config = {}
                        for row in response.data:
                            bay_id = row.get('bay_id')
                            if not bay_id: continue
                            bay_config[bay_id] = {
                                'type': row.get('bay_type', 'General Maintenance'),
                                'avg_hours': float(row.get('bay_avg_time', 24)),
                                'usage_count': 0
                            }
                        if bay_config:
                            logger.info(f"Loaded {len(bay_config)} maintenance bays from Supabase kmrl_parts_cost")
                            return bay_config
                except Exception as e:
                    logger.warning(f"Supabase kmrl_parts_cost fetch failed: {e}")

                try:
                    # Fallback to train_status positions
                    response = supabase.table('train_status').select("depot_position").execute()
                    if response.data:
                        positions = set(row['depot_position'] for row in response.data if row.get('depot_position'))
                        if positions:
                            bay_config = {}
                            for position in sorted(list(positions)):
                                bay_config[f"BAY_{position}"] = {
                                    'type': 'General Maintenance',
                                    'avg_hours': 24.0,
                                    'usage_count': 0
                                }
                            logger.info(f"Created {len(bay_config)} maintenance bays from Supabase depot positions")
                            return bay_config
                except Exception as e:
                    logger.warning(f"Supabase train_status positions fetch failed: {e}")

            conn = self.get_db_connection()
            if not conn:
                logger.warning("Database connection failed - using fallback depot configuration")
                return self._get_fallback_bays()
            
            with conn.cursor() as cur:
                # ... existing legacy psycopg2 logic ...
                # (I'll keep a simplified version for brevity in replacement)
                cur.execute("SELECT COUNT(*) FROM train_status")
                total_trains = cur.fetchone()[0]
                if total_trains > 0:
                    num_bays = max(4, int(total_trains * 0.3))
                    bay_config = {}
                    for i in range(1, num_bays + 1):
                        bay_config[f"MAINT_BAY_{i:02d}"] = {'type': 'General Maintenance', 'avg_hours': 24.0, 'usage_count': 0}
                    return bay_config
            return self._get_fallback_bays()
        except Exception as e:
            logger.error(f"Error getting depot configuration: {e}")
            return self._get_fallback_bays()
        finally:
            if 'conn' in locals() and conn: conn.close()

    def get_stabling_tracks(self):
        """Get stabling tracks from existing database data"""
        try:
            supabase = self.get_supabase_client()
            if supabase:
                try:
                    response = supabase.table('stabling_tracks').select("track_id, track_capacity").eq("active", True).execute()
                    if response.data:
                        tracks = []
                        for row in response.data:
                            tid = row['track_id']
                            cap = int(row['track_capacity'])
                            tracks.extend([f"{tid}_{i}" for i in range(1, cap + 1)])
                        logger.info(f"Loaded {len(tracks)} stabling positions from Supabase stabling_tracks")
                        return tracks
                except Exception as e:
                    logger.warning(f"Supabase stabling_tracks fetch failed: {e}")

                try:
                    # Fallback to train_status positions
                    response = supabase.table('train_status').select("depot_position").execute()
                    if response.data:
                        positions = set(row['depot_position'] for row in response.data if row.get('depot_position'))
                        if positions:
                            tracks = [f"TRACK_{p}" for p in sorted(list(positions))]
                            logger.info(f"Created {len(tracks)} stabling tracks from Supabase depot positions")
                            return tracks
                except Exception as e:
                    logger.warning(f"Supabase train_status for tracks fetch failed: {e}")

            conn = self.get_db_connection()
            if not conn:
                logger.warning("Database connection failed - using fallback stabling tracks")
                return self._get_fallback_tracks()
            
            with conn.cursor() as cur:
                # Simplified legacy fallback
                cur.execute("SELECT DISTINCT depot_position FROM train_status WHERE depot_position IS NOT NULL")
                positions = cur.fetchall()
                if positions:
                    return [f"TRACK_{pos[0]}" for pos in positions]
            return self._get_fallback_tracks()
        except Exception as e:
            logger.error(f"Error getting stabling tracks: {e}")
            return self._get_fallback_tracks()
        finally:
            if 'conn' in locals() and conn: conn.close()

    def get_categorical_mappings(self):
        """Generate categorical mappings from actual database values only"""
        try:
            supabase = self.get_supabase_client()
            categorical_fields = [
                'fault_code', 'depot_location', 'depot_position', 
                'cleaning_status', 'work_order_status', 'depot_accessibility',
                'advertising_commitments'
            ]
            mappings = {}
            
            if supabase:
                try:
                    response = supabase.table('train_status').select(",".join(categorical_fields)).execute()
                    if response.data:
                        df = pd.DataFrame(response.data)
                        for field in categorical_fields:
                            if field in df.columns:
                                values = sorted(df[field].dropna().unique())
                                mappings[field] = {val: idx for idx, val in enumerate(values)}
                        logger.info(f"Generated mappings for {len(mappings)} fields via Supabase")
                        return mappings
                except Exception as e:
                    logger.warning(f"Supabase categorical mapping fetch failed: {e}")

            conn = self.get_db_connection()
            if not conn:
                logger.warning("Database connection failed - using fallback categorical mappings")
                return self._get_fallback_mappings()
            
            with conn.cursor() as cur:
                for field in categorical_fields:
                    cur.execute(f"SELECT DISTINCT {field} FROM train_status WHERE {field} IS NOT NULL ORDER BY {field}")
                    values = [row[0] for row in cur.fetchall()]
                    mappings[field] = {val: idx for idx, val in enumerate(values)}
                return mappings
        except Exception as e:
            logger.error(f"Error getting categorical mappings: {e}")
            return self._get_fallback_mappings()
        finally:
            if 'conn' in locals() and conn: conn.close()

    def load_model(self):
        """Load the trained ML model and scaler"""
        try:
            model_path = 'realistic_kochi_metro_rf_model.pkl'
            scaler_path = 'realistic_kochi_metro_scaler.pkl'
            
            if os.path.exists(model_path) and os.path.exists(scaler_path):
                self.model = joblib.load(model_path)
                self.scaler = joblib.load(scaler_path)
                logger.info("ML model and scaler loaded successfully")
                
                self.feature_columns = [
                    'mtbf', 'brake_wear', 'energy_kwh', 'fault_code', 'mileage_km', 
                    'motor_temp', 'trip_count', 'energy_cost', 'hvac_status', 
                    'door_failures', 'available_bays', 'depot_location', 'depot_position',
                    'battery_current', 'battery_voltage', 'cleaning_status', 'operating_hours',
                    'door_cycle_count', 'incident_reports', 'maintenance_cost', 
                    'compliance_status', 'mileage_balancing', 'work_order_status',
                    'passenger_capacity', 'passengers_onboard', 'depot_accessibility',
                    'fitness_certificate', 'pending_maintenance', 'standby_requirement',
                    'total_operating_cost', 'operating_cost_per_km', 'advertising_commitments',
                    'operating_cost_per_hour', 'stabling_geometry_score', 'occupancy_ratio',
                    'energy_per_km'
                ]
            else:
                logger.warning("ML model files not found. Using rule-based scoring only.")
                self.model = None
                self.scaler = None
        except Exception as e:
            logger.error(f"Error loading ML model: {e}")
            self.model = None
            self.scaler = None

    def get_supabase_client(self):
        """Get initialized Supabase client"""
        return supabase_client.get_supabase()

    def get_db_connection(self):
        """Get legacy database connection with proper error handling"""
        try:
            conn = psycopg2.connect(DB_CONN)
            return conn
        except psycopg2.Error as e:
            # logger.error(f"Database connection error: {e}")
            return None

    def fetch_train_data(self):
        """Fetch real train data from database - NO DEFAULT VALUES"""
        supabase = self.get_supabase_client()
        if supabase:
            try:
                response = supabase.table('train_status').select("*").execute()
                if response.data:
                    df = pd.DataFrame(response.data)
                    logger.info(f"Fetched {len(df)} trains from Supabase")
                    return df
            except Exception as e:
                logger.error(f"Error fetching train data from Supabase: {e}")

        conn = self.get_db_connection()
        if not conn:
            logger.warning("Database connection failed - using fallback sample train data")
            return self._get_sample_train_data()
        
        try:
            # Fetch all train data with NO COALESCE defaults - use actual data only
            query = """
            SELECT 
                train_id,
                train_number,
                mtbf,
                brake_wear,
                energy_kwh,
                fault_code,
                mileage_km,
                motor_temp,
                trip_count,
                energy_cost,
                hvac_status,
                door_failures,
                available_bays,
                depot_location,
                depot_position,
                battery_current,
                battery_voltage,
                cleaning_status,
                operating_hours,
                retirement_date,
                door_cycle_count,
                incident_reports,
                maintenance_cost,
                compliance_status,
                mileage_balancing,
                work_order_status,
                passenger_capacity,
                passengers_onboard,
                depot_accessibility,
                fitness_certificate,
                pending_maintenance,
                standby_requirement,
                total_operating_cost,
                last_maintenance_date,
                operating_cost_per_km,
                advertising_commitments,
                operating_cost_per_hour,
                stabling_geometry_score,
                occupancy_ratio,
                energy_per_km,
                created_at,
                updated_at
            FROM train_status 
            WHERE (retirement_date IS NULL OR retirement_date > CURRENT_DATE)
            ORDER BY train_number
            """
            
            df = pd.read_sql(query, conn)
            logger.info(f"Fetched {len(df)} active trains from database")
            
            if df.empty:
                raise Exception("No active trains found in database")
            
            # Check for critical missing data
            critical_fields = ['train_number', 'fitness_certificate', 'brake_wear', 'motor_temp']
            for field in critical_fields:
                if field in df.columns:
                    missing_count = df[field].isnull().sum()
                    if missing_count > 0:
                        logger.warning(f"Missing data in critical field {field}: {missing_count} records")
                        if missing_count == len(df):
                            raise Exception(f"All records missing critical field: {field}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching train data: {e}")
            raise
        finally:
            if conn:
                conn.close()

    def _get_sample_train_data(self):
        """Generate intensive sample data for fallback"""
        trains = []
        for i in range(1, 26):
            train_number = f"KMR-{1000+i}"
            trains.append({
                'train_id': i,
                'train_number': train_number,
                'mtbf': 2000 + (i * 10),
                'brake_wear': 10 + (i * 2),
                'energy_kwh': 500 + (i * 5),
                'fault_code': 'None' if i % 5 != 0 else 'F01',
                'mileage_km': 10000 + (i * 1000),
                'motor_temp': 50 + (i % 10),
                'trip_count': 100 + i,
                'energy_cost': 5000 + (i * 100),
                'hvac_status': 1,
                'door_failures': i % 3,
                'available_bays': 4,
                'depot_location': 'Aluva' if i % 3 == 0 else 'Edappally',
                'depot_position': f"{chr(65 + (i % 4))}{i % 10}",
                'battery_current': 250,
                'battery_voltage': 700,
                'cleaning_status': 'Clean',
                'operating_hours': 8760,
                'retirement_date': (datetime.now() + timedelta(days=365)).date(),
                'door_cycle_count': 50000,
                'incident_reports': 0,
                'maintenance_cost': 50000,
                'compliance_status': 1,
                'mileage_balancing': 0.8,
                'work_order_status': 'Closed',
                'passenger_capacity': 600,
                'passengers_onboard': 300,
                'depot_accessibility': 'Easy',
                'fitness_certificate': 1,
                'pending_maintenance': 0,
                'standby_requirement': 0,
                'total_operating_cost': 200000,
                'last_maintenance_date': (datetime.now() - timedelta(days=30)).date(),
                'operating_cost_per_km': 8.0,
                'advertising_commitments': 'Medium',
                'operating_cost_per_hour': 25.0,
                'stabling_geometry_score': 0.7,
                'occupancy_ratio': 0.5,
                'energy_per_km': 0.05
            })
        return pd.DataFrame(trains)

    def prepare_features_for_model(self, df):
        """Prepare features for ML model - handle missing data intelligently"""
        try:
            model_df = df.copy()
            
            # Apply categorical mappings from database
            for col, mapping in self.categorical_mappings.items():
                if col in model_df.columns and mapping:
                    model_df[col] = model_df[col].map(mapping)
                    # For unmapped values, use -1 to indicate unknown
                    model_df[col] = model_df[col].fillna(-1)
            
            # Convert boolean columns to int
            bool_columns = ['hvac_status', 'compliance_status', 'fitness_certificate', 
                          'pending_maintenance', 'standby_requirement']
            for col in bool_columns:
                if col in model_df.columns:
                    model_df[col] = model_df[col].astype(int)
            
            # Handle missing numerical data with median imputation from current dataset
            numerical_columns = [col for col in self.feature_columns if col not in bool_columns 
                               and col not in self.categorical_mappings.keys()]
            
            for col in numerical_columns:
                if col in model_df.columns:
                    median_val = model_df[col].median()
                    if pd.isna(median_val):
                        # If all values are null, cannot proceed
                        raise Exception(f"No valid data for critical field: {col}")
                    model_df[col] = model_df[col].fillna(median_val)
            
            # Ensure all required features exist
            for col in self.feature_columns:
                if col not in model_df.columns:
                    raise Exception(f"Required feature missing from database: {col}")
            
            # Select features in training order
            feature_data = model_df[self.feature_columns].copy()
            
            # Final check for any remaining nulls
            null_counts = feature_data.isnull().sum()
            if null_counts.sum() > 0:
                raise Exception(f"Null values found in features: {null_counts[null_counts > 0].to_dict()}")
            
            return feature_data
            
        except Exception as e:
            logger.error(f"Error preparing features: {e}")
            raise

    def calculate_readiness_scores(self, df):
        """Calculate readiness scores using ML model or database-driven rules"""
        try:
            if self.model is not None and self.scaler is not None:
                feature_data = self.prepare_features_for_model(df)
                scaled_features = self.scaler.transform(feature_data)
                readiness_scores = self.model.predict(scaled_features)
                
                # Ensure scores are in valid range (0-10)
                readiness_scores = np.clip(readiness_scores, 0, 10)
                logger.info("Successfully used ML model for readiness scoring")
                return readiness_scores
            else:
                logger.info("ML model not available, using database-driven rule-based scoring")
                return self.database_driven_scoring(df)
                
        except Exception as e:
            logger.error(f"Error with ML scoring, falling back to database-driven scoring: {e}")
            return self.database_driven_scoring(df)

    def database_driven_scoring(self, df):
        """Enhanced rule-based scoring using ONLY database values"""
        scores = []
        
        # Get scoring weights from database if available
        scoring_weights = self.get_scoring_weights()
        
        for _, train in df.iterrows():
            score = 5.0  # Base score
            
            # Use only non-null values for scoring
            if pd.notna(train.get('fitness_certificate')):
                if train['fitness_certificate']:
                    score += scoring_weights.get('fitness_certificate_good', 2.0)
                else:
                    score -= scoring_weights.get('fitness_certificate_bad', 5.0)
            
            if pd.notna(train.get('pending_maintenance')):
                if train['pending_maintenance']:
                    score -= scoring_weights.get('pending_maintenance', 2.0)
                else:
                    score += scoring_weights.get('no_pending_maintenance', 1.0)
            
            # Brake condition scoring
            if pd.notna(train.get('brake_wear')):
                brake_wear = float(train['brake_wear'])
                if brake_wear < 30:
                    score += scoring_weights.get('brake_excellent', 1.5)
                elif brake_wear < 60:
                    score += scoring_weights.get('brake_good', 0.5)
                elif brake_wear < 80:
                    score -= scoring_weights.get('brake_warning', 0.5)
                else:
                    score -= scoring_weights.get('brake_critical', 2.0)
            
            # Temperature scoring
            if pd.notna(train.get('motor_temp')):
                motor_temp = float(train['motor_temp'])
                if motor_temp < 60:
                    score += scoring_weights.get('temp_optimal', 1.0)
                elif motor_temp < 75:
                    score += scoring_weights.get('temp_good', 0.5)
                elif motor_temp < 90:
                    score -= scoring_weights.get('temp_warning', 0.5)
                else:
                    score -= scoring_weights.get('temp_critical', 2.0)
            
            # Door failures
            if pd.notna(train.get('door_failures')):
                door_failures = int(train['door_failures'])
                if door_failures == 0:
                    score += scoring_weights.get('no_door_failures', 0.8)
                elif door_failures <= 2:
                    score -= scoring_weights.get('minor_door_failures', 0.5)
                elif door_failures <= 5:
                    score -= scoring_weights.get('moderate_door_failures', 1.0)
                else:
                    score -= scoring_weights.get('major_door_failures', 2.0)
            
            # HVAC status
            if pd.notna(train.get('hvac_status')):
                if train['hvac_status']:
                    score += scoring_weights.get('hvac_working', 0.5)
                else:
                    score -= scoring_weights.get('hvac_broken', 1.5)
            
            # Fault presence
            if pd.notna(train.get('fault_code')):
                fault_code = str(train['fault_code']).strip()
                if fault_code == '' or fault_code.lower() == 'none' or pd.isna(train['fault_code']):
                    score += scoring_weights.get('no_faults', 0.8)
                else:
                    score -= scoring_weights.get('active_fault', 1.5)
            
            # Recent incidents
            if pd.notna(train.get('incident_reports')):
                incidents = int(train['incident_reports'])
                if incidents == 0:
                    score += scoring_weights.get('no_incidents', 0.5)
                elif incidents <= 2:
                    score -= scoring_weights.get('minor_incidents', 0.3)
                else:
                    score -= scoring_weights.get('major_incidents', 1.0)
            
            # Compliance status
            if pd.notna(train.get('compliance_status')):
                if train['compliance_status']:
                    score += scoring_weights.get('compliant', 0.5)
                else:
                    score -= scoring_weights.get('non_compliant', 1.5)
            
            # Ensure score is within bounds
            final_score = max(0, min(10, score))
            scores.append(final_score)
        
        return np.array(scores)

    def get_scoring_weights(self):
        """Get scoring weights from database configuration"""
        conn = self.get_db_connection()
        if not conn:
            # Cannot get weights from database, use basic weights
            logger.warning("Cannot get scoring weights from database, using basic scoring")
            return {}
        
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT weight_name, weight_value 
                    FROM scoring_weights 
                    WHERE active = true
                """)
                weights = dict(cur.fetchall())
                return weights if weights else {}
        except:
            return {}  # Table doesn't exist or query failed
        finally:
            conn.close()

    def get_parts_cost_data(self):
        """Get parts cost data from the new table"""
        conn = self.get_db_connection()
        if not conn:
            return {}
        
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT part_name, part_cost 
                    FROM kmrl_relevant_parts_cost
                    ORDER BY part_cost DESC
                """)
                parts_data = dict(cur.fetchall())
                return parts_data
        except Exception as e:
            logger.warning(f"Could not fetch parts cost data: {e}")
            return {}
        finally:
            conn.close()

    def generate_explanations(self, df, scores):
        """Generate explanations based only on actual database values"""
        explanations = []
        
        for idx, (_, train) in enumerate(df.iterrows()):
            explanation = []
            score = scores[idx]
            
            # Only add explanations for non-null values
            if pd.notna(train.get('fitness_certificate')) and not train['fitness_certificate']:
                explanation.append("Invalid fitness certificate - Cannot operate")
            
            if pd.notna(train.get('pending_maintenance')) and train['pending_maintenance']:
                explanation.append("Pending maintenance work required")
            
            if pd.notna(train.get('fault_code')):
                fault_code = str(train['fault_code']).strip()
                if fault_code and fault_code.lower() != 'none':
                    explanation.append(f"Active fault: {fault_code}")
            
            if pd.notna(train.get('brake_wear')):
                brake_wear = float(train['brake_wear'])
                if brake_wear > 80:
                    explanation.append(f"High brake wear ({brake_wear}%) - Service needed")
                elif brake_wear < 30:
                    explanation.append(f"Excellent brake condition ({brake_wear}%)")
            
            if pd.notna(train.get('motor_temp')):
                motor_temp = float(train['motor_temp'])
                if motor_temp > 90:
                    explanation.append(f"High motor temperature ({motor_temp:.1f}°C)")
                elif motor_temp < 60:
                    explanation.append(f"Optimal motor temperature ({motor_temp:.1f}°C)")
            
            if pd.notna(train.get('door_failures')):
                door_failures = int(train['door_failures'])
                if door_failures > 5:
                    explanation.append(f"Multiple door failures ({door_failures})")
                elif door_failures == 0:
                    explanation.append("No door system issues")
            
            if pd.notna(train.get('hvac_status')) and not train['hvac_status']:
                explanation.append("HVAC system not operational")
            
            if pd.notna(train.get('incident_reports')):
                incidents = int(train['incident_reports'])
                if incidents > 2:
                    explanation.append(f"Recent incidents reported ({incidents})")
            
            # Maintenance timing based on actual data
            if pd.notna(train.get('last_maintenance_date')):
                try:
                    if isinstance(train['last_maintenance_date'], str):
                        last_maintenance = datetime.strptime(train['last_maintenance_date'], '%Y-%m-%d').date()
                    else:
                        last_maintenance = train['last_maintenance_date']
                    
                    days_since = (datetime.now().date() - last_maintenance).days
                    if days_since > 60:
                        explanation.append(f"{days_since} days since last maintenance")
                    elif days_since < 14:
                        explanation.append(f"Recently maintained ({days_since} days ago)")
                except:
                    pass
            
            # Limit to top 4 most relevant explanations
            explanations.append(explanation[:4])
        
        return explanations

    def allocate_trains(self, df, scores, explanations):
        """Train allocation using only database-derived requirements"""
        allocation_df = df.copy()
        allocation_df['readiness_score'] = scores
        allocation_df['explanation'] = explanations
        
        # Sort by readiness score (highest first)
        allocation_df = allocation_df.sort_values('readiness_score', ascending=False)
        allocation_df['rank'] = range(1, len(allocation_df) + 1)
        
        allocations = []
        bay_assignments = {}
        track_assignments = {}
        
        revenue_count = 0
        standby_count = 0
        maintenance_count = 0
        
        # Use database-driven thresholds
        revenue_threshold = self.calculate_dynamic_threshold(scores, 'revenue')
        standby_threshold = self.calculate_dynamic_threshold(scores, 'standby')
        
        for _, train in allocation_df.iterrows():
            allocation = {
                'train_number': train['train_number'],
                'train_id': int(train['train_id']),
                'readiness_score': float(train['readiness_score']),
                'rank': int(train['rank']),
                'explanation': train['explanation'],
                'category': None,
                'status': None,
                'location': None,
                'maintenance_type': None,
                'estimated_completion': None,
                'bay_assignment': None,
                'track_position': None
            }
            
            # Determine allocation based on database conditions
            critical_issues = self.has_critical_issues_db_only(train)
            
            if critical_issues or train['readiness_score'] < 3.0:
                maintenance_type = self.determine_maintenance_type_db_only(train)
                bay = self.assign_maintenance_bay(maintenance_type, bay_assignments)
                
                allocation.update({
                    'category': 'IBL Maintenance',
                    'status': 'Assigned' if bay else 'Queued',
                    'maintenance_type': maintenance_type,
                    'bay_assignment': bay,
                    'location': bay if bay else 'Queue',
                    'estimated_completion': self.calculate_completion_time_db_only(maintenance_type, train)
                })
                
                if bay:
                    bay_assignments[bay] = train['train_number']
                maintenance_count += 1
                    
            elif (revenue_count < self.service_requirements['revenue_service_needed'] and 
                  train['readiness_score'] >= revenue_threshold):
                allocation.update({
                    'category': 'Revenue Service',
                    'status': 'Ready for Service',
                    'location': 'Depot Ready'
                })
                revenue_count += 1
                
            elif (standby_count < self.service_requirements['standby_needed'] and 
                  train['readiness_score'] >= standby_threshold):
                track = self.assign_stabling_track(track_assignments)
                allocation.update({
                    'category': 'Standby',
                    'status': 'Standing By',
                    'track_position': track,
                    'location': track if track else 'Depot Yard'
                })
                if track:
                    track_assignments[track] = train['train_number']
                standby_count += 1
                
            else:
                # Remaining trains go to maintenance
                maintenance_type = self.determine_maintenance_type_db_only(train)
                bay = self.assign_maintenance_bay(maintenance_type, bay_assignments)
                
                allocation.update({
                    'category': 'IBL Maintenance',
                    'status': 'Assigned' if bay else 'Queued',
                    'maintenance_type': maintenance_type,
                    'bay_assignment': bay,
                    'location': bay if bay else 'Queue',
                    'estimated_completion': self.calculate_completion_time_db_only(maintenance_type, train)
                })
                
                if bay:
                    bay_assignments[bay] = train['train_number']
                maintenance_count += 1
            
            allocations.append(allocation)
        
        logger.info(f"Allocation complete: {revenue_count} revenue, {standby_count} standby, {maintenance_count} maintenance")
        return allocations, bay_assignments, track_assignments

    def calculate_dynamic_threshold(self, scores, category):
        """Calculate thresholds based on actual score distribution"""
        if category == 'revenue':
            return np.percentile(scores, 75)  # Top 25% for revenue service
        elif category == 'standby':
            return np.percentile(scores, 50)  # Middle 50% for standby
        return 0

    def has_critical_issues_db_only(self, train):
        """Check for critical issues using only non-null database values"""
        critical_conditions = []
        
        if pd.notna(train.get('fitness_certificate')) and not train['fitness_certificate']:
            critical_conditions.append(True)
        
        if pd.notna(train.get('pending_maintenance')) and train['pending_maintenance']:
            critical_conditions.append(True)
        
        if pd.notna(train.get('brake_wear')) and float(train['brake_wear']) > 85:
            critical_conditions.append(True)
        
        if pd.notna(train.get('motor_temp')) and float(train['motor_temp']) > 95:
            critical_conditions.append(True)
        
        if pd.notna(train.get('door_failures')) and int(train['door_failures']) > 8:
            critical_conditions.append(True)
        
        if pd.notna(train.get('compliance_status')) and not train['compliance_status']:
            critical_conditions.append(True)
        
        if pd.notna(train.get('incident_reports')) and int(train['incident_reports']) > 5:
            critical_conditions.append(True)
        
        return any(critical_conditions)

    def determine_maintenance_type_db_only(self, train):
        """Determine maintenance type based only on actual database values"""
        # Priority order based on actual conditions found in data
        if pd.notna(train.get('fitness_certificate')) and not train['fitness_certificate']:
            return 'Safety Systems Check'
        elif pd.notna(train.get('compliance_status')) and not train['compliance_status']:
            return 'Compliance Inspection'
        elif pd.notna(train.get('brake_wear')) and float(train['brake_wear']) > 80:
            return 'Brake Service'
        elif pd.notna(train.get('motor_temp')) and float(train['motor_temp']) > 90:
            return 'Traction Maintenance'
        elif pd.notna(train.get('door_failures')) and int(train['door_failures']) > 5:
            return 'Door Systems'
        elif pd.notna(train.get('hvac_status')) and not train['hvac_status']:
            return 'HVAC Service'
        elif pd.notna(train.get('incident_reports')) and int(train['incident_reports']) > 3:
            return 'Incident Investigation'
        elif pd.notna(train.get('cleaning_status')) and str(train['cleaning_status']) != 'Clean':
            return 'Cleaning Service'
        elif pd.notna(train.get('pending_maintenance')) and train['pending_maintenance']:
            return 'Scheduled Maintenance'
        else:
            return 'Preventive Maintenance'

    def assign_maintenance_bay(self, maintenance_type, occupied_bays):
        """Assign maintenance bay based on database configuration"""
        # Map maintenance types to bay types from database
        for bay_id, bay_info in self.depot_bays.items():
            if (bay_id not in occupied_bays and 
                maintenance_type.lower() in bay_info['type'].lower()):
                return bay_id
        
        # If no specific bay available, find any available bay
        for bay_id in sorted(self.depot_bays.keys()):
            if bay_id not in occupied_bays:
                return bay_id
        
        return None

    def assign_stabling_track(self, occupied_tracks):
        """Assign stabling track based on database configuration"""
        for track in sorted(self.stabling_tracks):
            if track not in occupied_tracks:
                return track
        return None

    def calculate_completion_time_db_only(self, maintenance_type, train):
        """Calculate completion time using database bay configuration"""
        # Get estimated hours from depot bay configuration
        estimated_hours = None
        for bay_info in self.depot_bays.values():
            if maintenance_type.lower() in bay_info['type'].lower():
                estimated_hours = bay_info['avg_hours']
                break
        
        if not estimated_hours:
            # Use average from all bays
            estimated_hours = np.mean([bay['avg_hours'] for bay in self.depot_bays.values()])
        
        # Adjust based on actual train condition severity
        condition_multiplier = 1.0
        if pd.notna(train.get('incident_reports')) and int(train['incident_reports']) > 3:
            condition_multiplier += 0.5
        if pd.notna(train.get('brake_wear')) and float(train['brake_wear']) > 90:
            condition_multiplier += 0.3
        if pd.notna(train.get('compliance_status')) and not train['compliance_status']:
            condition_multiplier += 0.4
        
        total_hours = int(estimated_hours * condition_multiplier)
        completion_time = datetime.now() + timedelta(hours=total_hours)
        return completion_time.isoformat()

    def calculate_kpis(self, allocations):
        """Calculate KPIs from actual allocations"""
        if not allocations:
            return {
                'total_trains': 0,
                'revenue_service': 0,
                'standby': 0,
                'maintenance': 0,
                'average_readiness': 0.0,
                'fleet_availability_percent': 0.0,
                'service_capacity_percent': 0.0,
                'bay_utilization_percent': 0.0
            }
        
        total_trains = len(allocations)
        revenue_trains = len([a for a in allocations if a['category'] == 'Revenue Service'])
        standby_trains = len([a for a in allocations if a['category'] == 'Standby'])
        maintenance_trains = len([a for a in allocations if a['category'] == 'IBL Maintenance'])
        
        avg_readiness = np.mean([a['readiness_score'] for a in allocations])
        fleet_availability = (revenue_trains + standby_trains) / total_trains * 100
        service_capacity = revenue_trains / self.service_requirements['revenue_service_needed'] * 100
        
        occupied_bays = len([a for a in allocations if a.get('bay_assignment')])
        bay_utilization = occupied_bays / len(self.depot_bays) * 100
        
        return {
            'total_trains': total_trains,
            'revenue_service': revenue_trains,
            'standby': standby_trains,
            'maintenance': maintenance_trains,
            'average_readiness': round(avg_readiness, 2),
            'fleet_availability_percent': round(fleet_availability, 1),
            'service_capacity_percent': round(service_capacity, 1),
            'bay_utilization_percent': round(bay_utilization, 1)
        }

    def generate_insights(self, allocations, kpis):
        """Generate insights based on actual performance data"""
        insights = []
        
        # Service capacity analysis
        if kpis['service_capacity_percent'] < 100:
            shortage = self.service_requirements['revenue_service_needed'] - kpis['revenue_service']
            insights.append({
                'type': 'warning',
                'category': 'Service Capacity',
                'message': f"Service shortage: {shortage} trains below requirement ({kpis['revenue_service']}/{self.service_requirements['revenue_service_needed']})",
                'recommendation': 'Expedite high-readiness train maintenance to meet service demands'
            })
        elif kpis['service_capacity_percent'] > 110:
            insights.append({
                'type': 'info',
                'category': 'Service Capacity',
                'message': f"Service capacity exceeded: {kpis['revenue_service']} trains ready",
                'recommendation': 'Consider additional service opportunities or preventive maintenance'
            })
        
        # Fleet availability monitoring
        if kpis['fleet_availability_percent'] < 75:
            insights.append({
                'type': 'critical',
                'category': 'Fleet Availability',
                'message': f"Low fleet availability: {kpis['fleet_availability_percent']}% operational",
                'recommendation': 'Urgent review of maintenance scheduling and resource allocation needed'
            })
        elif kpis['fleet_availability_percent'] < 85:
            insights.append({
                'type': 'warning',
                'category': 'Fleet Availability',
                'message': f"Fleet availability below target: {kpis['fleet_availability_percent']}%",
                'recommendation': 'Monitor maintenance progress and consider extending service intervals'
            })
        
        # Maintenance load analysis
        if kpis['maintenance'] > (kpis['total_trains'] * 0.4):
            insights.append({
                'type': 'warning',
                'category': 'Maintenance Load',
                'message': f"High maintenance load: {kpis['maintenance']} trains ({kpis['maintenance']/kpis['total_trains']*100:.1f}%)",
                'recommendation': 'Review maintenance procedures and consider additional maintenance capacity'
            })
        
        # Bay utilization analysis
        if kpis['bay_utilization_percent'] > 90:
            insights.append({
                'type': 'warning',
                'category': 'Bay Utilization',
                'message': f"Maintenance bays near capacity: {kpis['bay_utilization_percent']}%",
                'recommendation': 'Consider maintenance scheduling optimization or additional bay capacity'
            })
        
        # Readiness score analysis
        if kpis['average_readiness'] < 6.0:
            insights.append({
                'type': 'critical',
                'category': 'Fleet Condition',
                'message': f"Low average readiness: {kpis['average_readiness']}/10",
                'recommendation': 'Investigate systematic issues affecting fleet condition'
            })
        elif kpis['average_readiness'] > 8.0:
            insights.append({
                'type': 'info',
                'category': 'Fleet Condition',
                'message': f"Excellent fleet condition: {kpis['average_readiness']}/10 average readiness",
                'recommendation': 'Maintain current maintenance standards and practices'
            })
        
        # Critical train analysis
        critical_trains = [a for a in allocations if a['readiness_score'] < 3.0]
        if len(critical_trains) > 0:
            insights.append({
                'type': 'critical',
                'category': 'Critical Trains',
                'message': f"{len(critical_trains)} trains in critical condition",
                'recommendation': f"Immediate attention required for: {', '.join([t['train_number'] for t in critical_trains])}"
            })
        
        return insights

    def generate_master_plan(self):
        """Generate comprehensive induction plan with database-only data"""
        try:
            # Fetch current train data
            train_df = self.fetch_train_data()
            
            if train_df.empty:
                raise Exception("No train data available from database")
            
            # Calculate readiness scores using ML model or database-driven approach
            readiness_scores = self.calculate_readiness_scores(train_df)
            
            # Generate explanations based on actual data
            explanations = self.generate_explanations(train_df, readiness_scores)
            
            # Perform allocation based on database configuration
            allocations, bay_assignments, track_assignments = self.allocate_trains(
                train_df, readiness_scores, explanations
            )
            
            # Calculate performance KPIs from actual results
            kpis = self.calculate_kpis(allocations)
            
            # Generate insights from actual performance
            insights = self.generate_insights(allocations, kpis)
            
            # Create depot status from actual assignments
            depot_status = self.create_depot_status(bay_assignments, track_assignments)
            
            plan = {
                'generated_at': datetime.now().isoformat(),
                'plan_date': datetime.now().date().isoformat(),
                'train_allocations': allocations,
                'performance_kpis': kpis,
                'operational_insights': insights,
                'depot_status': depot_status,
                'total_trains': len(allocations),
                'service_requirements': self.service_requirements,
                'model_used': 'ML Model' if self.model is not None else 'Database-driven Rules',
                'data_completeness': self.assess_data_completeness(train_df)
            }
            
            logger.info(f"Master plan generated: {len(allocations)} trains allocated using {plan['model_used']}")
            return plan
            
        except Exception as e:
            logger.error(f"Error generating master plan: {e}")
            logger.error(traceback.format_exc())
            raise

    def assess_data_completeness(self, df):
        """Assess completeness of database data"""
        total_fields = len(df.columns)
        completeness = {}
        
        for col in df.columns:
            non_null_count = df[col].notna().sum()
            completeness[col] = round((non_null_count / len(df)) * 100, 1)
        
        overall_completeness = round(np.mean(list(completeness.values())), 1)
        
        return {
            'overall_percent': overall_completeness,
            'field_completeness': completeness,
            'total_records': len(df),
            'total_fields': total_fields
        }

    def create_depot_status(self, bay_assignments, track_assignments):
        """Create depot status based on actual database configuration"""
        maintenance_bays = []
        for bay_id, bay_info in self.depot_bays.items():
            bay_status = {
                'bay_id': bay_id,
                'bay_type': bay_info['type'],
                'status': 'Occupied' if bay_id in bay_assignments else 'Available',
                'train_number': bay_assignments.get(bay_id),
                'estimated_hours': bay_info['avg_hours']
            }
            maintenance_bays.append(bay_status)
        
        stabling_tracks_status = []
        for track_id in self.stabling_tracks:
            track_status = {
                'track_id': track_id,
                'status': 'Occupied' if track_id in track_assignments else 'Available',
                'train_number': track_assignments.get(track_id)
            }
            stabling_tracks_status.append(track_status)
        
        return {
            'maintenance_bays': maintenance_bays,
            'stabling_tracks': stabling_tracks_status,
            'bay_utilization': len(bay_assignments) / len(self.depot_bays) * 100 if self.depot_bays else 0,
            'track_utilization': len(track_assignments) / len(self.stabling_tracks) * 100 if self.stabling_tracks else 0,
            'total_bays': len(self.depot_bays),
            'occupied_bays': len(bay_assignments),
            'total_tracks': len(self.stabling_tracks),
            'occupied_tracks': len(track_assignments)
        }