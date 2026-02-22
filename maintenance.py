from flask import Flask, jsonify
from flask_cors import CORS
from datetime import datetime, timedelta
import random
from flask import render_template

def register(app):

    # =========================
    # UI PAGE
    # =========================
    @app.route("/maintenance")
    def maintenance_page():
        return render_template("maintenance.html")

    # =========================
    # MAIN API
    # =========================
    @app.route("/api/maintenance")
    def maintenance_api():
        return get_urgent_maintenance()

    @app.route("/api/maintenance/stats")
    def maintenance_stats():
        return get_maintenance_stats()

    @app.route("/api/team-availability")
    def team_availability():
        return jsonify(generate_team_availability())

    @app.route("/api/maintenance/by-priority/<priority>")
    def maintenance_by_priority(priority):
        data = generate_maintenance_data()
        filtered = [m for m in data if m["priority"].lower() == priority.lower()]
        return jsonify(filtered)

    @app.route("/api/maintenance/train/<train_id>")
    def maintenance_by_train(train_id):
        data = generate_maintenance_data()
        return jsonify([m for m in data if m["id"] == train_id])


# Train IDs that need maintenance
TRAIN_IDS = ['KM002', 'KM006', 'KM010', 'KM012', 'KM016', 'KM020', 'KM023']

# Maintenance issues with their typical costs and parts
MAINTENANCE_ISSUES = {
    'Brake System Failure': {
        'cost_range': (12000, 18000),
        'parts': ['Brake Pads', 'Hydraulic Fluid', 'Brake Discs'],
        'priority': 'Critical'
    },
    'Engine Overheating': {
        'cost_range': (20000, 30000),
        'parts': ['Radiator', 'Coolant', 'Thermostat'],
        'priority': 'High'
    },
    'Door Mechanism': {
        'cost_range': (6000, 10000),
        'parts': ['Door Motor', 'Sensors', 'Control Board'],
        'priority': 'High'
    },
    'AC Unit Malfunction': {
        'cost_range': (10000, 15000),
        'parts': ['Compressor', 'Filter', 'Refrigerant'],
        'priority': 'Medium'
    },
    'Lighting System': {
        'cost_range': (4000, 7000),
        'parts': ['LED Strips', 'Driver', 'Ballast'],
        'priority': 'Medium'
    },
    'Communication System': {
        'cost_range': (15000, 22000),
        'parts': ['Radio Module', 'Antenna', 'Receiver'],
        'priority': 'Medium'
    },
    'Wheel Bearing': {
        'cost_range': (5000, 9000),
        'parts': ['Bearings', 'Grease', 'Seals'],
        'priority': 'Low'
    },
    'Preventive Check': {
        'cost_range': (2000, 4000),
        'parts': ['Filters', 'Lubricants', 'Inspection Kit'],
        'priority': 'Low'
    },
    'Suspension System': {
        'cost_range': (18000, 25000),
        'parts': ['Shock Absorbers', 'Springs', 'Bushings'],
        'priority': 'High'
    },
    'Electrical Wiring': {
        'cost_range': (8000, 12000),
        'parts': ['Wiring Harness', 'Connectors', 'Fuses'],
        'priority': 'Medium'
    }
}

# Team names
TEAMS = ['Team A', 'Team B', 'Team C', 'Team D', 'Team E', 'Team F']

# Team statuses
TEAM_STATUSES = ['Available', 'Busy', 'On Break']

def generate_maintenance_data():
    """Generate urgent maintenance data"""
    maintenance_list = []
    
    # Select random issues
    num_issues = random.randint(4, 8)
    selected_trains = random.sample(TRAIN_IDS, min(num_issues, len(TRAIN_IDS)))
    selected_issues = random.sample(list(MAINTENANCE_ISSUES.keys()), num_issues)
    
    for i, train_id in enumerate(selected_trains):
        issue_name = selected_issues[i] if i < len(selected_issues) else random.choice(list(MAINTENANCE_ISSUES.keys()))
        issue_data = MAINTENANCE_ISSUES[issue_name]
        
        # Generate scheduled time (between now and 12 hours from now)
        hours_ahead = random.randint(1, 12)
        scheduled_time = datetime.now() + timedelta(hours=hours_ahead)
        
        # Randomly assign technician
        technician = random.choice(TEAMS)
        
        # Calculate cost within range
        cost = random.randint(issue_data['cost_range'][0], issue_data['cost_range'][1])
        
        # Select random parts from the parts list
        num_parts = random.randint(2, len(issue_data['parts']))
        parts = random.sample(issue_data['parts'], num_parts)
        
        maintenance_item = {
            'id': train_id,
            'issue': issue_name,
            'priority': issue_data['priority'],
            'scheduled_time': scheduled_time.isoformat(),
            'technician': technician,
            'cost': cost,
            'parts': parts,
            'eta': f"{hours_ahead}h"
        }
        
        maintenance_list.append(maintenance_item)
    
    # Sort by priority (Critical > High > Medium > Low)
    priority_order = {'Critical': 0, 'High': 1, 'Medium': 2, 'Low': 3}
    maintenance_list.sort(key=lambda x: priority_order.get(x['priority'], 4))
    
    return maintenance_list

def generate_team_availability():
    """Generate team availability status"""
    team_availability = {}
    
    for team in TEAMS:
        # Weight towards Available (60%), Busy (30%), On Break (10%)
        weights = [0.6, 0.3, 0.1]
        status = random.choices(TEAM_STATUSES, weights=weights)[0]
        
        team_availability[team] = {
            'status': status
        }
    
    return team_availability


def get_urgent_maintenance():
    """Main endpoint to get urgent maintenance data"""
    try:
        maintenance_data = generate_maintenance_data()
        team_availability = generate_team_availability()
        
        response_data = {
            'data': maintenance_data,
            'team_availability': team_availability,
            'timestamp': datetime.now().isoformat(),
            'total_issues': len(maintenance_data),
            'critical_count': len([m for m in maintenance_data if m['priority'] == 'Critical']),
            'high_count': len([m for m in maintenance_data if m['priority'] == 'High'])
        }
        
        return jsonify(response_data), 200
        
    except Exception as e:
        return jsonify({
            'error': str(e),
            'message': 'Failed to fetch maintenance data'
        }), 500


def get_maintenance_stats():
    """Endpoint to get maintenance statistics"""
    try:
        maintenance_data = generate_maintenance_data()
        
        stats = {
            'total_issues': len(maintenance_data),
            'critical_issues': len([m for m in maintenance_data if m['priority'] == 'Critical']),
            'high_priority': len([m for m in maintenance_data if m['priority'] == 'High']),
            'medium_priority': len([m for m in maintenance_data if m['priority'] == 'Medium']),
            'low_priority': len([m for m in maintenance_data if m['priority'] == 'Low']),
            'total_cost': sum(m['cost'] for m in maintenance_data),
            'avg_cost': sum(m['cost'] for m in maintenance_data) / len(maintenance_data) if maintenance_data else 0
        }
        
        return jsonify(stats), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


def get_team_availability():
    """Endpoint to get team availability only"""
    try:
        team_availability = generate_team_availability()
        return jsonify({'team_availability': team_availability}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


def get_maintenance_by_priority(priority):
    """Get maintenance issues filtered by priority"""
    try:
        maintenance_data = generate_maintenance_data()
        filtered_data = [m for m in maintenance_data if m['priority'].lower() == priority.lower()]
        
        return jsonify({
            'priority': priority,
            'count': len(filtered_data),
            'data': filtered_data
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def get_maintenance_by_train(train_id):
    """Get maintenance issues for a specific train"""
    try:
        maintenance_data = generate_maintenance_data()
        train_data = [m for m in maintenance_data if m['id'] == train_id]
        
        if train_data:
            return jsonify({
                'train_id': train_id,
                'issues': train_data
            }), 200
        else:
            return jsonify({
                'train_id': train_id,
                'message': 'No maintenance issues found for this train'
            }), 404
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500


def maintenance_health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'Kochi Metro Maintenance API',
        'timestamp': datetime.now().isoformat()
    }), 200

