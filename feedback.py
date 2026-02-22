from flask import Flask, jsonify
from flask_cors import CORS
from datetime import datetime, timedelta
import random
from flask import render_template

def register(app):
    @app.route("/feedback")
    def feedback_page():
        return render_template("feedback.html")

    @app.route('/api/feedback', methods=['GET'])
    def feedback_api_route():
        return get_feedback()

    @app.route('/api/feedback/summary', methods=['GET'])
    def feedback_summary_route():
        return get_feedback_summary()

    @app.route('/api/feedback/recent', methods=['GET'])
    def feedback_recent_route():
        return get_recent_feedback()

    @app.route('/api/health', methods=['GET'])
    def feedback_health_route():
        return feedback_health_check()

# Sample data structure for feedback
FEEDBACK_SOURCES = ['Mobile App', 'Station Kiosks', 'Website', 'Customer Service']

# Sample comments for different ratings
COMMENTS_5_STAR = [
    "Excellent service! Very satisfied with the metro experience.",
    "Great experience, trains are always on time!",
    "Outstanding cleanliness and staff professionalism.",
    "Quick response to my query. Very helpful!",
    "Best public transport system in the city!"
]

COMMENTS_4_STAR = [
    "Clean stations and helpful staff.",
    "Easy ticket booking process.",
    "Good service overall, minor delays sometimes.",
    "Comfortable journey and good facilities.",
    "Nice experience, could improve AC cooling."
]

COMMENTS_3_STAR = [
    "Average experience, needs improvement in peak hours.",
    "Service is okay, but could be better.",
    "Decent transport option for daily commute.",
    "Fair service, sometimes crowded.",
    "Acceptable but room for improvement."
]

COMMENTS_2_STAR = [
    "Frequent delays during peak hours.",
    "Poor crowd management at stations.",
    "Ticket machines often out of service.",
    "Needs better maintenance.",
    "Service quality has declined recently."
]

COMMENTS_1_STAR = [
    "Very disappointed with the service.",
    "Trains are always delayed.",
    "Poor customer service experience.",
    "Unhygienic conditions in some coaches.",
    "Not worth the fare price."
]

def generate_feedback_summary():
    """Generate feedback summary for different sources"""
    summary = []
    
    for source in FEEDBACK_SOURCES:
        rating = round(random.uniform(3.8, 4.8), 1)
        count = random.randint(400, 1500)
        trend_value = random.randint(-5, 10)
        trend = f"+{trend_value}%" if trend_value > 0 else f"{trend_value}%"
        
        summary.append({
            'source': source,
            'rating': rating,
            'count': count,
            'trend': trend
        })
    
    return summary

def get_comment_by_rating(rating):
    """Get appropriate comment based on rating"""
    if rating == 5:
        return random.choice(COMMENTS_5_STAR)
    elif rating == 4:
        return random.choice(COMMENTS_4_STAR)
    elif rating == 3:
        return random.choice(COMMENTS_3_STAR)
    elif rating == 2:
        return random.choice(COMMENTS_2_STAR)
    else:
        return random.choice(COMMENTS_1_STAR)

def generate_recent_feedback():
    """Generate recent feedback entries"""
    recent_feedback = []
    time_options = [
        '2 hours ago', '5 hours ago', '8 hours ago', '12 hours ago',
        '1 day ago', '2 days ago', '3 days ago', '5 days ago'
    ]
    
    for i in range(6):  # Generate 6 recent feedback entries
        rating = random.choice([3, 4, 4, 5, 5])  # Weighted towards positive
        source = random.choice(FEEDBACK_SOURCES)
        comment = get_comment_by_rating(rating)
        time_ago = time_options[i] if i < len(time_options) else f"{i} days ago"
        
        recent_feedback.append({
            'source': source,
            'comment': comment,
            'rating': rating,
            'time_ago': time_ago
        })
    
    return recent_feedback

def calculate_overall_metrics(summary):
    """Calculate overall rating and satisfaction"""
    if not summary:
        return 0, 0
    
    total_rating = sum(item['rating'] * item['count'] for item in summary)
    total_count = sum(item['count'] for item in summary)
    overall_rating = round(total_rating / total_count, 1) if total_count > 0 else 0
    
    # Satisfaction = percentage of 4+ star ratings
    satisfaction = int((overall_rating / 5.0) * 100)
    
    return overall_rating, satisfaction

def get_feedback():
    """Main endpoint to get all feedback data"""
    try:
        summary = generate_feedback_summary()
        recent_feedback = generate_recent_feedback()
        overall_rating, satisfaction = calculate_overall_metrics(summary)
        
        response_data = {
            'summary': summary,
            'recent_feedback': recent_feedback,
            'overall_rating': overall_rating,
            'satisfaction': satisfaction,
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify(response_data), 200
        
    except Exception as e:
        return jsonify({
            'error': str(e),
            'message': 'Failed to fetch feedback data'
        }), 500

def get_feedback_summary():
    """Endpoint to get only feedback summary"""
    try:
        summary = generate_feedback_summary()
        return jsonify({'summary': summary}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def get_recent_feedback():
    """Endpoint to get only recent feedback"""
    try:
        recent_feedback = generate_recent_feedback()
        return jsonify({'recent_feedback': recent_feedback}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def feedback_health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'Kochi Metro Feedback API',
        'timestamp': datetime.now().isoformat()
    }), 200

if __name__ == '__main__':
    print("Starting Kochi Metro Feedback Backend...")
    print("API will be available at: http://localhost:5006")
    print("Endpoints:")
    print("  - GET /api/feedback - Get all feedback data")
    print("  - GET /api/feedback/summary - Get feedback summary only")
    print("  - GET /api/feedback/recent - Get recent feedback only")
    print("  - GET /api/health - Health check")
    app.run(debug=True, host='0.0.0.0', port=5006)