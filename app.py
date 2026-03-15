from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify, send_file
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime, date
import numpy as np
import pandas as pd
import joblib
import re
import json
import random
import string
import os
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.units import inch
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
load_dotenv()
import requests







app = Flask(__name__)
app.secret_key = "supersecretkey"

# AI Configuration

 # ADD THIS LINE

# NEW: File Upload Configuration
UPLOAD_FOLDER = 'static/profile_pics'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024  # 5MB max file size

# Create upload folder if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# ------------------ Database Setup ------------------
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# ------------------ Database Models ------------------
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    dob = db.Column(db.String(20), nullable=False)
    mobile = db.Column(db.String(15), unique=True, nullable=False)
    gmail = db.Column(db.String(100), unique=True, nullable=False)
    password = db.Column(db.String(100), nullable=False)
    profile_pic = db.Column(db.String(200), default='default.png')  # NEW: Profile picture field

class Prediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    disease = db.Column(db.String(120), nullable=False)
    confidence = db.Column(db.Float, nullable=False)
    data_json = db.Column(db.Text, nullable=True)
    download = db.Column(db.String(200), nullable=True)  # PDF download link


# --- Add to models (near other db.Model definitions) ---
class AdminMessage(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, nullable=True)   # from session
    admin_id = db.Column(db.Integer, nullable=True)  # which admin (1..4)
    subject = db.Column(db.String(200))
    message = db.Column(db.Text)
    is_admin_reply = db.Column(db.Boolean, default=False)
    admin_name = db.Column(db.String(120), nullable=True)  # filled when admin replies
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    read = db.Column(db.Boolean, default=False)

# After models, run db.create_all() if not already present
# db.create_all()



with app.app_context():
    db.create_all()

# ------------------ ML Model Loading ------------------
model = joblib.load("disease_model.pkl")
imputer = joblib.load("imputer.pkl")
label_encoder = joblib.load("label_encoder.pkl")
expected_features = imputer.feature_names_in_

# ------------------ Constants ------------------
FEATURES = [
    'Glucose', 'Cholesterol', 'Hemoglobin', 'Platelets', 'WBC', 'RBC',
    'Hematocrit', 'MCV', 'MCH', 'MCHC', 'Insulin', 'BMI',
    'Blood Pressure - Systolic', 'Blood Pressure - Diastolic',
    'Triglycerides', 'HbA1c', 'LDL', 'HDL', 'ALT', 'AST',
    'Heart Rate', 'Creatinine', 'Troponin', 'C-reactive Protein'
]

normal_ranges = {
    "Glucose": (70, 110, "mg/dL"),
    "Cholesterol": (125, 200, "mg/dL"),
    "Hemoglobin": (13.5, 17.5, "g/dL"),
    "Platelets": (150000, 450000, "/µL"),
    "WBC": (4000, 11000, "/µL"),
    "RBC": (4.7, 6.1, "million/µL"),
    "Hematocrit": (38, 50, "%"),
    "MCV": (80, 100, "fL"),
    "MCH": (27, 33, "pg"),
    "MCHC": (32, 36, "g/dL"),
    "Insulin": (2, 25, "µU/mL"),
    "BMI": (18.5, 24.9, ""),
    "Blood Pressure - Systolic": (90, 120, "mmHg"),
    "Blood Pressure - Diastolic": (60, 80, "mmHg"),
    "Triglycerides": (0, 150, "mg/dL"),
    "HbA1c": (0, 5.7, "%"),
    "LDL": (0, 100, "mg/dL"),
    "HDL": (40, 100, "mg/dL"),
    "ALT": (7, 56, "U/L"),
    "AST": (10, 40, "U/L"),
    "Heart Rate": (60, 100, "bpm"),
    "Creatinine": (0.6, 1.3, "mg/dL"),
    "Troponin": (0, 0.04, "ng/mL"),
    "C-reactive Protein": (0, 3, "mg/L")
}

disease_mapping = {
    "Diabetes": ["Glucose", "HbA1c", "Insulin", "BMI"],
    "Anemia": ["Hemoglobin", "Hematocrit", "MCV", "MCH", "MCHC", "RBC"],
    "Heart Disease": ["Cholesterol", "Triglycerides", "LDL", "HDL", "Blood Pressure - Systolic", "Troponin"],
    "Kidney Disease": ["Creatinine", "Blood Pressure - Systolic", "Hemoglobin"],
    "Dengue": ["Platelets", "WBC", "Hematocrit"],
    "Malaria": ["Hemoglobin", "Platelets", "WBC", "RBC"],
    "High Cholesterol": ["Cholesterol", "LDL", "HDL", "Triglycerides"]
}

# ------------------ OTP Management ------------------
otp_storage = {}
email_otp_storage = {}

# NEW: Temporary OTP storage for profile updates
update_otp_storage = {}

def generate_otp():
    return ''.join(random.choices(string.digits, k=4))

# NEW: File upload helper
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# ------------------ Helper Functions ------------------
def calculate_risk_percentage(value, param_name):
    """Calculate risk as percentage (0-100) with label"""
    if param_name not in normal_ranges:
        return {"percentage": 50, "label": "Unknown"}
    
    low, high, unit = normal_ranges[param_name]
    
    if value < low:
        deviation = ((low - value) / low) * 100
        percentage = min(deviation, 100)
        return {"percentage": round(percentage, 1), "label": "Low"}
    elif value > high:
        deviation = ((value - high) / high) * 100
        percentage = min(deviation, 100)
        return {"percentage": round(percentage, 1), "label": "High"}
    else:
        return {"percentage": 0, "label": "Normal"}

def detect_diseases(user_input):
    """Detect multiple diseases based on abnormal parameters"""
    predictions = {}
    
    for disease, params in disease_mapping.items():
        abnormal_params = []
        for param in params:
            if param not in user_input or pd.isna(user_input[param]):
                continue
            
            value = float(user_input[param])
            low, high, unit = normal_ranges[param]
            
            if value < low or value > high:
                risk = calculate_risk_percentage(value, param)
                abnormal_params.append({
                    "parameter": param,
                    "value": value,
                    "normal_range": f"{low} - {high} {unit}".strip(),
                    "risk": risk
                })
        
        if abnormal_params:
            predictions[disease] = {
                "status": "Detected",
                "relevant_params": abnormal_params
            }
    
    if not predictions:
        predictions["Normal"] = {
            "status": "No abnormality",
            "relevant_params": []
        }
    
    return predictions

def generate_pdf_report(user_name, predictions, params, prediction_id):
    """Generate PDF report and return file path"""
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    elements = []
    styles = getSampleStyleSheet()
    
    # Title
    title_style = ParagraphStyle('CustomTitle', parent=styles['Heading1'], fontSize=24, textColor=colors.HexColor('#0a4b78'))
    elements.append(Paragraph("Smart Disease Prediction Report", title_style))
    elements.append(Spacer(1, 0.3*inch))
    
    # Patient Info
    elements.append(Paragraph(f"<b>Patient Name:</b> {user_name}", styles['Normal']))
    elements.append(Paragraph(f"<b>Date:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
    elements.append(Spacer(1, 0.3*inch))
    
    # Predictions
    elements.append(Paragraph("<b>Detected Diseases:</b>", styles['Heading2']))
    for disease, details in predictions.items():
        elements.append(Paragraph(f"• {disease}: {details['status']}", styles['Normal']))
    elements.append(Spacer(1, 0.3*inch))
    
    # Parameters Table
    elements.append(Paragraph("<b>Blood Parameters:</b>", styles['Heading2']))
    table_data = [["Parameter", "Value", "Normal Range", "Risk"]]
    for param, value in params.items():
        if param in normal_ranges and not pd.isna(value):
            low, high, unit = normal_ranges[param]
            risk = calculate_risk_percentage(value, param)
            table_data.append([param, str(value), f"{low}-{high} {unit}", f"{risk['label']} ({risk['percentage']}%)"])
    
    table = Table(table_data, colWidths=[2.5*inch, 1.5*inch, 1.5*inch, 1.5*inch])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#0a4b78')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    elements.append(table)
    
    doc.build(elements)
    
    # Save PDF
    os.makedirs('static/reports', exist_ok=True)
    filename = f"report_{prediction_id}_{datetime.now().strftime('%Y%m%d%H%M%S')}.pdf"
    filepath = os.path.join('static/reports', filename)
    
    with open(filepath, 'wb') as f:
        f.write(buffer.getvalue())
    
    return f"/static/reports/{filename}"

# ------------------ Authentication Routes ------------------
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/send_otp', methods=['POST'])
def send_otp():
    mobile = request.form.get('mobile')
    if not mobile:
        return "Mobile number required", 400
    otp = generate_otp()
    otp_storage[mobile] = otp
    print(f"✅ OTP for {mobile}: {otp}")
    return "OTP sent successfully (check terminal)"

@app.route('/verify_otp', methods=['POST'])
def verify_otp():
    mobile = request.form.get('mobile')
    entered_otp = request.form.get('otp')
    if mobile in otp_storage and otp_storage[mobile] == entered_otp:
        session['otp_verified'] = True
        return "OTP verified"
    return "Invalid OTP"

@app.route('/send_email_otp', methods=['POST'])
def send_email_otp():
    email = request.form.get('email')
    if not email:
        return "Email required", 400
    otp = generate_otp()
    email_otp_storage[email] = otp
    print(f"✅ Email OTP for {email}: {otp}")
    return "Email OTP sent successfully (check terminal)"

@app.route('/verify_email_otp', methods=['POST'])
def verify_email_otp():
    email = request.form.get('email')
    entered_otp = request.form.get('otp')
    if email in email_otp_storage and email_otp_storage[email] == entered_otp:
        session['email_otp_verified'] = True
        return "Email OTP verified"
    return "Invalid Email OTP"

@app.route('/register', methods=['POST'])
def register():
    name = request.form.get('name')
    dob = request.form.get('dob')
    mobile = request.form.get('mobile')
    gmail = request.form.get('gmail')
    password = request.form.get('password')
    
    if not all([name, dob, mobile, gmail, password]):
        return "Missing registration fields", 400
    
    existing_user = User.query.filter((User.mobile == mobile) | (User.gmail == gmail)).first()
    if existing_user:
        return "User already exists. Please login."
    
    new_user = User(name=name, dob=dob, mobile=mobile, gmail=gmail, password=password)
    db.session.add(new_user)
    db.session.commit()
    
    session.pop('otp_verified', None)
    session.pop('email_otp_verified', None)
    flash("Registration successful! Please login now.", "success")
    return redirect(url_for('home'))

@app.route('/login', methods=['POST'])
def login():
    login_value = request.form.get('mobile') or request.form.get('gmail')
    password = request.form.get('password')
    
    user = User.query.filter(
        db.or_(User.mobile == login_value, User.gmail == login_value),
        User.password == password
    ).first()
    
    if user:
        session['user_id'] = user.id
        session['user_name'] = user.name
        session.permanent = True
        print(f"✅ LOGIN SUCCESS - User: {user.name} (ID: {user.id})")
        return "Login successful!", 200
    
    flash("Invalid credentials", "danger")
    return redirect(url_for('home'))

@app.route('/dashboard')
def dashboard():
    if 'user_id' not in session:
        flash("Please login first.", "warning")
        return redirect(url_for('home'))
    
    user = db.session.get(User, session['user_id'])
    if not user:
        session.clear()
        return redirect(url_for('home'))
    
    return render_template('dashboard.html', user_name=user.name)

@app.route('/logout')
def logout():
    session.clear()
    flash("Logged out successfully.", "info")
    return redirect(url_for('home'))





# ------------------ NEW: Profile Management Routes ------------------

@app.route('/get_user_profile')
def get_user_profile():
    """Get current user profile data"""
    if 'user_id' not in session:
        return jsonify({"error": "Not authenticated"}), 401
    
    user = db.session.get(User, session['user_id'])
    if not user:
        return jsonify({"error": "User not found"}), 404
    
    return jsonify({
        "name": user.name,
        "dob": user.dob,
        "mobile": user.mobile,
        "gmail": user.gmail,
        "profile_pic": user.profile_pic if user.profile_pic else 'default.png'
    })

@app.route('/upload_profile_pic', methods=['POST'])
def upload_profile_pic():
    """Upload profile picture"""
    if 'user_id' not in session:
        return jsonify({"error": "Not authenticated"}), 401
    
    if 'profile_pic' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files['profile_pic']
    
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400
    
    if file and allowed_file(file.filename):
        # Secure filename and add user ID
        filename = secure_filename(file.filename)
        unique_filename = f"user_{session['user_id']}_{datetime.now().strftime('%Y%m%d%H%M%S')}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        
        # Save file
        file.save(filepath)
        
        # Update database
        user = db.session.get(User, session['user_id'])
        
        # Delete old profile pic if not default
        if user.profile_pic and user.profile_pic != 'default.png':
            old_path = os.path.join(app.config['UPLOAD_FOLDER'], user.profile_pic)
            if os.path.exists(old_path):
                os.remove(old_path)
        
        user.profile_pic = unique_filename
        db.session.commit()
        
        return jsonify({
            "success": True,
            "profile_pic": unique_filename,
            "url": f"/static/profile_pics/{unique_filename}"
        })
    
    return jsonify({"error": "Invalid file type"}), 400

@app.route('/update_name', methods=['POST'])
def update_name():
    """Update user name (no verification needed)"""
    if 'user_id' not in session:
        return jsonify({"error": "Not authenticated"}), 401
    
    data = request.get_json()
    new_name = data.get('name', '').strip()
    
    if not new_name or len(new_name) < 2:
        return jsonify({"error": "Invalid name"}), 400
    
    user = db.session.get(User, session['user_id'])
    user.name = new_name
    session['user_name'] = new_name
    db.session.commit()
    
    return jsonify({"success": True, "message": "Name updated successfully"})

@app.route('/send_update_mobile_otp', methods=['POST'])
def send_update_mobile_otp():
    """Send OTP to new mobile number for verification"""
    if 'user_id' not in session:
        return jsonify({"error": "Not authenticated"}), 401
    
    data = request.get_json()
    new_mobile = data.get('mobile', '').strip()
    
    if not new_mobile or len(new_mobile) < 10:
        return jsonify({"error": "Invalid mobile number"}), 400
    
    # Check if mobile already exists
    existing = User.query.filter(User.mobile == new_mobile, User.id != session['user_id']).first()
    if existing:
        return jsonify({"error": "Mobile number already in use"}), 400
    
    # Generate and store OTP
    otp = generate_otp()
    update_otp_storage[f"mobile_{session['user_id']}"] = {
        "otp": otp,
        "mobile": new_mobile,
        "timestamp": datetime.now()
    }
    
    print(f"✅ Mobile Update OTP for {new_mobile}: {otp}")
    return jsonify({"success": True, "message": "OTP sent successfully (check terminal)"})

@app.route('/verify_update_mobile_otp', methods=['POST'])
def verify_update_mobile_otp():
    """Verify OTP and update mobile number"""
    if 'user_id' not in session:
        return jsonify({"error": "Not authenticated"}), 401
    
    data = request.get_json()
    entered_otp = data.get('otp', '').strip()
    
    key = f"mobile_{session['user_id']}"
    
    if key not in update_otp_storage:
        return jsonify({"error": "OTP expired or not sent"}), 400
    
    stored_data = update_otp_storage[key]
    
    if stored_data['otp'] != entered_otp:
        return jsonify({"error": "Invalid OTP"}), 400
    
    # Update mobile number
    user = db.session.get(User, session['user_id'])
    user.mobile = stored_data['mobile']
    db.session.commit()
    
    # Clean up OTP storage
    del update_otp_storage[key]
    
    return jsonify({"success": True, "message": "Mobile number updated successfully"})

@app.route('/send_update_email_otp', methods=['POST'])
def send_update_email_otp():
    """Send OTP to new email for verification"""
    if 'user_id' not in session:
        return jsonify({"error": "Not authenticated"}), 401
    
    data = request.get_json()
    new_email = data.get('email', '').strip()
    
    if not new_email or '@' not in new_email:
        return jsonify({"error": "Invalid email address"}), 400
    
    # Check if email already exists
    existing = User.query.filter(User.gmail == new_email, User.id != session['user_id']).first()
    if existing:
        return jsonify({"error": "Email already in use"}), 400
    
    # Generate and store OTP
    otp = generate_otp()
    update_otp_storage[f"email_{session['user_id']}"] = {
        "otp": otp,
        "email": new_email,
        "timestamp": datetime.now()
    }
    
    print(f"✅ Email Update OTP for {new_email}: {otp}")
    return jsonify({"success": True, "message": "OTP sent successfully (check terminal)"})

@app.route('/verify_update_email_otp', methods=['POST'])
def verify_update_email_otp():
    """Verify OTP and update email"""
    if 'user_id' not in session:
        return jsonify({"error": "Not authenticated"}), 401
    
    data = request.get_json()
    entered_otp = data.get('otp', '').strip()
    
    key = f"email_{session['user_id']}"
    
    if key not in update_otp_storage:
        return jsonify({"error": "OTP expired or not sent"}), 400
    
    stored_data = update_otp_storage[key]
    
    if stored_data['otp'] != entered_otp:
        return jsonify({"error": "Invalid OTP"}), 400
    
    # Update email
    user = db.session.get(User, session['user_id'])
    user.gmail = stored_data['email']
    db.session.commit()
    
    # Clean up OTP storage
    del update_otp_storage[key]
    
    return jsonify({"success": True, "message": "Email updated successfully"})

@app.route('/change_password', methods=['POST'])
def change_password():
    """Change user password"""
    if 'user_id' not in session:
        return jsonify({"error": "Not authenticated"}), 401
    
    data = request.get_json()
    current_password = data.get('current_password', '')
    new_password = data.get('new_password', '')
    confirm_password = data.get('confirm_password', '')
    
    if not all([current_password, new_password, confirm_password]):
        return jsonify({"error": "All fields are required"}), 400
    
    if new_password != confirm_password:
        return jsonify({"error": "New passwords do not match"}), 400
    
    if len(new_password) < 6:
        return jsonify({"error": "Password must be at least 6 characters"}), 400
    
    user = db.session.get(User, session['user_id'])
    
    if user.password != current_password:
        return jsonify({"error": "Current password is incorrect"}), 400
    
    user.password = new_password
    db.session.commit()
    
    return jsonify({"success": True, "message": "Password changed successfully"})



# ------------------ Prediction Routes ------------------
@app.route('/predict', methods=['POST'])
def predict():
    if 'user_id' not in session:
        return jsonify({"error": "Not authenticated"}), 401
    
    try:
        # Get input data
        data = request.get_json()
        user_input = {}
        
        for feature in FEATURES:
            val = data.get(feature)
            if val is None or val == '':
                user_input[feature] = np.nan
            else:
                user_input[feature] = float(val)
        
        # Prepare data for model
        row = [user_input.get(col, np.nan) for col in expected_features]
        X_new = pd.DataFrame([row], columns=expected_features)
        X_new = pd.DataFrame(imputer.transform(X_new), columns=expected_features)
        
        # Detect diseases
        predictions = detect_diseases(user_input)
        
        # Calculate overall confidence (average of risk percentages)
        all_risks = []
        for disease_info in predictions.values():
            for param in disease_info.get('relevant_params', []):
                all_risks.append(param['risk']['percentage'])
        confidence = (100 - np.mean(all_risks)) / 100 if all_risks else 1.0
        
        # Determine primary disease
        primary_disease = list(predictions.keys())[0]
        
        # Save to database
        new_prediction = Prediction(
            user_id=session['user_id'],
            disease=primary_disease,
            confidence=confidence,
            data_json=json.dumps({
                "predictions": predictions,
                "params": {k: v for k, v in user_input.items() if not pd.isna(v)}
            })
        )
        db.session.add(new_prediction)
        db.session.commit()
        
        # Generate PDF
        user = db.session.get(User, session['user_id'])
        pdf_path = generate_pdf_report(user.name, predictions, user_input, new_prediction.id)
        new_prediction.download = pdf_path
        db.session.commit()
        
        return jsonify({
            "success": True,
            "prediction_id": new_prediction.id,
            "predictions": predictions,
            "confidence": round(confidence * 100, 1),
            "timestamp": new_prediction.timestamp.isoformat(),
            "pdf_url": pdf_path
        })
    
    except Exception as e:
        print(f"❌ Prediction error: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/get_history')
def get_history():
    if 'user_id' not in session:
        return jsonify({"error": "Not authenticated"}), 401
    
    predictions = Prediction.query.filter_by(user_id=session['user_id']).order_by(Prediction.timestamp.desc()).all()
    
    history = []
    for pred in predictions:
        history.append({
            "id": pred.id,
            "date": pred.timestamp.strftime('%Y-%m-%d'),
            "time": pred.timestamp.strftime('%H:%M:%S'),
            "disease": pred.disease,
            "confidence": round(pred.confidence * 100, 1),
            "download_url": pred.download
        })
    
    return jsonify({"history": history})

@app.route('/generate_ai_suggestions', methods=['GET'])
def generate_ai_suggestions():
    if 'user_id' not in session:
        return jsonify({"error": "Not authenticated"}), 401
    
    try:
        # Get latest prediction
        latest_pred = Prediction.query.filter_by(user_id=session['user_id']).order_by(Prediction.timestamp.desc()).first()
        
        if not latest_pred:
            return jsonify({"error": "No predictions found. Please run a prediction first."}), 400
        
        # Parse prediction data
        pred_data = json.loads(latest_pred.data_json)
        predictions = pred_data.get('predictions', {})
        
        # Extract diseases
        diseases = list(predictions.keys())
        
        # Build prompt for AI
        prompt = build_ai_prompt(diseases, predictions, pred_data.get('params', {}))
        
        # Call OpenAI API using Google API (fallback to Claude/local if needed)
        insights = call_claude_api(prompt)  # CHANGE THIS LINE
        
        # Parse and structure the response
        structured_insights = parse_ai_response(insights, diseases)
        
        return jsonify(structured_insights)
    
    except Exception as e:
        print(f"❌ AI Suggestions error: {str(e)}")
        return jsonify({"error": f"Failed to generate suggestions: {str(e)}"}), 500



def build_ai_prompt(diseases, predictions, params):
    """Build comprehensive prompt for real AI analysis"""
    
    prompt = f"""You are an AI health educator. You provide general, educational health and wellness information based on data, NOT professional medical advice. Always include a disclaimer that this does not replace a real doctor.

ASSESSMENT DATE: {datetime.now().strftime('%B %d, %Y at %H:%M')}

DETECTED HEALTH CONDITIONS:
{chr(10).join([f"• {disease}" for disease in diseases])}

COMPLETE BLOOD PARAMETER ANALYSIS:
"""
    
    # Add all parameters with their values
    prompt += "\nMeasured Parameters:\n"
    for param_name, value in params.items():
        if not pd.isna(value) and param_name in normal_ranges:
            low, high, unit = normal_ranges[param_name]
            status = "NORMAL" if low <= value <= high else "ABNORMAL"
            prompt += f"• {param_name}: {value} {unit} (Normal: {low}-{high} {unit}) [{status}]\n"
    
    prompt += "\n" + "="*70 + "\n"
    prompt += "DETAILED CONDITION ANALYSIS:\n" + "="*70 + "\n"
    
    # Add detailed breakdown for each detected disease
    for disease, details in predictions.items():
        prompt += f"\n🔍 {disease.upper()}:\n"
        prompt += f"   Detection Status: {details.get('status', 'Detected')}\n"
        
        if details.get('relevant_params'):
            prompt += f"   Key Abnormal Parameters:\n"
            for param in details['relevant_params']:
                prompt += f"      - {param['parameter']}: {param['value']} (Normal: {param['normal_range']})\n"
                prompt += f"        Risk Assessment: {param['risk']['label']} - {param['risk']['percentage']}% deviation\n"
    
    prompt += f"""

{"="*70}
REQUIRED ANALYSIS FORMAT:
{"="*70}

Please provide a comprehensive health report in the following EXACT format:

FEEDBACK:
[Write 3-4 detailed paragraphs explaining:
1. Overall health status and what the test results mean
2. The significance of each detected condition
3. How these conditions might be related or affect each other
4. The urgency level and potential risks if left untreated
Be specific, medical, but also understandable to a patient]

PRECAUTIONS:
[List 7-10 specific, actionable precautions as bullet points with dashes]
- [Each precaution should be clear and immediately implementable]
- [Include monitoring activities, warning signs to watch for]
- [Be specific to the detected conditions]

LIFESTYLE:
[List 7-10 lifestyle modifications as bullet points with dashes]
- [Include specific diet recommendations with food examples]
- [Include exercise recommendations with duration and type]
- [Include sleep, stress management, and habit changes]

FOLLOW_UP:
[Write 2-3 paragraphs covering:
1. When to see a doctor (timeframe and urgency)
2. What specific tests to request in the next visit
3. Monitoring frequency and what to track at home
4. Emergency symptoms that require immediate medical attention]

IMPORTANT: Use the exact section headers (FEEDBACK:, PRECAUTIONS:, LIFESTYLE:, FOLLOW_UP:) and format lists with dashes (-)."""
    
    return prompt



def call_claude_api(prompt):
    """
    Call LOCAL LLM (Ollama) for AI insights.
    Uses your local model via http://localhost:11434 (no API key).
    """

    # Extra instruction so model follows our format and does NOT just say
    # "I can't provide medical advice".
    local_prompt = f"""
You are an AI health educator. You ONLY provide general, educational health and wellness information, 
NOT professional medical advice. You MUST always follow the required output format exactly.

Very important rules:
- Do NOT say "I cannot provide medical advice" or similar refusal.
- Instead, give high-level, general guidance and ALWAYS include a disclaimer that
  this does not replace a real doctor.
- You MUST use these section headers exactly: FEEDBACK:, PRECAUTIONS:, LIFESTYLE:, FOLLOW_UP:
- Under PRECAUTIONS and LIFESTYLE, always use bullet points starting with "- ".

Now analyze the following patient data and instructions:

{prompt}
"""

    try:
        print("🤖 Calling local Ollama model for AI insights...")

        payload = {
            "model": "llama3.2",   # or "llama3" if that's what you pulled
            "prompt": local_prompt,
            "stream": False,
            "options": {
                "num_predict": 1024,   # allow longer answer
                "temperature": 0.7
            }
        }

        response = requests.post(
            "http://localhost:11434/api/generate",
            json=payload,
            timeout=180
        )

        response.raise_for_status()
        data = response.json()
        ai_text = data.get("response", "").strip()

        if not ai_text:
            raise ValueError("Empty response from local LLM")

        print("✅ Local AI response received!")
        print(f"📊 Response length: {len(ai_text)} characters")
        return ai_text

    except Exception as e:
        print(f"❌ Local AI (Ollama) error: {str(e)}")
        return generate_fallback_insights(prompt)



def generate_fallback_insights(prompt):
    return """FEEDBACK:
Based on your health assessment, your blood parameters indicate some variations from normal ranges. It's important to monitor these values closely and consult with a healthcare professional for proper diagnosis and treatment recommendations.

PRECAUTIONS:
- Schedule a comprehensive health check-up with your doctor
- Maintain a food and symptoms diary
- Monitor your blood pressure and glucose levels regularly
- Avoid stress and get adequate rest
- Stay hydrated throughout the day

LIFESTYLE:
- Incorporate regular physical exercise (30 minutes daily)
- Follow a balanced diet rich in fruits and vegetables
- Reduce salt and sugar intake
- Quit smoking if applicable
- Maintain a healthy sleep schedule (7-8 hours)

FOLLOW_UP:
Please consult with a qualified healthcare professional for a detailed diagnosis and personalized treatment plan. Consider scheduling follow-up tests within the next 1-2 weeks to monitor your health status."""

def parse_ai_response(response_text, diseases):
    """Parse AI response into structured format with better error handling"""
    
    sections = {
        'diseases': diseases,
        'feedback': '',
        'precautions': [],
        'lifestyle': [],
        'follow_up': ''
    }
    
    try:
        print("📝 Parsing AI response...")
        print(f"Response preview: {response_text[:200]}...")
        
        # Parse FEEDBACK section
        if 'FEEDBACK:' in response_text:
            feedback_start = response_text.find('FEEDBACK:') + len('FEEDBACK:')
            feedback_end = response_text.find('PRECAUTIONS:', feedback_start)
            if feedback_end == -1:
                feedback_end = len(response_text)
            
            feedback_text = response_text[feedback_start:feedback_end].strip()
            sections['feedback'] = feedback_text
            print(f"✓ Feedback parsed: {len(feedback_text)} chars")
        
        # Parse PRECAUTIONS section
        if 'PRECAUTIONS:' in response_text:
            precautions_start = response_text.find('PRECAUTIONS:') + len('PRECAUTIONS:')
            precautions_end = response_text.find('LIFESTYLE:', precautions_start)
            if precautions_end == -1:
                precautions_end = response_text.find('FOLLOW_UP:', precautions_start)
            if precautions_end == -1:
                precautions_end = len(response_text)
            
            precautions_text = response_text[precautions_start:precautions_end].strip()
            
            # Parse bullet points
            lines = precautions_text.split('\n')
            for line in lines:
                line = line.strip()
                if line and (line.startswith('-') or line.startswith('•') or line.startswith('*')):
                    # Remove bullet point and clean up
                    clean_line = line.lstrip('-•* ').strip()
                    if clean_line and len(clean_line) > 10:  # Ignore very short lines
                        sections['precautions'].append(clean_line)
            
            print(f"✓ Precautions parsed: {len(sections['precautions'])} items")
        
        # Parse LIFESTYLE section
        if 'LIFESTYLE:' in response_text:
            lifestyle_start = response_text.find('LIFESTYLE:') + len('LIFESTYLE:')
            lifestyle_end = response_text.find('FOLLOW_UP:', lifestyle_start)
            if lifestyle_end == -1:
                lifestyle_end = len(response_text)
            
            lifestyle_text = response_text[lifestyle_start:lifestyle_end].strip()
            
            # Parse bullet points
            lines = lifestyle_text.split('\n')
            for line in lines:
                line = line.strip()
                if line and (line.startswith('-') or line.startswith('•') or line.startswith('*')):
                    clean_line = line.lstrip('-•* ').strip()
                    if clean_line and len(clean_line) > 10:
                        sections['lifestyle'].append(clean_line)
            
            print(f"✓ Lifestyle parsed: {len(sections['lifestyle'])} items")
        
        # Parse FOLLOW_UP section
        if 'FOLLOW_UP:' in response_text:
            follow_up_start = response_text.find('FOLLOW_UP:') + len('FOLLOW_UP:')
            follow_up_text = response_text[follow_up_start:].strip()
            sections['follow_up'] = follow_up_text
            print(f"✓ Follow-up parsed: {len(follow_up_text)} chars")
        
        # Validation: Ensure we have content
        if not sections['feedback']:
            print("⚠️ No feedback found, using default")
            sections['feedback'] = "Based on your health parameters, please consult with a healthcare professional for detailed analysis."
        
        if not sections['precautions'] or len(sections['precautions']) < 2:
            print("⚠️ Insufficient precautions, adding defaults")
            sections['precautions'] = [
                "Schedule a comprehensive medical check-up within the next 7-14 days",
                "Monitor your symptoms daily and keep a health diary",
                "Take all prescribed medications as directed by your doctor"
            ]
        
        if not sections['lifestyle'] or len(sections['lifestyle']) < 2:
            print("⚠️ Insufficient lifestyle items, adding defaults")
            sections['lifestyle'] = [
                "Maintain a balanced diet with plenty of fruits and vegetables",
                "Exercise for at least 30 minutes daily (walking, swimming, or cycling)",
                "Ensure 7-8 hours of quality sleep each night"
            ]
        
        if not sections['follow_up']:
            print("⚠️ No follow-up found, using default")
            sections['follow_up'] = "Consult with your healthcare provider for personalized medical advice and treatment planning. Schedule regular follow-ups to monitor your condition."
        
        print("✅ AI response parsing complete!")
        
    except Exception as e:
        print(f"❌ Response parsing error: {str(e)}")
        import traceback
        traceback.print_exc()
        
        # Return safe defaults
        sections['feedback'] = "Your health parameters require medical attention. Please consult a healthcare professional."
        sections['precautions'] = ["Consult a doctor soon", "Monitor your symptoms", "Follow medical advice"]
        sections['lifestyle'] = ["Eat healthy", "Exercise regularly", "Get adequate rest"]
        sections['follow_up'] = "Schedule a medical consultation for detailed evaluation."
    
    return sections



@app.route('/get_chart_data')
def get_chart_data():
    if 'user_id' not in session:
        return jsonify({"error": "Not authenticated"}), 401
    
    predictions = Prediction.query.filter_by(user_id=session['user_id']).order_by(Prediction.timestamp).all()
    
    # Count statistics
    total = len(predictions)
    disease_count = sum(1 for p in predictions if p.disease != 'Normal')
    normal_count = sum(1 for p in predictions if p.disease == 'Normal')



    # Prepare data for line chart (predictions over time)
    dates = []
    counts = {}
    
    for pred in predictions:
        date_str = pred.timestamp.strftime('%Y-%m-%d')
        if date_str not in dates:
            dates.append(date_str)
        
        if pred.disease not in counts:
            counts[pred.disease] = []
        counts[pred.disease].append(date_str)
    
    # Build datasets
    datasets = []
    colors_list = ['#0077cc', '#0b5f97', '#99ccff', '#66b3ff', '#FF6384', '#36A2EB', '#FFCE56']
    
    for idx, (disease, disease_dates) in enumerate(counts.items()):
        data = [disease_dates.count(d) for d in dates]
        datasets.append({
            "label": disease,
            "data": data,
            "borderColor": colors_list[idx % len(colors_list)],
            "tension": 0.4
        })
    
    return jsonify({
        "labels": dates,
        "datasets": datasets,
        "stats": {
            "total": total,
            "diseases": disease_count,
            "normal": normal_count
        }
    })

@app.route('/get_stats')
def get_stats():
    if 'user_id' not in session:
        return jsonify({"error": "Not authenticated"}), 401
    
    total = Prediction.query.filter_by(user_id=session['user_id']).count()
    disease_count = Prediction.query.filter(
        Prediction.user_id == session['user_id'],
        Prediction.disease != 'Normal'
    ).count()
    normal_count = Prediction.query.filter_by(user_id=session['user_id'], disease='Normal').count()
    
    return jsonify({
        "total_predictions": total,
        "detected_diseases": disease_count,
        "normal_reports": normal_count
    })

# ------------------ Sensor Routes ------------------
# Enhanced sensor storage with metadata
sensor_data_storage = {
    "timestamp": None,
    "device_id": None,
    "data": {},
    "received": False
}

@app.route('/sensor_predict_from_esp', methods=['POST'])
def sensor_predict_from_esp():
    """Receive and store sensor data from ESP via app1.py"""
    global sensor_data_storage
    try:
        data = request.get_json(force=True)
        
        # Extract device_id if present
        device_id = data.pop('device_id', 'unknown')

        # 🔁 SENSOR → ML FEATURE MAPPING
        SENSOR_KEY_MAPPING = {
            "pulse": "Heart Rate"
        }

        mapped_data = {}

        for key, value in data.items():
            if key in SENSOR_KEY_MAPPING:
                mapped_data[SENSOR_KEY_MAPPING[key]] = value

        
        # Store with metadata
        sensor_data_storage = {
            "timestamp": datetime.now().isoformat(),
            "device_id": device_id,
            "data": data,  # Only blood parameters
            "received": True
        }
        
        print(f"📡 Sensor data stored: {sensor_data_storage}")
        return jsonify({"status": "success"}), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/get_latest_sensor')
def get_latest_sensor():
    """Return latest sensor data for dashboard"""
    if sensor_data_storage.get("received"):
        return jsonify({
            "status": "ok", 
            "timestamp": sensor_data_storage["timestamp"],
            "device_id": sensor_data_storage["device_id"],
            "data": sensor_data_storage["data"]
        }), 200
    return jsonify({"status": "empty", "message": "No sensor data"}), 404

@app.route('/clear_sensor_data', methods=['POST'])
def clear_sensor_data():
    """Clear stored sensor data (optional endpoint)"""
    global sensor_data_storage
    sensor_data_storage = {"received": False, "data": {}}
    return jsonify({"status": "cleared"}), 200



# Add to app.py
@app.route('/get_ai_insights')
def get_ai_insights():
    if 'user_id' not in session:
        return jsonify({"error": "Not authenticated"}), 401
    
    predictions = Prediction.query.filter_by(user_id=session['user_id']).order_by(Prediction.timestamp.desc()).limit(10).all()
    
    insights = []
    
    if not predictions:
        return jsonify({"insights": ["No data yet. Complete a health check to get AI insights!"]})
    
    # Analyze recent trends
    recent_diseases = [p.disease for p in predictions[:5]]
    disease_count = {}
    for d in recent_diseases:
        disease_count[d] = disease_count.get(d, 0) + 1
    
    # Generate insights based on patterns
    if disease_count.get('Diabetes', 0) >= 2:
        insights.append("⚠️ <strong>Diabetes Alert:</strong> Your last few tests show elevated glucose. Consider reducing sugar intake and increasing physical activity.")
        insights.append("💡 <strong>Diet Tip:</strong> Include more whole grains, leafy vegetables, and avoid processed foods.")
        insights.append("🏃 <strong>Exercise:</strong> 30 minutes of walking daily can reduce blood sugar by 15-20%.")
    
    if disease_count.get('Anemia', 0) >= 2:
        insights.append("⚠️ <strong>Anemia Detected:</strong> Low hemoglobin levels. Increase iron-rich foods.")
        insights.append("🥗 <strong>Food Recommendation:</strong> Spinach, lentils, red meat, and vitamin C-rich fruits.")
    
    if disease_count.get('Heart Disease', 0) >= 2:
        insights.append("❤️ <strong>Heart Health Warning:</strong> Cholesterol levels need attention.")
        insights.append("🚫 <strong>Avoid:</strong> Fried foods, trans fats, excessive salt.")
        insights.append("✅ <strong>Recommended:</strong> Omega-3 rich fish, nuts, olive oil.")
    
    if disease_count.get('Normal', 0) >= 3:
        insights.append("✅ <strong>Great Job!</strong> Your health parameters are consistently normal. Keep up the healthy lifestyle!")
        insights.append("🎉 <strong>Streak Bonus:</strong> You've had 3+ normal readings. Maintain this momentum!")
    
    # Trend analysis
    if len(predictions) >= 3:
        latest = json.loads(predictions[0].data_json)
        oldest = json.loads(predictions[-1].data_json)
        
        if 'params' in latest and 'params' in oldest:
            if latest['params'].get('Glucose', 0) > oldest['params'].get('Glucose', 0):
                insights.append("📈 <strong>Trend Alert:</strong> Your glucose levels are increasing over time. Schedule a doctor visit.")
            else:
                insights.append("📉 <strong>Positive Trend:</strong> Your glucose levels are improving! Keep it up.")
    
    if not insights:
        insights.append("ℹ️ <strong>General Health Tip:</strong> Stay hydrated, sleep 7-8 hours, and exercise regularly.")
    
    return jsonify({"insights": insights})


# # --- Add to models (near other db.Model definitions) ---
# class AdminMessage(db.Model):
#     id = db.Column(db.Integer, primary_key=True)
#     user_id = db.Column(db.Integer, nullable=True)   # from session
#     admin_id = db.Column(db.Integer, nullable=True)  # which admin (1..4)
#     subject = db.Column(db.String(200))
#     message = db.Column(db.Text)
#     is_admin_reply = db.Column(db.Boolean, default=False)
#     admin_name = db.Column(db.String(120), nullable=True)  # filled when admin replies
#     timestamp = db.Column(db.DateTime, default=datetime.utcnow)
#     read = db.Column(db.Boolean, default=False)

# # After models, run db.create_all() if not already present
# # db.create_all()

# --- Route: send message (user -> admin) ---
@app.route('/send_admin_message', methods=['POST'])
def send_admin_message():
    if 'user_id' not in session:
        return jsonify({'success': False, 'error': 'Not authenticated'}), 401

    data = request.get_json() or {}
    admin_id = data.get('admin_id')
    subject = data.get('subject', '').strip()
    message = data.get('message', '').strip()

    if not subject or not message:
        return jsonify({'success': False, 'error': 'Subject and message required'}), 400

    msg = AdminMessage(
        user_id = session.get('user_id'),
        admin_id = int(admin_id) if admin_id else None,
        subject = subject,
        message = message
    )
    db.session.add(msg)
    db.session.commit()
    # Optionally: notify admins (email/webhook) — later
    return jsonify({'success': True, 'message_id': msg.id})

# --- Route: get user messages (user sees their messages and replies) ---
@app.route('/get_user_messages', methods=['GET'])
def get_user_messages():
    if 'user_id' not in session:
        return jsonify({'messages': []})

    uid = session['user_id']
    results = AdminMessage.query.filter_by(user_id=uid).order_by(AdminMessage.timestamp.desc()).limit(200).all()
    out = []
    for r in results:
        out.append({
            'id': r.id,
            'admin_id': r.admin_id,
            'subject': r.subject,
            'message': r.message,
            'is_admin_reply': r.is_admin_reply,
            'admin_name': r.admin_name,
            'timestamp': r.timestamp.isoformat()
        })
    return jsonify({'messages': out})

# --- (Optional) Admin API to list all messages for admin dashboard ---
@app.route('/get_admin_messages', methods=['GET'])
def get_admin_messages():
    # protect this route with admin-only check in your session
    messages = AdminMessage.query.order_by(AdminMessage.timestamp.desc()).limit(500).all()
    out = []
    for r in messages:
        out.append({
            'id': r.id,
            'user_id': r.user_id,
            'admin_id': r.admin_id,
            'subject': r.subject,
            'message': r.message,
            'is_admin_reply': r.is_admin_reply,
            'admin_name': r.admin_name,
            'timestamp': r.timestamp.isoformat()
        })
    return jsonify({'messages': out})






# ------------------ Forgot Password Routes ------------------
forgot_password_otp_storage = {}

@app.route('/send_reset_otp', methods=['POST'])
def send_reset_otp():
    """Send OTP to email for password reset"""
    data = request.get_json() or request.form
    identifier = data.get('identifier', '').strip()  # Can be mobile or email
    
    if not identifier:
        return jsonify({"error": "Email or mobile number required"}), 400
    
    # Check if user exists
    user = User.query.filter(
        db.or_(User.mobile == identifier, User.gmail == identifier)
    ).first()
    
    if not user:
        return jsonify({"error": "No account found with this email/mobile"}), 404
    
    # Generate OTP
    otp = generate_otp()
    forgot_password_otp_storage[identifier] = {
        "otp": otp,
        "user_id": user.id,
        "timestamp": datetime.now()
    }
    
    print(f"✅ Password Reset OTP for {identifier}: {otp}")
    return jsonify({"success": True, "message": "OTP sent successfully (check terminal)"})

@app.route('/verify_reset_otp', methods=['POST'])
def verify_reset_otp():
    """Verify OTP for password reset"""
    data = request.get_json() or request.form
    identifier = data.get('identifier', '').strip()
    entered_otp = data.get('otp', '').strip()
    
    if identifier not in forgot_password_otp_storage:
        return jsonify({"error": "OTP expired or not sent"}), 400
    
    stored_data = forgot_password_otp_storage[identifier]
    
    if stored_data['otp'] != entered_otp:
        return jsonify({"error": "Invalid OTP"}), 400
    
    # OTP verified - mark as verified
    forgot_password_otp_storage[identifier]['verified'] = True
    
    return jsonify({"success": True, "message": "OTP verified successfully"})

@app.route('/reset_password', methods=['POST'])
def reset_password():
    """Reset password after OTP verification"""
    data = request.get_json() or request.form
    identifier = data.get('identifier', '').strip()
    new_password = data.get('new_password', '').strip()
    confirm_password = data.get('confirm_password', '').strip()
    
    if not all([identifier, new_password, confirm_password]):
        return jsonify({"error": "All fields are required"}), 400
    
    if new_password != confirm_password:
        return jsonify({"error": "Passwords do not match"}), 400
    
    if len(new_password) < 6:
        return jsonify({"error": "Password must be at least 6 characters"}), 400
    
    # Check if OTP was verified
    if identifier not in forgot_password_otp_storage:
        return jsonify({"error": "Please verify OTP first"}), 400
    
    stored_data = forgot_password_otp_storage[identifier]
    
    if not stored_data.get('verified'):
        return jsonify({"error": "Please verify OTP first"}), 400
    
    # Update password
    user = db.session.get(User, stored_data['user_id'])
    if not user:
        return jsonify({"error": "User not found"}), 404
    
    user.password = new_password
    db.session.commit()
    
    # Clean up OTP storage
    del forgot_password_otp_storage[identifier]
    
    return jsonify({"success": True, "message": "Password reset successfully"})



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)