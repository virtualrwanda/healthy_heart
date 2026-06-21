from flask import Flask, request, jsonify, render_template, redirect, url_for, flash, session
from functools import wraps
import numpy as np
import joblib
import sqlite3
import datetime
import hashlib
import secrets
from dataclasses import dataclass
from typing import List, Dict, Optional

app = Flask(__name__)
app.secret_key = secrets.token_hex(32)

# Load ML models
try:
    tachycardia_model = joblib.load('model/tachycardia_model.joblib')
    hypertrophy_model = joblib.load('model/hypertrophy_model.joblib')
    cholesterol_model = joblib.load('model/cholesterol_model.joblib')
    MODELS_LOADED = True
    print("✅ ML models loaded successfully")
except Exception as e:
    print(f"⚠️ Warning: Could not load ML models: {e}")
    MODELS_LOADED = False

DB_PATH = 'heart_monitor.db'

# ============ DATABASE SETUP ============
def init_db():
    """Initialize database with all required tables"""
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        
        # Users table (authentication)
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            full_name TEXT NOT NULL,
            role TEXT NOT NULL CHECK(role IN ('admin', 'doctor', 'caregiver')),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        
        # Patients table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS patients (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            age INTEGER NOT NULL,
            gender TEXT CHECK(gender IN ('Male', 'Female', 'Other')),
            medical_history TEXT,
            created_by INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (created_by) REFERENCES users(id)
        )
        ''')
        
        # Caregiver assignments
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS patient_caregivers (
            patient_id INTEGER,
            caregiver_id INTEGER,
            relationship TEXT,
            assigned_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (patient_id, caregiver_id),
            FOREIGN KEY (patient_id) REFERENCES patients(id),
            FOREIGN KEY (caregiver_id) REFERENCES users(id)
        )
        ''')
        
        # Heart readings (enhanced with patient_id)
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS heart_readings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id INTEGER NOT NULL,
            timestamp TEXT NOT NULL,
            heart_rate INTEGER NOT NULL,
            hrv REAL NOT NULL,
            spo2 REAL NOT NULL,
            systolic INTEGER NOT NULL,
            diastolic INTEGER NOT NULL,
            body_temp REAL NOT NULL,
            tachycardia_pred INTEGER NOT NULL,
            hypertrophy_pred INTEGER NOT NULL,
            cholesterol_pred INTEGER NOT NULL,
            tachycardia_prob REAL,
            hypertrophy_prob REAL,
            cholesterol_prob REAL,
            notification_sent INTEGER DEFAULT 0,
            FOREIGN KEY (patient_id) REFERENCES patients(id)
        )
        ''')
        
        # Create default admin user if none exists
        cursor.execute("SELECT COUNT(*) FROM users")
        if cursor.fetchone()[0] == 0:
            default_password = hashlib.sha256("admin123".encode()).hexdigest()
            cursor.execute(
                "INSERT INTO users (email, password_hash, full_name, role) VALUES (?, ?, ?, ?)",
                ("admin@heartmonitor.com", default_password, "System Administrator", "admin")
            )
            print("✅ Default admin user created: admin@heartmonitor.com / admin123")
        
        # Create sample patient if none exists
        cursor.execute("SELECT COUNT(*) FROM patients")
        if cursor.fetchone()[0] == 0:
            cursor.execute(
                "INSERT INTO patients (name, age, gender, medical_history, created_by) VALUES (?, ?, ?, ?, ?)",
                ("John Doe", 65, "Male", "Hypertension, Type 2 Diabetes", 1)
            )
            cursor.execute(
                "INSERT INTO patients (name, age, gender, medical_history, created_by) VALUES (?, ?, ?, ?, ?)",
                ("Jane Smith", 58, "Female", "Previous MI, High Cholesterol", 1)
            )
            print("✅ Sample patients created")
        
        conn.commit()

# ============ AUTH DECORATORS ============
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash("Please log in to access this page", "error")
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

def role_required(*roles):
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            if session.get('user_role') not in roles:
                flash("You don't have permission to access this page", "error")
                return redirect(url_for('index'))
            return f(*args, **kwargs)
        return decorated_function
    return decorator

# ============ HELPER FUNCTIONS ============
def get_current_user():
    """Get current user from session"""
    if 'user_id' not in session:
        return None
    
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT id, email, full_name, role FROM users WHERE id = ?", (session['user_id'],))
        user = cursor.fetchone()
        return dict(user) if user else None

def calculate_health_metrics(patient_id: int, heart_rate: int):
    """Calculate metrics and make predictions"""
    # Calculate derived metrics
    hrv = max(20, 100 - heart_rate)
    spo2 = 98 if heart_rate < 100 else 95
    systolic = 110 + (heart_rate // 10)
    diastolic = 70 + (heart_rate // 20)
    body_temp = 36.6 + (heart_rate - 70) * 0.01
    
    features = np.array([[heart_rate, hrv, spo2, systolic, diastolic, body_temp]])
    
    # Make predictions
    if MODELS_LOADED:
        tachycardia_pred = tachycardia_model.predict(features)[0]
        hypertrophy_pred = hypertrophy_model.predict(features)[0]
        cholesterol_pred = cholesterol_model.predict(features)[0]
        
        try:
            tachycardia_prob = tachycardia_model.predict_proba(features)[0][1]
            hypertrophy_prob = hypertrophy_model.predict_proba(features)[0][1]
            cholesterol_prob = cholesterol_model.predict_proba(features)[0][1]
        except:
            tachycardia_prob = hypertrophy_prob = cholesterol_prob = None
    else:
        # Fallback logic
        tachycardia_pred = 1 if heart_rate > 100 else 0
        hypertrophy_pred = 0
        cholesterol_pred = 0
        tachycardia_prob = hypertrophy_prob = cholesterol_prob = None
    
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Check for dangerous conditions to trigger notifications
    notification_sent = 0
    if tachycardia_pred == 1 or heart_rate > 120 or spo2 < 92:
        notification_sent = 1
    
    # Store in database
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute('''
        INSERT INTO heart_readings 
        (patient_id, timestamp, heart_rate, hrv, spo2, systolic, diastolic, body_temp,
         tachycardia_pred, hypertrophy_pred, cholesterol_pred,
         tachycardia_prob, hypertrophy_prob, cholesterol_prob, notification_sent)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            patient_id, timestamp, heart_rate, hrv, spo2, systolic, diastolic, body_temp,
            int(tachycardia_pred), int(hypertrophy_pred), int(cholesterol_pred),
            tachycardia_prob, hypertrophy_prob, cholesterol_prob, notification_sent
        ))
    
    return {
        "success": True,
        "timestamp": timestamp,
        "tachycardia_pred": int(tachycardia_pred),
        "hypertrophy_pred": int(hypertrophy_pred),
        "cholesterol_pred": int(cholesterol_pred),
        "tachycardia_prob": float(tachycardia_prob) if tachycardia_prob else None,
        "hypertrophy_prob": float(hypertrophy_prob) if hypertrophy_prob else None,
        "cholesterol_prob": float(cholesterol_prob) if cholesterol_prob else None,
        "heart_rate": heart_rate,
        "hrv": hrv,
        "spo2": spo2,
        "systolic": systolic,
        "diastolic": diastolic,
        "body_temp": round(body_temp, 1),
        "blood_pressure": f"{systolic}/{diastolic}",
        "warning": "Tachycardia detected!" if tachycardia_pred == 1 else None
    }

def get_patients(user_role: str, user_id: int) -> List[Dict]:
    """Get patients based on user role"""
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        if user_role == 'admin':
            cursor.execute("SELECT id, name, age, gender, medical_history FROM patients ORDER BY name")
        elif user_role == 'doctor':
            cursor.execute("""
                SELECT DISTINCT p.id, p.name, p.age, p.gender, p.medical_history 
                FROM patients p
                ORDER BY p.name
            """)
        else:  # caregiver
            cursor.execute("""
                SELECT p.id, p.name, p.age, p.gender, p.medical_history
                FROM patients p
                JOIN patient_caregivers pc ON p.id = pc.patient_id
                WHERE pc.caregiver_id = ?
                ORDER BY p.name
            """, (user_id,))
        
        return [dict(row) for row in cursor.fetchall()]

def get_patient_details(patient_id: int) -> Optional[Dict]:
    """Get detailed patient information including caregivers"""
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute("SELECT id, name, age, gender, medical_history FROM patients WHERE id = ?", (patient_id,))
        patient = cursor.fetchone()
        if not patient:
            return None
        
        patient_dict = dict(patient)
        
        # Get caregivers
        cursor.execute("""
            SELECT u.id, u.full_name as name, pc.relationship, u.email
            FROM users u
            JOIN patient_caregivers pc ON u.id = pc.caregiver_id
            WHERE pc.patient_id = ? AND u.role = 'caregiver'
        """, (patient_id,))
        patient_dict['caregivers'] = [dict(row) for row in cursor.fetchall()]
        
        return patient_dict

def get_recent_readings(patient_id: Optional[int] = None, limit: int = 20) -> List[Dict]:
    """Get recent readings, optionally filtered by patient"""
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        if patient_id:
            cursor.execute("""
                SELECT hr.*, p.name as patient_name
                FROM heart_readings hr
                JOIN patients p ON hr.patient_id = p.id
                WHERE hr.patient_id = ?
                ORDER BY hr.id DESC
                LIMIT ?
            """, (patient_id, limit))
        else:
            cursor.execute("""
                SELECT hr.*, p.name as patient_name
                FROM heart_readings hr
                JOIN patients p ON hr.patient_id = p.id
                ORDER BY hr.id DESC
                LIMIT ?
            """, (limit,))
        
        readings = []
        for row in cursor.fetchall():
            reading = dict(row)
            reading['blood_pressure'] = f"{reading['systolic']}/{reading['diastolic']}"
            readings.append(reading)
        
        return readings

# ============ ROUTES ============
@app.route("/login", methods=["GET", "POST"])
def login():
    if 'user_id' in session:
        return redirect(url_for('index'))
    
    if request.method == "POST":
        email = request.form.get("email")
        password = request.form.get("password")
        password_hash = hashlib.sha256(password.encode()).hexdigest()
        
        with sqlite3.connect(DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(
                "SELECT id, full_name, role FROM users WHERE email = ? AND password_hash = ?",
                (email, password_hash)
            )
            user = cursor.fetchone()
            
            if user:
                session['user_id'] = user['id']
                session['user_name'] = user['full_name']
                session['user_role'] = user['role']
                flash(f"Welcome back, {user['full_name']}!", "success")
                return redirect(url_for('index'))
            else:
                flash("Invalid email or password", "error")
    
    return render_template("login.html")

@app.route("/register", methods=["GET", "POST"])
def register():
    if 'user_id' in session:
        return redirect(url_for('index'))
    
    if request.method == "POST":
        email = request.form.get("email")
        password = request.form.get("password")
        full_name = request.form.get("full_name")
        role = request.form.get("role", "caregiver")
        
        # Validate input
        if not all([email, password, full_name]):
            flash("All fields are required", "error")
            return redirect(url_for('register'))
        
        if len(password) < 6:
            flash("Password must be at least 6 characters", "error")
            return redirect(url_for('register'))
        
        # Hash password
        password_hash = hashlib.sha256(password.encode()).hexdigest()
        
        try:
            with sqlite3.connect(DB_PATH) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "INSERT INTO users (email, password_hash, full_name, role) VALUES (?, ?, ?, ?)",
                    (email, password_hash, full_name, role)
                )
                flash("Registration successful! Please log in.", "success")
                return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            flash("Email already exists. Please use a different email.", "error")
        except Exception as e:
            flash(f"Registration error: {str(e)}", "error")
    
    return render_template("register.html")

@app.route("/logout")
def logout():
    session.clear()
    flash("You have been logged out", "success")
    return redirect(url_for('login'))

@app.route("/")
@login_required
def index():
    current_user = get_current_user()
    patients = get_patients(current_user['role'], current_user['id'])
    
    selected_patient_id = request.args.get('patient_id', type=int)
    selected_patient = None
    readings = []
    
    if selected_patient_id:
        selected_patient = get_patient_details(selected_patient_id)
        readings = get_recent_readings(selected_patient_id, 20)
    elif patients:
        selected_patient = get_patient_details(patients[0]['id'])
        readings = get_recent_readings(patients[0]['id'], 20)
    
    return render_template("index.html", 
                         current_user=current_user,
                         patients=patients,
                         selected_patient=selected_patient,
                         readings=readings)

@app.route("/submit_reading", methods=["POST"])
@login_required
def submit_reading():
    patient_id = request.form.get("patient_id", type=int)
    heart_rate = request.form.get("heart_rate", type=int)
    
    if not patient_id or not heart_rate:
        return jsonify({"success": False, "error": "Please provide patient ID and heart rate"}), 400
    
    if heart_rate < 30 or heart_rate > 250:
        return jsonify({"success": False, "error": "Heart rate out of valid range (30-250 BPM)"}), 400
    
    try:
        result = calculate_health_metrics(patient_id, heart_rate)
        return jsonify(result)
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/api/data")
@login_required
def api_data():
    patient_id = request.args.get('patient_id', type=int)
    readings = get_recent_readings(patient_id, 20)
    return jsonify({"readings": readings})

@app.route("/api/patients")
@login_required
def api_patients():
    """API endpoint to get patients list"""
    current_user = get_current_user()
    patients = get_patients(current_user['role'], current_user['id'])
    return jsonify({"patients": patients})

@app.route("/patients")
@login_required
def patients_list():
    current_user = get_current_user()
    patients = get_patients(current_user['role'], current_user['id'])
    return render_template("patients.html", 
                         current_user=current_user,
                         patients=patients)

@app.route("/patients/<int:patient_id>")
@login_required
def patient_detail(patient_id):
    current_user = get_current_user()
    patient = get_patient_details(patient_id)
    if not patient:
        flash("Patient not found", "error")
        return redirect(url_for('patients_list'))
    
    readings = get_recent_readings(patient_id, 50)
    return render_template("patient_detail.html", 
                         current_user=current_user,
                         patient=patient,
                         readings=readings)

@app.route("/users")
@login_required
@role_required('admin')
def users_list():
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT id, email, full_name, role, created_at FROM users ORDER BY created_at")
        users = [dict(row) for row in cursor.fetchall()]
    
    return render_template("users.html", current_user=get_current_user(), users=users)

@app.route("/caregivers")
@login_required
@role_required('admin', 'doctor')
def caregivers_list():
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT id, email, full_name, role, created_at FROM users WHERE role = 'caregiver' ORDER BY full_name")
        caregivers = [dict(row) for row in cursor.fetchall()]
    
    return render_template("caregivers.html", current_user=get_current_user(), caregivers=caregivers)

@app.route("/devices")
@login_required
def devices():
    """IoT Devices page placeholder"""
    current_user = get_current_user()
    return render_template("devices.html", current_user=current_user)

@app.route("/profile")
@login_required
def profile():
    """User profile page"""
    current_user = get_current_user()
    return render_template("profile.html", current_user=current_user)

@app.route("/change_password", methods=["POST"])
@login_required
def change_password():
    """Change user password"""
    old_password = request.form.get("old_password")
    new_password = request.form.get("new_password")
    
    if not old_password or not new_password:
        flash("Both old and new passwords are required", "error")
        return redirect(url_for('profile'))
    
    if len(new_password) < 6:
        flash("New password must be at least 6 characters", "error")
        return redirect(url_for('profile'))
    
    old_hash = hashlib.sha256(old_password.encode()).hexdigest()
    new_hash = hashlib.sha256(new_password.encode()).hexdigest()
    
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT id FROM users WHERE id = ? AND password_hash = ?",
            (session['user_id'], old_hash)
        )
        if cursor.fetchone():
            cursor.execute(
                "UPDATE users SET password_hash = ? WHERE id = ?",
                (new_hash, session['user_id'])
            )
            flash("Password changed successfully!", "success")
        else:
            flash("Current password is incorrect", "error")
    
    return redirect(url_for('profile'))
# Add this BEFORE your existing routes (near the top after app initialization)

@app.route('/hybridaction/<path:path>', methods=['GET', 'POST', 'PUT', 'DELETE'])
def hybridaction_catchall(path):
    """Catch and ignore malicious hybridaction requests"""
    # Log it but don't show error
    app.logger.warning(f"Blocked malicious request to /hybridaction/{path}")
    # Return empty 200 response to stop the requester
    return '', 200

# Also catch any other suspicious patterns
@app.route('/<path:invalid_path>', methods=['GET', 'POST'])
def catch_all(invalid_path):
    """Catch all other invalid routes"""
    # Only block specific patterns we know are malicious
    if 'hybridaction' in invalid_path or 'zybTracker' in invalid_path:
        return '', 200
    # For real 404s, return proper error
    return render_template('404.html'), 404
@app.route("/health")
def health_check():
    """Health check endpoint"""
    try:
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute("SELECT 1")
        return jsonify({
            "status": "healthy",
            "models_loaded": MODELS_LOADED,
            "timestamp": datetime.datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({"status": "unhealthy", "error": str(e)}), 500

# Initialize database
init_db()

if __name__ == '__main__':
    print("\n" + "="*50)
    print("🚀 Heart Monitor Application Starting...")
    print("="*50)
    print(f"📍 Access the app at: http://localhost:5000")
    print(f"🔑 Default admin login: admin@heartmonitor.com / admin123")
    print("="*50 + "\n")
    app.run(debug=True, host='0.0.0.0', port=5000)