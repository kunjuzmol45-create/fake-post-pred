from flask import Flask, request, render_template, redirect, url_for, flash, jsonify, session
from flask_cors import CORS
import sqlite3
import os
from werkzeug.security import generate_password_hash, check_password_hash
import pandas as pd
from scipy.sparse import hstack
import pickle
import requests
from bs4 import BeautifulSoup
from PIL import Image
import pytesseract
from io import BytesIO


# Configure pytesseract path (adjust path based on your Tesseract installation)
# IMPORTANT: You must install Tesseract OCR separately!
# 
# For Windows:
#   1. Download: https://github.com/UB-Mannheim/tesseract/wiki
#   2. Run installer
#   3. Uncomment the line below and update the path if needed:
import sys
import platform
if platform.system() == 'Windows':
    tesseract_path = r'C:\Program Files\Tesseract-OCR'
    pytesseract.pytesseract.pytesseract_cmd = tesseract_path + r'\tesseract.exe'
    # Also add to PATH so it can find DLL dependencies
    if tesseract_path not in os.environ.get('PATH', ''):
        os.environ['PATH'] += os.pathsep + tesseract_path
# 
# For macOS: brew install tesseract
# For Linux: sudo apt-get install tesseract-ocr

# Verify Tesseract is accessible
try:
    pytesseract.get_tesseract_version()
    print("✅ Tesseract OCR found and working!")
except Exception as e:
    print(f"⚠️  Warning: Tesseract OCR not accessible: {e}")
    print(f"   Tesseract path: {pytesseract.pytesseract.pytesseract_cmd}")
    print("   Image upload features will not work. Please ensure Tesseract is installed.")


app = Flask(__name__)
app.secret_key = os.urandom(24)
CORS(app)


# Database helpers for users
def get_db_connection():
    conn = sqlite3.connect('users.db')
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_db_connection()
    conn.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL
        )
    ''')
    conn.commit()
    conn.close()


# Ensure users DB exists
init_db()


# Load ML models and vectorizer

tfidfs = pickle.load( open('vectorizers.pickle', 'rb') )
Passives = pickle.load( open('classifiers.pickle', 'rb') )
mlp = pickle.load(open('mlp.pkl', 'rb'))
tfidf = pickle.load(open('tfidf.pkl', 'rb'))
Passive = pickle.load(open('Passive.pkl', 'rb'))
Gradient = pickle.load(open('Gradient.pkl', 'rb'))

# Internshala Scraper
def scrape_internshala(url):
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        res = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(res.text, 'html.parser')
        blocks = soup.find_all('div', {'class': ['text-container', 'text-container additional_detail']})
        if blocks:
            return " ".join([b.get_text(" ", strip=True) for b in blocks])
    except Exception as e:
        print(f"Internshala scraping error: {e}")
    return None



@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html')
    


@app.route('/chart')
def chart():
    return render_template('chart.html')

@app.route('/performance')
def performance():
    return render_template('performance.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        if not username or not password:
            flash('Username and password are required.')
            return redirect(url_for('login'))

        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute('SELECT * FROM users WHERE username = ?', (username,))
        user = cur.fetchone()
        conn.close()

        if user and check_password_hash(user['password'], password):
            # mark user as logged in
            session['user'] = username
            return redirect(url_for('upload_image'))
        else:
            flash('Invalid username or password.')
            return redirect(url_for('login'))

    return render_template('login.html')


@app.route('/upload-image')
def upload_image():
    """Display the image upload page for authenticated users"""
    return render_template('upload_image.html')


@app.route('/upload-image-process', methods=['POST'])
def upload_image_process():
    """Process the uploaded job post image with OCR and prediction"""
    try:
        # Check if image file is provided
        if 'image' not in request.files:
            return jsonify(success=False, message='No image file provided'), 400
        
        file = request.files['image']
        
        if file.filename == '':
            return jsonify(success=False, message='No file selected'), 400
        
        # Validate file type
        allowed_extensions = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}
        if not ('.' in file.filename and file.filename.rsplit('.', 1)[1].lower() in allowed_extensions):
            return jsonify(success=False, message='Invalid file type. Allowed: PNG, JPG, JPEG, GIF, BMP'), 400
        
        # Create uploads directory if it doesn't exist
        upload_dir = 'uploads'
        if not os.path.exists(upload_dir):
            os.makedirs(upload_dir)
        
        # Save the file with a unique name
        import uuid
        from datetime import datetime
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        unique_id = str(uuid.uuid4())[:8]
        filename = f"{timestamp}_{unique_id}_{file.filename}"
        filepath = os.path.join(upload_dir, filename)
        
        file.save(filepath)
        
        # Extract text from image using OCR
        try:
            extracted_text = extract_text_from_image(filepath)
            if not extracted_text or len(extracted_text.strip()) < 10:
                # If OCR doesn't extract enough text, return with message
                os.remove(filepath)  # Clean up
                return jsonify(success=False, message='Could not extract clear text from image. Please ensure the image is clear and readable.'), 400
        except Exception as ocr_error:
            os.remove(filepath)  # Clean up
            return jsonify(success=False, message=f'Error processing image: {str(ocr_error)}'), 500
        
        # Make prediction based on extracted text (includes explanation)
        prediction_result = predict_from_text(extracted_text)

        # Return success with prediction and explanation
        return jsonify(
            success=True,
            message='Image processed successfully',
            filename=filename,
            extracted_text=extracted_text[:500],  # Send first 500 chars
            prediction=prediction_result.get('prediction'),
            model_used=prediction_result.get('model'),
            confidence='High' if prediction_result.get('prediction') == 'Fraudulent' else 'Moderate',
            explanation=prediction_result.get('explanation', ''),
            feature_explanations=prediction_result.get('feature_explanations', [])
        ), 200
        
    except Exception as e:
        return jsonify(success=False, message=f'Upload error: {str(e)}'), 500


def extract_text_from_image(image_path):
    """Extract text from image using OCR"""
    try:
        # Open image
        img = Image.open(image_path)
        
        # Convert to RGB if necessary (for PNG with transparency, etc.)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Enhance image for better OCR results
        from PIL import ImageEnhance, ImageFilter
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(2)
        enhancer = ImageEnhance.Sharpness(img)
        img = enhancer.enhance(2)
        
        # Extract text using Tesseract
        text = pytesseract.image_to_string(img)
        
        return text if text else "No text found in image"
    except Exception as e:
        error_msg = str(e)
        # Check if it's a Tesseract-related error
        if "tesseract is not installed" in error_msg.lower() or "tesseract" in error_msg.lower():
            raise Exception("Tesseract OCR not installed or not found. Please ensure Tesseract OCR is installed at C:\\Program Files\\Tesseract-OCR\\tesseract.exe")
        else:
            raise Exception(f"OCR failed: {error_msg}")


def predict_from_text(description):
    """Make prediction based on extracted text from image"""
    try:
        # Clean and prepare the text
        description = str(description).strip()
        if not description:
            return {'prediction': 'Unknown', 'model': 'N/A'}
        
        # Use PassiveAggressiveClassifier (as default model)
        # TF-IDF transformation
        sample_text_tfidf = tfidf.transform([description])
        
        # For single text prediction, we'll use default values for numeric features
        # These represent: [Telecommuting, Has_company_logo, Has_questions, 
        #                   Employment_type, Required_experience, Required_education, Function]
        default_numeric = [[0, 0, 0, 6, 8, 14, 41]]
        
        # Combine text and numeric features
        from scipy.sparse import csr_matrix, hstack
        numeric_sparse = csr_matrix(default_numeric)
        combined_features = hstack([sample_text_tfidf, numeric_sparse])
        
        # Make predictions with all available models
        predictions = {
            'PassiveAggressiveClassifier': Passive.predict(combined_features)[0],
            'MLPClassifier': mlp.predict(combined_features)[0],
            'GradientBoostingClassifier': Gradient.predict(combined_features)[0]
        }

        # Use PassiveAggressiveClassifier as the default reported model
        pred_value = predictions['PassiveAggressiveClassifier']
        result = "Fraudulent" if pred_value == 1 else "Legitimate"

        # Build a simple, transparent explanation based on keyword heuristics
        explanation_reasons = []
        text_lower = description.lower()

        patterns = {
            'requests_money': ['bank transfer', 'wire transfer', 'transfer', 'pay to', 'payment', 'fee', 'deposit', 'send money'],
            'messaging_contact': ['whatsapp', 'telegram', 'viber', 'imo', 'line'],
            'personal_info': ['passport', 'id card', 'social security', 'ssn', 'personal details', 'cv', 'resume', 'date of birth'],
            'external_link': ['http://', 'https://', 'click here', 'apply now', 'link', 'website'],
            'urgent_language': ['urgent', 'asap', 'immediately', 'hurry', 'apply now'],
            'free_email': ['gmail.com', 'yahoo.com', 'hotmail.com', 'outlook.com'],
            'salary_terms': ['salary', 'per month', 'per year', 'per annum', 'earn', 'income', '₹', '$']
        }

        if any(k in text_lower for k in patterns['requests_money']):
            explanation_reasons.append('Asks for payment or bank transfer')
        if any(k in text_lower for k in patterns['messaging_contact']):
            explanation_reasons.append('Requests contact via external messaging app (e.g. WhatsApp)')
        if any(k in text_lower for k in patterns['personal_info']):
            explanation_reasons.append('Requests sensitive personal documents or info')
        if any(k in text_lower for k in patterns['external_link']):
            explanation_reasons.append('Contains external links or prompts to click')
        if any(k in text_lower for k in patterns['urgent_language']):
            explanation_reasons.append('Uses urgent language to pressure applicants')
        if any(k in text_lower for k in patterns['free_email']):
            explanation_reasons.append('Uses free/public email addresses instead of corporate email')
        if any(k in text_lower for k in patterns['salary_terms']):
            explanation_reasons.append('Mentions earnings/salary (may be a lure)')

        # Convert short reason phrases into user-friendly sentences
        explanation_text = ''
        if explanation_reasons:
            # map short phrases to full sentences for users
            phrase_map = {
                'Asks for payment or bank transfer': 'The posting asks for payment or bank transfer — legitimate employers do not ask applicants to pay.',
                'Requests contact via external messaging app (e.g. WhatsApp)': 'The post asks you to move the conversation to external messaging apps (e.g. WhatsApp/Telegram) instead of company email, which can be a way to avoid platform oversight.',
                'Requests sensitive personal documents or info': 'It requests sensitive personal documents or information (e.g. passport, SSN), which is unnecessary at early stages and risky to share.',
                'Contains external links or prompts to click': 'It contains external links or prompts to click; malicious links can lead to phishing sites.',
                'Uses urgent language to pressure applicants': 'The message uses urgent or high-pressure language to make you act quickly, a common scam tactic.',
                'Uses free/public email addresses instead of corporate email': 'It uses free/public email addresses rather than a company domain, which is suspicious for official job offers.',
                "Mentions earnings/salary (may be a lure)": 'It promises earnings or highlights salary in a way that looks like a lure; be cautious of vague high-pay claims.'
            }
            sentences = []
            seen = set()
            for ph in explanation_reasons:
                if ph in phrase_map and ph not in seen:
                    sentences.append(phrase_map[ph])
                    seen.add(ph)
            explanation_text = ' '.join(sentences)
        else:
            # When model flags as fraudulent but no clear heuristic reason found
            if result == 'Fraudulent':
                explanation_text = 'The model flagged this post as potentially fraudulent, but no clear human-readable rule matched the text.'

        # Add model-based top contributing text features when possible
        feature_explanations = []
        try:
            # number of text features from TF-IDF
            n_text_feats = sample_text_tfidf.shape[1]

            # get feature names from tfidf
            try:
                feature_names = tfidf.get_feature_names_out()
            except Exception:
                feature_names = tfidf.get_feature_names()

            # convert sparse vector to dense array
            text_vec = sample_text_tfidf.toarray()[0]

            # Try linear model contributions (PassiveAggressiveClassifier)
            coef = getattr(Passive, 'coef_', None)
            if coef is not None and coef.size >= n_text_feats:
                coef_text = coef.reshape(coef.shape[0], -1)[0][:n_text_feats]
                contributions = text_vec * coef_text
            else:
                # fallback to gradient feature importances if available
                fi = getattr(Gradient, 'feature_importances_', None)
                if fi is not None and fi.size >= n_text_feats:
                    contributions = text_vec * fi[:n_text_feats]
                else:
                    contributions = None

            if contributions is not None:
                # pick top positive contributors
                import numpy as _np
                idx = _np.argsort(contributions)[::-1]
                topn = 6
                count = 0
                for i in idx:
                    if count >= topn:
                        break
                    score = float(contributions[i])
                    if score <= 0:
                        break
                    fname = feature_names[i] if i < len(feature_names) else str(i)
                    feature_explanations.append({'feature': fname, 'score': round(score, 6)})
                    count += 1
        except Exception:
            feature_explanations = []

        # Enhance the human-readable explanation by appending top features
        if feature_explanations:
            # make a user-friendly sentence listing top keywords
            top_words = [f['feature'] for f in feature_explanations][:6]
            if top_words:
                words_display = ', '.join([f"'{w}'" for w in top_words])
                feature_sentence = f"The text contains keywords such as {words_display} that the model associates with fraudulent posts."
                if explanation_text:
                    explanation_text = explanation_text + ' ' + feature_sentence
                else:
                    explanation_text = feature_sentence

        return {
            'prediction': result,
            'model': 'PassiveAggressiveClassifier',
            'confidence_score': float(pred_value),
            'explanation': explanation_text,
            'feature_explanations': feature_explanations
        }
        
    except Exception as e:
        return {
            'prediction': 'Error',
            'model': 'Unknown',
            'error': str(e)
        }


@app.route('/logout')
def logout():
    """Logout user and redirect to home page"""
    session.pop('user', None)
    return redirect(url_for('index'))


@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        if not username or not password:
            flash('Username and password are required.')
            return redirect(url_for('signup'))

        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute('SELECT id FROM users WHERE username = ?', (username,))
        if cur.fetchone():
            conn.close()
            flash('Username already exists.')
            return redirect(url_for('signup'))

        hashed = generate_password_hash(password)
        cur.execute('INSERT INTO users (username, password) VALUES (?, ?)', (username, hashed))
        conn.commit()
        conn.close()

        # After successful signup, mark user logged in and redirect to the prediction page
        session['user'] = username
        return redirect(url_for('prediction'))

    return render_template('signup.html')


@app.route('/api/signup', methods=['POST'])
def api_signup():
    data = request.get_json(force=True, silent=True) or request.form
    username = data.get('username')
    password = data.get('password')
    if not username or not password:
        return jsonify(success=False, message='Username and password required'), 400

    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute('SELECT id FROM users WHERE username = ?', (username,))
    if cur.fetchone():
        conn.close()
        return jsonify(success=False, message='Username exists'), 409

    hashed = generate_password_hash(password)
    cur.execute('INSERT INTO users (username, password) VALUES (?, ?)', (username, hashed))
    conn.commit()
    conn.close()
    # mark user as logged in for API signup
    session['user'] = username
    return jsonify(success=True, message='User created')


@app.route('/api/login', methods=['POST'])
def api_login():
    data = request.get_json(force=True, silent=True) or request.form
    username = data.get('username')
    password = data.get('password')
    if not username or not password:
        return jsonify(success=False, message='Username and password required'), 400

    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute('SELECT * FROM users WHERE username = ?', (username,))
    user = cur.fetchone()
    conn.close()
    if user and check_password_hash(user['password'], password):
        session['user'] = username
        return jsonify(success=True, message='Login successful')
    return jsonify(success=False, message='Invalid credentials'), 401

@app.route('/upload')
def upload():
    # Upload page removed — dataset is preloaded. Redirect to prediction page.
    return redirect(url_for('prediction'))

@app.route('/preview', methods=["POST"])
def preview():
    # Preview/upload disabled — using pretrained dataset instead.
    return redirect(url_for('prediction'))

@app.route('/prediction')
def prediction():
    return render_template('prediction.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get form inputs
        description = str(request.form['news'])
        Telecommuting = int(request.form['Telecommuting'])
        Has_company_logo = int(request.form['Has_company_logo'])
        Has_questions = int(request.form['Has_questions'])
        Employment_type = int(request.form['Employment_type'])
        Required_experience = int(request.form['Required_experience'])
        Required_education = int(request.form['Required_education'])
        Function = int(request.form['Function'])
        model = request.form['Model']

        # Map form inputs to readable strings
        Telecommutings = 'Yes' if Telecommuting == 1 else 'No'
        Has_company_logos = 'Yes' if Has_company_logo == 1 else 'No'
        Has_questionss = 'Yes' if Has_questions == 1 else 'No'

        employment_map = {
            1: 'Full-time', 2: 'Part-time', 3: 'Contract',
            4: 'Temporary', 5: 'Other', 6: 'Not Mentioned'
        }
        Employment_types = employment_map.get(Employment_type, 'Unknown')

        experience_map = {
            1: 'Mid-Senior level', 2: 'Executive', 3: 'Entry level', 4: 'Associate',
            5: 'Not Applicable', 6: 'Director', 7: 'Internship', 8: 'Not Mentioned'
        }
        Required_experiences = experience_map.get(Required_experience, 'Unknown')

        education_map = {
            1: 'Master Degree', 2: 'Bachelor Degree', 3: 'Unspecified', 4: 'High School or equivalent',
            5: 'Associate Degree', 6: 'Vocational', 7: 'Vocational - HS Diploma',
            8: 'Professional', 9: 'Some High School Coursework', 10: 'Some College Coursework Completed',
            11: 'Certification', 12: 'Doctorate', 13: 'Vocational - Degree',
            14: 'Not Mentioned required_education'
        }
        Required_educations = education_map.get(Required_education, 'Unknown')

        function_map = {
            1: 'Marketing', 2: 'Customer Service', 3: 'Information Technology', 4: 'Sales',
            5: 'Health Care Provider', 6: 'Management', 7: 'Other', 8: 'Engineering',
            9: 'Administrative', 10: 'Design', 11: 'Production', 12: 'Education',
            13: 'Supply Chain', 14: 'Business Development', 15: 'Product Management',
            16: 'Financial Analyst', 17: 'Consulting', 18: 'Human Resources', 22: 'Project Management',
            23: 'Manufacturing', 24: 'Public Relations', 25: 'Strategy/Planning',
            26: 'Advertising', 27: 'Finance', 28: 'General Business', 29: 'Research',
            30: 'Accounting/Auditing', 31: 'Art/Creative', 32: 'Quality Assurance',
            33: 'Data Analyst', 34: 'Business Analyst', 35: 'Writing/Editing',
            36: 'Distribution', 37: 'Science', 38: 'Training', 39: 'Purchasing',
            40: 'Legal', 41: 'Not Mentioned function'
        }
        Functions = function_map.get(Function, 'Unknown')

        # Prepare data for prediction
        sample = {
            'description': [description],
            'Telecommuting': [Telecommuting],
            'Has_company_logo': [Has_company_logo],
            'Has_questions': [Has_questions],
            'Employment_type': [Employment_type],
            'Required_experience': [Required_experience],
            'Required_education': [Required_education],
            'Function': [Function]
        }
        sample_df = pd.DataFrame(sample)

        sample_text_tfidf = tfidf.transform(sample_df['description'])
        sample_numeric = sample_df.drop('description', axis=1).values
        sample_combined = hstack([sample_text_tfidf, sample_numeric])

        # Predict
        if model == "MLPClassifier":
            RESULT = mlp.predict(sample_combined)
        elif model == "PassiveAggressiveClassifier":
            RESULT = Passive.predict(sample_combined)
        elif model == "GradientBoostingClassifier":
            RESULT = Gradient.predict(sample_combined)
        else:
            RESULT = [0]  # Default to legitimate if model not found

        result = "Fraudulent" if RESULT[0] == 1 else "Legitimate"

        explanation_result = predict_from_text(description)
        return render_template('result.html',
                       prediction_text=result,
                       model=model,
                       description=description,
                       Telecommutings=Telecommutings,
                       Has_company_logo=Has_company_logos,
                       Has_questionss=Has_questionss,
                       Employment_types=Employment_types,
                       Required_experiences=Required_experiences,
                       Required_educations=Required_educations,
                       Functions=Functions,
                       explanation=explanation_result.get('explanation', ''),
                       feature_explanations=explanation_result.get('feature_explanations', []))


@app.route('/api/predict', methods=['POST'])
def api_predict():
    data = request.get_json(force=True, silent=True) or request.form
    try:
        description = str(data.get('news') or data.get('description') or '')
        Telecommuting = int(data.get('Telecommuting', 0))
        Has_company_logo = int(data.get('Has_company_logo', 0))
        Has_questions = int(data.get('Has_questions', 0))
        Employment_type = int(data.get('Employment_type', 6))
        Required_experience = int(data.get('Required_experience', 8))
        Required_education = int(data.get('Required_education', 14))
        Function = int(data.get('Function', 41))
        model = data.get('Model', 'PassiveAggressiveClassifier')
    except Exception as e:
        return jsonify(success=False, message=f'Invalid input: {e}'), 400

    sample = {
        'description': [description],
        'Telecommuting': [Telecommuting],
        'Has_company_logo': [Has_company_logo],
        'Has_questions': [Has_questions],
        'Employment_type': [Employment_type],
        'Required_experience': [Required_experience],
        'Required_education': [Required_education],
        'Function': [Function]
    }
    sample_df = pd.DataFrame(sample)
    sample_text_tfidf = tfidf.transform(sample_df['description'])
    sample_numeric = sample_df.drop('description', axis=1).values
    sample_combined = hstack([sample_text_tfidf, sample_numeric])

    if model == "MLPClassifier":
        RESULT = mlp.predict(sample_combined)
    elif model == "PassiveAggressiveClassifier":
        RESULT = Passive.predict(sample_combined)
    elif model == "GradientBoostingClassifier":
        RESULT = Gradient.predict(sample_combined)
    else:
        RESULT = [0]

    result = "Fraudulent" if RESULT[0] == 1 else "Legitimate"
    explanation_result = predict_from_text(description)
    return jsonify(success=True, prediction=result, model=model, explanation=explanation_result.get('explanation', ''), feature_explanations=explanation_result.get('feature_explanations', []))

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
