from flask import Flask, render_template, request, redirect, jsonify, url_for
from datetime import datetime
from models import db, HealthLog
import pandas as pd
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
import torch
import joblib
import openai
import os
from dotenv import load_dotenv
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///health_data.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db.init_app(app)

with app.app_context():
    db.create_all()

@app.route('/')
def index():
    return redirect(url_for('home'))

@app.route('/health-log')
def health_log():
    return render_template('form.html')

@app.route('/submit', methods=['POST'])
def submit():
    data = HealthLog(
        patient_name=request.form['patient_name'],
        age=int(request.form['age']),
        gender=request.form['gender'],
        allergies=request.form['allergies'],
        medications=request.form['medications'],
        symptoms=request.form['symptoms'],
        conditions=request.form['conditions'],
        timestamp=datetime.now()
    )
    db.session.add(data)
    db.session.commit()
    return redirect('/entries')

@app.route('/entries')
def entries():
    logs = HealthLog.query.order_by(HealthLog.timestamp.desc()).all()
    return render_template('entries.html', logs=logs)

@app.route('/chat', methods=['GET', 'POST'])
def chat():
    patient_names = db.session.query(HealthLog.patient_name).distinct().all()
    patient_names = [name[0] for name in patient_names]
    return render_template('chat.html', patient_names=patient_names)

openai.api_key = os.getenv("OPENAI_API_KEY", "your-openai-api-key-here")

def is_disease_mentioned(message):
    message = message.lower()
    for disease in avoid_dict.keys():
        if disease.lower().strip() in message:
            return disease
    return None

precaution = pd.read_csv("disease_data/symptom_precaution.csv")
precaution_dict = precaution.groupby("Disease")[["Precaution_1", "Precaution_2", "Precaution_3", "Precaution_4"]].first().to_dict(orient="index")

def get_precautions(disease):
    return precaution_dict.get(disease, {"message": "No precautions available"})

def get_avoidance(disease):
    return avoid_dict.get(disease, {})

avoidance = pd.read_csv("disease_data/symptom_avoidance.csv")
avoid_dict = avoidance.groupby("Disease")[["Avoid_1", "Avoid_2", "Avoid_3", "Avoid_4"]].first().to_dict(orient="index")


@app.route('/get-response', methods=['POST'])
def smart_response():
    user_input = request.form['message']
    patient_name = request.form['patient_name']

    # Detect when it's better to switch to ChatGPT
    def is_general_question(text):
        text = text.lower()
        return any(keyword in text for keyword in [
            "can i", "should i", "is it okay", "what if", "what happens", 
            "why", "when", "how", "food", "exercise", "lifestyle", "avoid", "drink", "eat", "medication"
        ])

    if is_general_question(user_input):
        # Forward to ChatGPT
        return redirect(url_for('get_gpt_response'), code=307)  # POST preserve

    # Else, use your local NLP model route (predict disease etc.)
    return redirect(url_for('get_response_logic'), code=307)

@app.route('/get-response-logic', methods=['POST'])
def get_response_logic():
    user_input = request.form['message']
    patient_name = request.form['patient_name']

    # Get latest log for the patient
    log = HealthLog.query.filter_by(patient_name=patient_name).order_by(HealthLog.timestamp.desc()).first()
    if not log:
        return jsonify({'response': f"Hi {patient_name}, I couldn’t find any health data for you yet. Please fill out the form first."})

    allergies = [a.strip().lower() for a in log.allergies.split(',') if a.strip()]

    # Predict disease
    def predict_disease(text):
        known = is_disease_mentioned(text)
        if known:
            return known
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            logits = model(**inputs).logits
        pred_class = torch.argmax(logits, dim=1).item()
        return label_encoder.inverse_transform([pred_class])[0]

    try:
        predicted = predict_disease(user_input)
        precautions = get_precautions(predicted)
        avoid = get_avoidance(predicted)

        # Filter allergy-safe precautions
        safe_precautions = [p for p in precautions.values() if not any(a in p.lower() for a in allergies)]

        response = f"Hi {log.patient_name} 👋! Here's what I found for **{predicted}**.\n\n"
        if allergies:
            response += f"I noticed you're allergic to: {', '.join(allergies)}. I've filtered the precautions accordingly.\n\n"

        response += "✅ Recommended actions:\n" + (
            '\n'.join(f"• {p}" for p in safe_precautions) if safe_precautions else "• No allergy-safe precautions found.\n"
        )

        if avoid:
            avoid_lines = [f"• {a}" for a in avoid.values() if isinstance(a, str) and a.strip()]
            if avoid_lines:
                response += "\n\n🚫 Things to avoid:\n" + '\n'.join(avoid_lines)

    except Exception as e:
        response = f"Sorry {patient_name}, I couldn't understand your symptoms. Please try again later."

    return jsonify({'response': response})

@app.route('/get-gpt-response', methods=['POST'])
def get_gpt_response():
    user_input = request.form['message']
    patient_name = request.form['patient_name']

    log = HealthLog.query.filter_by(patient_name=patient_name).order_by(HealthLog.timestamp.desc()).first()
    if not log:
        return jsonify({'response': f"Hi {patient_name}, I couldn’t find any health data for you yet. Please fill out the form first."})

    # Set up the GPT prompt
    system_prompt = f"""You are a helpful health assistant.
    
Patient Info:
- Name: {log.patient_name}
- Age: {log.age}
- Gender: {log.gender}
- Allergies: {log.allergies}
- Medications: {log.medications}
- Known Conditions: {log.conditions}

You must give clear, conversational responses tailored to this user.
"""

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_input}
            ]
        )
        reply = response['choices'][0]['message']['content']
        return jsonify({'response': reply})
        return jsonify({'response': response.choices[0].message['content']})
    except Exception as e:
        print("❌ OpenAI Error:", str(e))
        return jsonify({'response': "Sorry, I couldn’t process your request right now. Please try again later."})

@app.route('/home')
def home():
    patient_names = db.session.query(HealthLog.patient_name).distinct().all()
    patient_names = [name[0] for name in patient_names]
    return render_template("home.html", patient_names=patient_names)

@app.route('/home/patient')
def view_patient():
    name = request.args.get('name')
    log = HealthLog.query.filter_by(patient_name=name).order_by(HealthLog.timestamp.desc()).first()
    logs = HealthLog.query.filter_by(patient_name=name).order_by(HealthLog.timestamp).all()
    return render_template("patient_profile.html", log=log, logs=logs)

model_path = "disease_nlp_model"
tokenizer = DistilBertTokenizerFast.from_pretrained(model_path, local_files_only=True)
model = DistilBertForSequenceClassification.from_pretrained(model_path, local_files_only=True)
model.eval()
label_encoder = joblib.load("label_encoder.pkl")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
print("🔑 Loaded API Key:", openai.api_key[:8] + "..." if openai.api_key else "❌ Not Loaded")