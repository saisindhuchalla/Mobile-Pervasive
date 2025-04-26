from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()

class HealthLog(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    patient_name = db.Column(db.String(100))
    age = db.Column(db.Integer)
    gender = db.Column(db.String(10))
    allergies = db.Column(db.Text)
    medications = db.Column(db.Text)
    symptoms = db.Column(db.Text)
    conditions = db.Column(db.Text)
    timestamp = db.Column(db.DateTime)
