from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from flask_cors import CORS

app = Flask(__name__, template_folder='templates')  # To serve HTML files
CORS(app)  # Enable CORS

# Database Configuration (Replace with your actual DB credentials)
app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://postgres:root@localhost:5432/diabetes_predictions'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# Define the Table Model
class DiabetesPrediction(db.Model):
    __tablename__ = 'diabetes_predictions'
    id = db.Column(db.Integer, primary_key=True)
    pregnancies = db.Column(db.Integer, nullable=False)
    glucose = db.Column(db.Integer, nullable=False)
    blood_pressure = db.Column(db.Integer, nullable=False)
    skin_thickness = db.Column(db.Integer, nullable=False)
    insulin = db.Column(db.Integer, nullable=False)
    bmi = db.Column(db.Float, nullable=False)
    diabetes_pedigree = db.Column(db.Float, nullable=False)
    age = db.Column(db.Integer, nullable=False)
    prediction_result = db.Column(db.String, nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

# Create database tables
with app.app_context():
    db.create_all()

# Load the dataset
data = pd.read_csv("D:/Diabetes_prediction/diabetes_dataset.csv")

# Prepare the data
X = data.drop(columns="Outcome", axis=1)
Y = data["Outcome"]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

# Standardize
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train model
classifier = svm.SVC(kernel='linear')
classifier.fit(X_train, Y_train)

@app.route('/')
def home():
    return render_template('index.html')  # Serve frontend

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        input_data = np.asarray(data['data']).reshape(1, -1)
        std_data = scaler.transform(input_data)
        prediction = classifier.predict(std_data)
        
        # Store in database
        new_entry = DiabetesPrediction(
            pregnancies=data['data'][0], glucose=data['data'][1], blood_pressure=data['data'][2],
            skin_thickness=data['data'][3], insulin=data['data'][4], bmi=data['data'][5],
            diabetes_pedigree=data['data'][6], age=data['data'][7], prediction_result="Has Diabetes" if prediction[0] == 1 else "No Diabetes"
        )
        db.session.add(new_entry)
        db.session.commit()

        return jsonify({'prediction': int(prediction[0])})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
