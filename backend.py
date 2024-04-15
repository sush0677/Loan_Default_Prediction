from flask import Flask, request, jsonify, send_from_directory
import pandas as pd
import io
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import joblib

from flask import Flask, render_template, request, jsonify, redirect, url_for
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Serve Home Page
@app.route('/')
def home():
    return render_template('index.html')

# Serve About Page
@app.route('/about')
def about():
    return render_template('Aboutus.html')

# Serve Contact Page
@app.route('/contact')
def contact():
    return render_template('Contact.html')
# Define the route for uploading and processing the file
@app.route('/upload', methods=['POST'])
def upload_file():
    # Check if the post request has the file part
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400

    file = request.files['file']

    # If the user does not select a file, the browser submits an empty file without a filename.
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file:
        # Read the uploaded file directly into a DataFrame
        df = pd.read_csv(io.BytesIO(file.read()))

        required_columns = ['Default']  # Example required columns

        # Check if the required columns are in the DataFrame
        if not all(column in df.columns for column in required_columns):
            return jsonify({"error": "Uploaded file is missing some required columns."}), 400

        # Proceed with preprocessing
        X = df.drop('Default', axis=1)
        y = df['Default']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Define and train the stacking model
        base_models = [
            ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
            ('xgb', xgb.XGBClassifier(n_estimators=100, use_label_encoder=False, eval_metric='logloss', random_state=42)),
            ('lr', LogisticRegression(max_iter=1000))
        ]

        meta_model = GradientBoostingClassifier(n_estimators=50, learning_rate=0.1, random_state=42)
        stacking_model = StackingClassifier(estimators=base_models, final_estimator=meta_model, cv=5)
        stacking_model.fit(X_train_scaled, y_train)

        # Predictions and evaluation
        y_pred = stacking_model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        cr = classification_report(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)

        # Save the model, plots, and classification report
        joblib.dump(stacking_model, 'stacking_model.pkl')
        save_confusion_matrix(cm, 'confusion_matrix.png')
        save_classification_report(cr, 'classification_report.png')

        return jsonify({"message": "File processed, model trained, and evaluation metrics saved."})

    return jsonify({"error": "An unexpected error occurred"}), 500

@app.route('/plot/<filename>')
def get_plot(filename):
    return send_from_directory(directory='.', filename=filename)

@app.route('/download_model')
def download_model():
    return send_from_directory(directory='.', filename='stacking_model.pkl', as_attachment=True)

# Additional functions for saving plots
def save_classification_report(cr, filename='classification_report.png', title='Classification Report', cmap=plt.cm.Blues):
    lines = cr.split('\n')
    classes = []
    plotMat = []
    for line in lines[2: (len(lines) - 5)]:
        t = line.split()
        classes.append(t[0])
        v = [float(x) * 100 for x in t[1: len(t) - 1]]
        plotMat.append(v)

    plotMat = np.array(plotMat)
    fig, ax = plt.subplots()
    sns.heatmap(plotMat, annot=True, cmap=cmap, ax=ax, cbar=False,
                xticklabels=['Precision', 'Recall', 'F1-score'],
                yticklabels=classes, fmt=".1f")
    ax.set_title(title)
    ax.set_xlabel('Metrics')
    ax.set_ylabel('Classes')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def save_confusion_matrix(cm, filename='confusion_matrix.png'):
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap='Blues', ax=ax,
                xticklabels=['Not Default', 'Default'],
                yticklabels=['Not Default', 'Default'])
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

if __name__ == '__main__':
    app.run(debug=True)
