import os
import pandas as pd
from flask import Flask, jsonify, request
from flask_cors import CORS
from datetime import datetime, timedelta
from twilio.rest import Client  

app = Flask(__name__)
CORS(app)

CSV_PATH = "donations.csv"  # Update with the correct path if needed

def load_data():
    try:
        df = pd.read_csv(CSV_PATH)
        df.rename(columns=lambda x: x.strip().lower(), inplace=True)
        df['manufacturing_date'] = pd.to_datetime(df['manufacturing_date'], errors='coerce')
        df['shelf_life'] = pd.to_numeric(df['shelf_life'], errors='coerce')
        df['expiry_date'] = df['manufacturing_date'] + pd.to_timedelta(df['shelf_life'], unit='D')
        return df
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return None

df = load_data()

@app.route("/donations", methods=["GET"])
def get_donations():
    if df is None:
        return jsonify({"error": "CSV file not found or invalid."}), 500
    return jsonify(df.to_dict(orient='records'))

@app.route("/check_expiry", methods=["GET"])
def check_expiry():
    if df is None:
        return jsonify({"error": "CSV file not found or invalid."})
    
    today = datetime.today()
    expiring_soon = df[df['expiry_date'] <= today + timedelta(days=3)]
    return jsonify(expiring_soon.to_dict(orient='records'))

@app.route("/donation/<int:donation_id>", methods=["GET"])
def get_donation_by_id(donation_id):
    if df is None:
        return jsonify({"error": "CSV file not found or invalid."})
    
    result = df[df['id'] == donation_id]
    if result.empty:
        return jsonify({"error": "Donation not found"}), 404
    return jsonify(result.to_dict(orient='records'))

if __name__ == "__main__":
    app.run(debug=True)
