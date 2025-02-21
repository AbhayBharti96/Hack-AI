from fastapi import FastAPI, HTTPException
import pickle
import pandas as pd
from pydantic import BaseModel
import smtplib
from email.message import EmailMessage
import os

# Load trained ML model
model_path = r"C:/Users/abhay/Desktop/python/FoodDonation/ml_model/expiry_model.pkl"

if not os.path.exists(model_path):
    raise FileNotFoundError(f"❌ Model file not found at {model_path}")

with open(model_path, "rb") as f:
    model = pickle.load(f)

# Initialize FastAPI app
app = FastAPI()

# Input Schema
class FoodItem(BaseModel):
    id: int
    organization: str
    food: str
    quantity: int
    expiry_date: str  # Expected format: 'YYYY-MM-DD'
    email: str
    phone: str
    status: str  # Example: "Pending", "Distributed", etc.

# Function to send email notifications
def send_email_notification(user_email, organization, food, quantity, days_left):
    sender_email = "your_email@gmail.com"  # Replace with your email
    sender_password = "your_email_password"  # Use an App Password if using Gmail

    subject = "⚠️ Food Expiry Alert!"
    body = f"Dear {organization},\n\nThe food item '{food}' with quantity {quantity} is expiring in {days_left} days. Please take necessary action.\n\nRegards,\nFood Donation System"

    msg = EmailMessage()
    msg.set_content(body)
    msg["Subject"] = subject
    msg["From"] = sender_email
    msg["To"] = user_email

    try:
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(sender_email, sender_password)
        server.send_message(msg)
        server.quit()
        print(f"✅ Email Sent Successfully to {user_email}!")
    except Exception as e:
        print(f"❌ Email Sending Failed: {e}")

# API Route for Prediction
@app.post("/predict")
def predict_expiry(item: FoodItem):
    try:
        # Convert expiry_date to datetime
        expiry_date = pd.to_datetime(item.expiry_date, errors="coerce")
        if pd.isna(expiry_date):
            raise HTTPException(status_code=400, detail="Invalid expiry date format. Use 'YYYY-MM-DD'.")

        # Calculate days to expiry
        days_to_expiry = (expiry_date - pd.Timestamp.today()).days

        # Prepare input data
        input_data = pd.DataFrame([[item.quantity, days_to_expiry]], columns=["quantity", "days_to_expiry"])

        # Predict
        prediction = model.predict(input_data)[0]  # 1 (Expiring soon), 0 (Safe)

        # Send email if expiring soon
        if prediction == 1:
            send_email_notification(item.email, item.organization, item.food, item.quantity, days_to_expiry)

        return {
            "id": item.id,
            "organization": item.organization,
            "food": item.food,
            "quantity": item.quantity,
            "days_to_expiry": days_to_expiry,
            "status": item.status,
            "expiring_soon": bool(prediction)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Run API Server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
