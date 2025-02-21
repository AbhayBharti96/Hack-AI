import os
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# ✅ Define CSV Path
csv_path = r"C:/Users/abhay/Desktop/python/FoodDonation/donations.csv"

# ✅ Load CSV
try:
    df = pd.read_csv(csv_path)
    print("✅ CSV loaded successfully!")
except FileNotFoundError:
    print(f"❌ Error: File not found at {csv_path}")
    exit()

# ✅ Check column names
print(f"📝 Columns in dataset: {df.columns.tolist()}")

# ✅ Check if necessary columns exist
required_columns = {"manufacturing_date", "shelf_life", "quantity"}
if not required_columns.issubset(df.columns):
    print(f"❌ Error: Missing columns! Expected {required_columns}, but found {set(df.columns)}")
    exit()

# 🔍 Debug: Show missing values before processing
print(f"🔍 Missing values before processing:\n{df.isnull().sum()}")

# ✅ Convert manufacturing_date to datetime, handling various formats
def convert_to_datetime(date_str):
    formats = ["%Y-%m-%d", "%d-%m-%Y", "%m/%d/%Y", "%d/%m/%Y", "%Y/%m/%d"]  # Add more formats if needed
    for fmt in formats:
        try:
            return pd.to_datetime(date_str, format=fmt, errors="raise")
        except ValueError:
            pass # try next format
    return pd.NaT # return NaT if none of the formats work

df["manufacturing_date"] = df["manufacturing_date"].apply(convert_to_datetime)

# 🔍 Debug: Show invalid dates
invalid_dates = df[df["manufacturing_date"].isnull()]
if not invalid_dates.empty:
    print(f"⚠️ Warning: {len(invalid_dates)} rows have invalid manufacturing dates!")
    print(invalid_dates[["manufacturing_date"]])  # Print some rows

# ✅ Drop rows with missing manufacturing_date or shelf_life
df = df.dropna(subset=["manufacturing_date", "shelf_life"])

# ✅ Convert shelf_life to integer
df["shelf_life"] = pd.to_numeric(df["shelf_life"], errors="coerce")

# 🔍 Debug: Show invalid shelf_life
invalid_shelf_life = df[df["shelf_life"].isnull()]
if not invalid_shelf_life.empty:
    print(f"⚠️ Warning: {len(invalid_shelf_life)} rows have invalid shelf_life values!")
    print(invalid_shelf_life[["shelf_life"]])  # Print some rows

# ✅ Calculate expiry_date
df["expiry_date"] = df["manufacturing_date"] + pd.to_timedelta(df["shelf_life"], unit="D")

# ✅ Drop invalid expiry_date rows
df = df.dropna(subset=["expiry_date"])

# ✅ Calculate days to expiry
df["days_to_expiry"] = (df["expiry_date"] - pd.Timestamp.today()).dt.days

# ✅ Print data check
print(f"✅ Valid rows after processing: {len(df)}")
if len(df) == 0:
    print("❌ No valid rows left after processing. Check your dataset.")
    exit()

# ✅ Features & Labels
X = df[["quantity", "days_to_expiry"]]
y = (df["days_to_expiry"] < 5).astype(int)  # 1 if expiring soon, else 0

# ✅ Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ✅ Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# ✅ Ensure directory exists
model_dir = r"C:/Users/abhay/Desktop/python/FoodDonation/ml_model"
os.makedirs(model_dir, exist_ok=True)

# ✅ Save model
model_path = os.path.join(model_dir, "expiry_model.pkl")
with open(model_path, "wb") as f:
    pickle.dump(model, f)

print(f"✅ Model saved successfully at {model_path}")

# ✅ Evaluate the model (Vis added)
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

y_pred = model.predict(X_test)

print("\n✅ Model Evaluation:")
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(f"Classification Report:\n{classification_report(y_test, y_pred)}")
print(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")

#✅ Save Evaluation results to a file (Vis added)
evaluation_results = {
    "Accuracy": accuracy_score(y_test, y_pred),
    "Classification Report": classification_report(y_test, y_pred, output_dict=True),
    "Confusion Matrix": confusion_matrix(y_test, y_pred).tolist()
}

evaluation_path = os.path.join(model_dir, "evaluation_results.pkl")
with open(evaluation_path, "wb") as f:
    pickle.dump(evaluation_results, f)

print(f"✅ Evaluation results saved at {evaluation_path}")

# ✅ Print the first 5 predictions and actual values (Vis added)
print("\n✅ First 5 Predictions vs. Actual:")
for i in range(min(5, len(y_test))):
    print(f"Prediction: {y_pred[i]}, Actual: {y_test.iloc[i]}")