import joblib
from data_loader import load_test_data
from preprocess import clean_text

# Load model
model = joblib.load("models/genre_model.pkl")

# Load test data
test_df = load_test_data("data/test_data.txt")
test_df["description"] = test_df["description"].apply(clean_text)

# Predict
test_df["predicted_genre"] = model.predict(test_df["description"])

# Save predictions
test_df.to_csv("predictions.csv", index=False)

print("✅ Predictions saved to predictions.csv")
