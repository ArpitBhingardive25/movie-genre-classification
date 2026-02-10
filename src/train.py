import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

from data_loader import load_train_data
from preprocess import clean_text

# Load data
df = load_train_data("data/train_data.txt")

# Clean text
df["description"] = df["description"].apply(clean_text)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    df["description"],
    df["genre"],
    test_size=0.2,
    random_state=42
)

# Pipeline
model = Pipeline([
    ("tfidf", TfidfVectorizer(stop_words="english", max_features=5000)),
    ("clf", LinearSVC())
])

# Train
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# Save model
joblib.dump(model, "models/genre_model.pkl")

print("✅ Model trained and saved successfully")
