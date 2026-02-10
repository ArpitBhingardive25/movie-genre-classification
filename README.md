##🎬 Movie Genre Classification using Machine Learning
##📌 Project Overview

-This project builds a Machine Learning based text classification system that predicts the genre of a movie using its plot summary / description.
-The model is trained using Natural Language Processing (NLP) techniques like TF-IDF vectorization and Support Vector Machine (SVM) classification.
-The dataset is provided in raw .txt format and is parsed and processed programmatically before training the model.
##🎯 Objectives

-Process raw movie dataset stored in text files
-Perform text preprocessing and feature extraction
-Train ML model to classify movie genres
-Evaluate model performance
-Predict genres for unseen movie descriptions

##🧠 Machine Learning Approach
🔹 Text Processing
Lowercasing text
Removing special characters
Cleaning extra spaces

🔹 Feature Extraction
TF-IDF Vectorization (Converts text → numeric vectors)

🔹 Model Used
Linear Support Vector Machine (LinearSVC)

📂 Project Structure
movie-genre-classification/
│
├── data/
│   ├── train_data.txt
│   └── test_data.txt
│
├── src/
│   ├── data_loader.py
│   ├── preprocess.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
│
├── models/
│   └── genre_model.pkl
│
├── requirements.txt
├── README.md
└── main.py

##⚙️ Technologies Used
Python
Scikit-learn
Pandas
Joblib
VS Code

##📊 Model Performance
The model was evaluated using:
Accuracy
Precision
Recall
F1 Score

Sample accuracy achieved:
👉 ~55–60% (depends on dataset split and preprocessing)

🚀 How to Run the Project
1️⃣ Install Dependencies
pip install pandas scikit-learn joblib

2️⃣ Train Model
python src/train.py


##Output:

Classification Report
Saved Model → models/genre_model.pkl
3️⃣ Run Prediction
python src/predict.py

##Output:
predictions.csv

Contains predicted genres for test dataset.

##📥 Dataset

Dataset contains movie records in .txt format:

Train Data Format
ID ::: TITLE ::: GENRE ::: DESCRIPTION

Test Data Format
ID ::: TITLE ::: DESCRIPTION

##💡 Key Features

✅ Handles raw text dataset
✅ End-to-end ML pipeline
✅ Model saving and reuse
✅ Real-world NLP application
✅ Clean modular code structure

##🔮 Future Improvements

Add Deep Learning (LSTM / BERT)
Build Web Interface (Flask / Streamlit)
Add Multi-label Genre Prediction
Hyperparameter tuning

##👨‍💻 Author
Arpit Bhingardive
