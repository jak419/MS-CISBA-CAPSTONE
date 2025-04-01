# 🛡️ Fraud Detection Pipeline
This project uses machine learning and secure preprocessing to detect fraudulent bank transactions. This is a **production-ready fraud detection system** using a Decision Tree classifier with a Scikit-learn pipeline. It is designed to process bank transactions and predict fraudulent activity. 

---

## ✅ Features
- 🔁 **Scikit-learn pipeline**: Preprocessing and model training in one pipeline.
- 🧼 **PII-safe**: Sensitive fields like `nameOrig` and `nameDest` are SHA-256 hashed as part of security measures to prevent PII information moved on into business analytics system and decision support systems downstream.
- 📊 **Model evaluation**: Accuracy, precision, recall, F1-score, ROC curve.
- 🧠 **Threshold tuning**: Customizable probability threshold (`0.3`) to minimize false negatives.
- 🧪 **Unit tests**: Unit testing with `pytest`.
- 📂 **Handles both labeled and unlabeled datasets**.
- 📁 **Outputs**:
  - `fraud_predictions.csv` or `fraud_predictions_unlabeled.csv`
  - `model_report.txt`
  - `roc_curve.png`
  - `decision_tree_pipeline.joblib`

  ## 📦 Installation
<<<<<<< HEAD
# Clone the Repository
First, clone this repository from GitHub:
git clone https://github.com/jak419/fraud-detection-pipeline.git
cd fraud-detection-pipeline

# Create and Activate a Virtual Environment
python -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate

# Install Required libraries
pip install -r requirements.txt

# Install database transactions.db using create_db.py
python create_db.py

# Load csv sample data into database
python load_csv_to_db.py

## Usage (Testing program on csv)
# 🧠 For training and prediction (labeled data):
=======
## Create and activate a virtual environment
python -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate

## Install required libraries
pip install -r requirements.txt

## Usage
## 🧠 For training and prediction (labeled data):
>>>>>>> a41a5163d06dcc6881d7e35b86a81f359cc8ebe0
python fraud_detection.py sample_data.csv

## 🔍 For prediction only (unlabeled data):
python fraud_detection.py unseen_data.csv

## Usage (Testing program on from data in database). Use SQLite DB
python fraud_detection.py transactions.db --db

## Saving results back to SQLite
python fraud_detection.py transactions.db --db --save-db --output-db transactions.db

## 🧪 Testing
👉 Make sure you are in the right directory to execute the unit tests.
cd /c/Users/your_computer_username/Capstone_CIDM-6395/fraud_detection_project

<<<<<<< HEAD
🚀 Then eexute the scripts:
=======
Then exute the scripts:
>>>>>>> a41a5163d06dcc6881d7e35b86a81f359cc8ebe0
pytest tests/
or
pytest -p no:warnings
