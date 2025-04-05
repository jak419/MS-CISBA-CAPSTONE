# 🛡️ Fraud Detection Pipeline
This project uses machine learning and secure preprocessing to detect fraudulent bank transactions. This is a **production-ready fraud detection system** using a Decision Tree classifier with a Scikit-learn pipeline. It is designed to process bank transactions and predict fraudulent activity. This folder constitutes the  Software Systems (SS), Data Management (DM) and Cybersuecity and Networking (CN) areas of the course as they are linked together.

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
1. **Create and Activate a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate      # On Windows: venv\Scripts\activate

2. **Install required libraries**
   ```bash
   pip install -r requirements.txt

3. **Create transactions databse(transactions.db) by running create_db.py**
   ```bash
   python create_db.py

4. **Load data into Database in CSV format. You can reuse this script to import any CSV data into the transactions.db**
   ```bash
   python load_csv_to_db.py

## ✅ Usage
5. **🧠 For training and prediction (labeled data)**
      ```bash
   python fraud_detection.py sample_data.csv

6. **🔍 For prediction only (unlabeled data)**
      ```bash
   python fraud_detection.py unseen_data.csv

7. **🧼 Usage (Testing program on from data in database). Use SQLite DB**
      ```bash
   python fraud_detection.py transactions.db --db

8. **💾 Saving results back to SQLite**
      ```bash
   python fraud_detection.py transactions.db --db --save-db --output-db transactions.db

## 🧪 Testing
9. **👉 Make sure you are in the right directory to execute the unit tests.**
      ```bash
   cd /c/Users/your_computer_username/Capstone_CIDM-6395/fraud_detection_project

10. **Then exute the scripts:**
      ```bash
    pytest tests/
    pytest -p no:warnings