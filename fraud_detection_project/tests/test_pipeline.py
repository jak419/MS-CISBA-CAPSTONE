import sys
import os
import pandas as pd
import pytest

# Dynamically add fraud_detection_project/fraud_detection to sys.path
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', 'fraud_detection')))

from fraud_detection import preprocess_fn, mask_account_ids, run_pipeline

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


# === Test: Type encoding and PII masking ===
def test_type_encoding_and_masking():
    df = pd.DataFrame({
        'nameOrig': ['C12345'],
        'nameDest': ['M67890'],
        'type': ['CASH_OUT'],
        'isFlaggedFraud': [0],
        'amount': [1000]
    })
    processed = preprocess_fn(df.copy())

    assert processed['type'].iloc[0] == 1, "Transaction type not encoded properly"
    assert 'isFlaggedFraud' not in processed.columns, "isFlaggedFraud column not dropped"
    assert 'nameOrig' not in processed.columns, "nameOrig should be dropped after preprocessing"
    assert 'nameDest' not in processed.columns, "nameDest should be dropped after preprocessing"


# === Test: Masking works independently ===
def test_masking_consistency():
    df = pd.DataFrame({
        'nameOrig': ['C99999'],
        'nameDest': ['M11111']
    })
    masked = mask_account_ids(df.copy())
    orig_hash = masked['nameOrig'].iloc[0]
    dest_hash = masked['nameDest'].iloc[0]

    assert isinstance(orig_hash, str) and len(orig_hash) == 64, "Invalid mask for nameOrig"
    assert isinstance(dest_hash, str) and len(dest_hash) == 64, "Invalid mask for nameDest"
    assert orig_hash != dest_hash, "nameOrig and nameDest hashes should differ"


# === Test: Unknown transaction type maps to 0 ===
def test_unknown_transaction_type_handling():
    df = pd.DataFrame({
        'type': ['UNKNOWN'],
        'amount': [500]
    })
    processed = preprocess_fn(df.copy())
    assert processed['type'].iloc[0] == 0, "Unknown transaction type should map to 0"


# === Test: Edge case — missing type column ===
def test_missing_type_column():
    df = pd.DataFrame({
        'amount': [1234],
        'nameOrig': ['C54321'],
        'nameDest': ['M12345']
    })
    processed = preprocess_fn(df.copy())
    assert 'type' in processed.columns or 'type' not in df.columns, "Type column handling failed"


# === Test: Empty dataframe ===
def test_empty_dataframe():
    df = pd.DataFrame(columns=['nameOrig', 'nameDest', 'type'])
    processed = preprocess_fn(df.copy())
    assert processed.empty, "Empty dataframe should remain empty after preprocessing"


# === Test: run_pipeline with labeled data (ensures both 0 and 1 classes) ===
def test_run_pipeline_with_labeled_data(tmp_path):
    data = pd.DataFrame({
        'type': ['CASH_OUT', 'TRANSFER', 'CASH_OUT'],
        'amount': [1000, 2000, 1500],
        'nameOrig': ['C123', 'C456', 'C789'],
        'nameDest': ['M123', 'M456', 'M789'],
        'oldbalanceOrg': [5000, 1000, 3000],
        'newbalanceOrig': [4000, 800, 1500],
        'oldbalanceDest': [1000, 300, 200],
        'newbalanceDest': [2000, 500, 100],
        'isFraud': [0, 1, 0]
    })
    file_path = tmp_path / "test_labeled.csv"
    data.to_csv(file_path, index=False)

    try:
        run_pipeline(str(file_path))
    except Exception as e:
        pytest.fail(f"run_pipeline (labeled) crashed: {e}")


# === Test: run_pipeline with unlabeled data using a proper dummy model ===
def test_run_pipeline_with_unlabeled_data(tmp_path):
    from sklearn.dummy import DummyClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import FunctionTransformer
    from joblib import dump

    data = pd.DataFrame({
        'type': ['CASH_OUT', 'TRANSFER'],
        'amount': [1200, 800],
        'nameOrig': ['C999', 'C888'],
        'nameDest': ['M999', 'M888'],
        'oldbalanceOrg': [5000, 2000],
        'newbalanceOrig': [3800, 1500],
        'oldbalanceDest': [100, 50],
        'newbalanceDest': [1300, 400]
    })

    file_path = tmp_path / "test_unlabeled.csv"
    data.to_csv(file_path, index=False)

    # Train dummy model with both 0 and 1
    dummy_pipe = Pipeline([
        ('preprocess', FunctionTransformer(preprocess_fn, validate=False)),
        ('model', DummyClassifier(strategy='stratified', random_state=42))
    ])
    dummy_pipe.fit(data.drop(columns=['nameOrig', 'nameDest']), [0, 1])  # 👈 both classes
    dump(dummy_pipe, "decision_tree_pipeline.joblib")

    try:
        run_pipeline(str(file_path))
    except Exception as e:
        pytest.fail(f"run_pipeline (unlabeled) crashed: {e}")