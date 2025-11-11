import os
import pandas as pd
import pytest


def test_raw_data_file_exists():
    """Ensure the raw data file exists."""
    assert os.path.exists("data/raw_data.csv"), "raw_data.csv is missing!"

def test_imb_data_file_exists():
    """Ensure the imbalance data file exists."""
    assert os.path.exists("data/imb_data.csv"), "imb_data.csv is missing!"

def test_required_columns_in_raw_data():
    """Check raw_data.csv has all required columns."""
    df = pd.read_csv("data/raw_data.csv")
    required_columns = ['count', 'hate_speech', 'offensive_language', 'neither', 'class', 'tweet']
    
    for col in required_columns:
        assert col in df.columns, f"Column '{col}' is missing in raw_data.csv"

def test_required_columns_in_imb_data():
    """Check imb_data.csv has all required columns."""
    df = pd.read_csv("data/imb_data.csv")
    required_columns = ['label', 'tweet']
    
    for col in required_columns:
        assert col in df.columns, f"Column '{col}' is missing in imb_data.csv"



def test_no_missing_values_in_raw_data():
    """Ensure no missing values in 'text' and 'label' columns."""
    df = pd.read_csv("data/raw_data.csv")
    assert df["class"].notnull().all(), "Some texts are missing in raw_data.csv"
    assert df["tweet"].notnull().all(), "Some labels are missing in raw_data.csv"

def test_no_missing_values_in_imb_data():
    """Ensure no missing values in imbalance dataset."""
    df = pd.read_csv("data/imb_data.csv")
    assert df["label"].notnull().all(), "Some texts are missing in imb_data.csv"
    assert df["tweet"].notnull().all(), "Some labels are missing in imb_data.csv"
