import os
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

try:
    from datasets import load_dataset
    DATASETS_AVAILABLE = True
except Exception:
    DATASETS_AVAILABLE = False

try:
    from huggingface_hub import create_repo, upload_file
    HF_AVAILABLE = True
except Exception:
    HF_AVAILABLE = False

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"
RAW_FILE = DATA_DIR / "tourism.csv"
CLEAN_FILE = DATA_DIR / "cleaned_tourism.csv"
TRAIN_FILE = DATA_DIR / "train.csv"
TEST_FILE = DATA_DIR / "test.csv"

HF_TOKEN = os.getenv("HF_TOKEN")
HF_DATASET_REPO = os.getenv("HF_DATASET_REPO", "your-username/tourism-package-data")
TARGET_COLUMN = "ProdTaken"
DROP_COLUMNS = ["Unnamed: 0", "CustomerID"]

def upload_file_to_hf(local_path: Path):
    if not HF_AVAILABLE or not HF_TOKEN:
        print(f"Skipping upload for {local_path.name}")
        return

    create_repo(repo_id=HF_DATASET_REPO, repo_type="dataset", exist_ok=True, token=HF_TOKEN)
    upload_file(
        path_or_fileobj=str(local_path),
        path_in_repo=local_path.name,
        repo_id=HF_DATASET_REPO,
        repo_type="dataset",
        token=HF_TOKEN
    )

def load_data():
    if DATASETS_AVAILABLE:
        try:
            dataset = load_dataset(HF_DATASET_REPO, split="train")
            return dataset.to_pandas()
        except Exception:
            pass
    return pd.read_csv(RAW_FILE)

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.drop(columns=[col for col in DROP_COLUMNS if col in df.columns])
    df = df.drop_duplicates().reset_index(drop=True)

    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype(str).str.strip()
        df[col] = df[col].replace({"nan": pd.NA, "": pd.NA})

    return df

def main():
    df = load_data()
    df = clean_data(df)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(CLEAN_FILE, index=False)

    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        stratify=df[TARGET_COLUMN]
    )

    train_df.to_csv(TRAIN_FILE, index=False)
    test_df.to_csv(TEST_FILE, index=False)

    for file_path in [CLEAN_FILE, TRAIN_FILE, TEST_FILE]:
        upload_file_to_hf(file_path)

    print("Data preparation completed successfully.")

if __name__ == "__main__":
    main()