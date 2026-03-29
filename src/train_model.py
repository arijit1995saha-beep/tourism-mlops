import json
import os
from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

try:
    import mlflow
    import mlflow.sklearn
    MLFLOW_AVAILABLE = True
except Exception:
    MLFLOW_AVAILABLE = False

try:
    from datasets import load_dataset
    DATASETS_AVAILABLE = True
except Exception:
    DATASETS_AVAILABLE = False

try:
    from huggingface_hub import create_repo, upload_folder
    HF_AVAILABLE = True
except Exception:
    HF_AVAILABLE = False

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"
ARTIFACTS_DIR = PROJECT_DIR / "artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

HF_TOKEN = os.getenv("HF_TOKEN")
HF_DATASET_REPO = os.getenv("HF_DATASET_REPO", "your-username/tourism-package-data")
HF_MODEL_REPO = os.getenv("HF_MODEL_REPO", "your-username/tourism-package-model")
TARGET_COLUMN = "ProdTaken"

def load_split(file_name: str) -> pd.DataFrame:
    if DATASETS_AVAILABLE:
        try:
            dataset_dict = load_dataset(HF_DATASET_REPO, data_files={"data": file_name})
            return dataset_dict["data"].to_pandas()
        except Exception:
            pass
    return pd.read_csv(DATA_DIR / file_name)

def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    return {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions),
        "recall": recall_score(y_test, predictions),
        "f1_score": f1_score(y_test, predictions),
        "roc_auc": roc_auc_score(y_test, probabilities),
    }

def main():
    train_df = load_split("train.csv")
    test_df = load_split("test.csv")

    X_train = train_df.drop(columns=[TARGET_COLUMN])
    y_train = train_df[TARGET_COLUMN]
    X_test = test_df.drop(columns=[TARGET_COLUMN])
    y_test = test_df[TARGET_COLUMN]

    categorical_columns = X_train.select_dtypes(include="object").columns.tolist()
    numeric_columns = [col for col in X_train.columns if col not in categorical_columns]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), numeric_columns),
            ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("encoder", OneHotEncoder(handle_unknown="ignore"))]), categorical_columns),
        ]
    )

    models_and_params = {
        "random_forest": (
            RandomForestClassifier(random_state=42, class_weight="balanced"),
            {
                "model__n_estimators": [100, 200],
                "model__max_depth": [None, 8],
                "model__min_samples_split": [2, 5],
            }
        ),
        "gradient_boosting": (
            GradientBoostingClassifier(random_state=42),
            {
                "model__n_estimators": [100, 150],
                "model__learning_rate": [0.05, 0.1],
                "model__max_depth": [2, 3],
            }
        )
    }

    if MLFLOW_AVAILABLE:
        mlflow.set_tracking_uri(f"file:{PROJECT_DIR / 'mlruns'}")
        mlflow.set_experiment("tourism_package_prediction")

    best_model = None
    best_model_name = None
    best_params = None
    best_metrics = None
    best_f1 = -1
    results = []

    for model_name, (model, param_grid) in models_and_params.items():
        pipeline = Pipeline([("preprocessor", preprocessor), ("model", model)])
        grid_search = GridSearchCV(pipeline, param_grid=param_grid, cv=3, scoring="f1", n_jobs=-1)

        if MLFLOW_AVAILABLE:
            with mlflow.start_run(run_name=model_name):
                grid_search.fit(X_train, y_train)
                tuned_model = grid_search.best_estimator_
                metrics = evaluate_model(tuned_model, X_test, y_test)
                mlflow.log_params(grid_search.best_params_)
                mlflow.log_metrics(metrics)
                mlflow.sklearn.log_model(tuned_model, artifact_path="model")
        else:
            grid_search.fit(X_train, y_train)
            tuned_model = grid_search.best_estimator_
            metrics = evaluate_model(tuned_model, X_test, y_test)

        results.append({
            "model_name": model_name,
            "best_params": str(grid_search.best_params_),
            **metrics
        })

        if metrics["f1_score"] > best_f1:
            best_f1 = metrics["f1_score"]
            best_model = tuned_model
            best_model_name = model_name
            best_params = grid_search.best_params_
            best_metrics = metrics

    joblib.dump(best_model, ARTIFACTS_DIR / "best_model.joblib")
    pd.DataFrame(results).to_csv(ARTIFACTS_DIR / "all_model_results.csv", index=False)

    with open(ARTIFACTS_DIR / "metrics.json", "w", encoding="utf-8") as file:
        json.dump(best_metrics, file, indent=4)

    with open(ARTIFACTS_DIR / "model_summary.json", "w", encoding="utf-8") as file:
        json.dump({
            "best_model_name": best_model_name,
            "best_params": best_params,
            "best_metrics": best_metrics
        }, file, indent=4)

    if HF_AVAILABLE and HF_TOKEN:
        create_repo(repo_id=HF_MODEL_REPO, repo_type="model", exist_ok=True, token=HF_TOKEN)
        upload_folder(folder_path=str(ARTIFACTS_DIR), repo_id=HF_MODEL_REPO, repo_type="model", token=HF_TOKEN)

    print("Model training completed successfully.")

if __name__ == "__main__":
    main()