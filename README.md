# Tourism Wellness Package MLOps Project

## Project Flow
1. Register raw data in Hugging Face dataset repo
2. Load dataset from Hugging Face
3. Clean data and create train/test split
4. Upload train/test files back to Hugging Face
5. Train and tune ML models
6. Track experiments with MLflow
7. Save and upload the best model to Hugging Face model hub
8. Deploy with Streamlit and Docker
9. Automate everything with GitHub Actions

## Important GitHub Secrets
- HF_TOKEN
- HF_DATASET_REPO
- HF_MODEL_REPO
- HF_SPACE_REPO