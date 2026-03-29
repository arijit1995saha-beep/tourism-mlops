
import os
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
from huggingface_hub import hf_hub_download

st.set_page_config(page_title="Tourism Package Predictor", page_icon="🌍")
st.title("🌍 Wellness Tourism Package Predictor")
st.write("Enter customer details to predict package purchase.")

HF_MODEL_REPO = os.getenv("HF_MODEL_REPO", "your-username/tourism-package-model")
HF_TOKEN = os.getenv("HF_TOKEN")
LOCAL_MODEL_PATH = Path(__file__).resolve().parents[1] / "artifacts" / "best_model.joblib"

@st.cache_resource
def load_model():
    try:
        model_path = hf_hub_download(
            repo_id=HF_MODEL_REPO,
            filename="best_model.joblib",
            repo_type="model",
            token=HF_TOKEN,
        )
        return joblib.load(model_path)
    except Exception:
        return joblib.load(LOCAL_MODEL_PATH)

model = load_model()

with st.form("prediction_form"):
    age = st.number_input("Age", min_value=18, max_value=80, value=35)
    typeofcontact = st.selectbox("TypeofContact", ["Company Invited", "Self Inquiry", "Self Enquiry"])
    citytier = st.selectbox("CityTier", [1, 2, 3])
    durationofpitch = st.number_input("DurationOfPitch", min_value=1, max_value=100, value=15)
    occupation = st.selectbox("Occupation", ["Salaried", "Freelancer", "Small Business", "Large Business"])
    gender = st.selectbox("Gender", ["Male", "Female"])
    numberofpersonvisiting = st.number_input("NumberOfPersonVisiting", min_value=1, max_value=10, value=2)
    numberoffollowups = st.number_input("NumberOfFollowups", min_value=0, max_value=10, value=2)
    productpitched = st.selectbox("ProductPitched", ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"])
    preferredpropertystar = st.selectbox("PreferredPropertyStar", [1, 2, 3, 4, 5])
    maritalstatus = st.selectbox("MaritalStatus", ["Single", "Married", "Divorced", "Unmarried"])
    numberoftrips = st.number_input("NumberOfTrips", min_value=0, max_value=20, value=3)
    passport = st.selectbox("Passport", [0, 1])
    pitchsatisfactionscore = st.selectbox("PitchSatisfactionScore", [1, 2, 3, 4, 5])
    owncar = st.selectbox("OwnCar", [0, 1])
    numberofchildrenvisiting = st.number_input("NumberOfChildrenVisiting", min_value=0, max_value=5, value=0)
    designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])
    monthlyincome = st.number_input("MonthlyIncome", min_value=1000, max_value=200000, value=30000)

    submitted = st.form_submit_button("Predict")

if submitted:
    input_df = pd.DataFrame([{
        "Age": age,
        "TypeofContact": typeofcontact,
        "CityTier": citytier,
        "DurationOfPitch": durationofpitch,
        "Occupation": occupation,
        "Gender": gender,
        "NumberOfPersonVisiting": numberofpersonvisiting,
        "NumberOfFollowups": numberoffollowups,
        "ProductPitched": productpitched,
        "PreferredPropertyStar": preferredpropertystar,
        "MaritalStatus": maritalstatus,
        "NumberOfTrips": numberoftrips,
        "Passport": passport,
        "PitchSatisfactionScore": pitchsatisfactionscore,
        "OwnCar": owncar,
        "NumberOfChildrenVisiting": numberofchildrenvisiting,
        "Designation": designation,
        "MonthlyIncome": monthlyincome
    }])

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    st.write("### Prediction Result")
    st.dataframe(input_df)
    st.write(f"Prediction: {'Yes' if prediction == 1 else 'No'}")
    st.write(f"Probability of purchase: {probability:.2%}")

