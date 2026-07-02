import streamlit as st
import pandas as pd
import numpy as np
import pickle
import base64
import plotly.express as px

# 1. Helper function for bulk prediction download link
def get_binary_file_downloader_html(bin_df, file_label="Download"):
    csv = bin_df.to_csv(index=False)
    b64 = base64.b64encode(csv.encod3e()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="predicted_heart_disease_lr.csv">{file_label}</a>'
    return href

# 2. Page Configuration & Setup
st.title("Heart Disease Predictor")

tab1, tab2, tab3 = st.tabs(["Predict", "Bulk Predict", "Model Information"])

# ==========================================
# TAB 1: SINGLE INSTANCE PREDICTION
# ==========================================
with tab1:
    # Defining input fields based on dataset features
    age = st.number_input("Age (in years)", min_value=1, max_value=150, value=45)
    sex = st.selectbox("Sex", ["Male", "Female"])
    chest_pain = st.selectbox("Chest Pain Type", ["Typical Angina", "Atypical Angina", "Non-Anginal Pain", "Asymptomatic"])
    resting_bp = st.number_input("Resting Blood Pressure", min_value=50, max_value=250, value=120)
    cholesterol = st.number_input("Serum Cholesterol", min_value=50, max_value=600, value=200)
    fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["No", "Yes"])
    resting_ecg = st.selectbox("Resting ECG Results", ["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"])
    max_hr = st.number_input("Maximum Heart Rate Achieved", min_value=50, max_value=250, value=150)
    exercise_angina = st.selectbox("Exercise Induced Angina", ["No", "Yes"])
    oldpeak = st.number_input("Oldpeak (ST depression)", min_value=0.0, max_value=10.0, value=0.0, step=0.1)
    st_slope = st.selectbox("ST Slope", ["Upsloping", "Flat", "Downsloping"])

    # Pre-processing categorical fields into numeric weights to match model expectations
    sex_num = 0 if sex == "Male" else 1
    
    # Mapping other values appropriately based on categorical mapping
    cp_mapping = {"Typical Angina": 0, "Atypical Angina": 1, "Non-Anginal Pain": 2, "Asymptomatic": 3}
    fbs_mapping = {"No": 0, "Yes": 1}
    ecg_mapping = {"Normal": 0, "ST-T Wave Abnormality": 1, "Left Ventricular Hypertrophy": 2}
    exang_mapping = {"No": 0, "Yes": 1}
    slope_mapping = {"Upsloping": 0, "Flat": 1, "Downsloping": 2}

    # Construct the Input DataFrame format
    input_data = pd.DataFrame([{
        "Age": age,
        "Sex": sex_num,
        "ChestPainType": cp_mapping[chest_pain],
        "RestingBP": resting_bp,
        "Cholesterol": cholesterol,
        "FastingBS": fbs_mapping[fasting_bs],
        "RestingECG": ecg_mapping[resting_ecg],
        "MaxHR": max_hr,
        "ExerciseAngina": exang_mapping[exercise_angina],
        "Oldpeak": oldpeak,
        "ST_Slope": slope_mapping[st_slope]
    }])

    # Model file names list & front-end mapping
    algo_names = ["Decision Tree", "Logistic Regression", "Random Forest", "Support Vector Machine"]
    model_files = ["DesicionTree.pkl", "LogisticR.pkl", "RandomForest.pkl", "SVM.pkl"]
    

    def predict_heart_disease(data):
        predictions = []
        for file in model_files:
            with open(file, 'rb') as f:
                model = pickle.load(f)
            pred = model.predict(data)
            predictions.append(pred[0])
        return predictions

    if st.button("Submit"):
        st.subheader("Results")
        results = predict_heart_disease(input_data)
        
        for idx, algo in enumerate(algo_names):
            if results[idx] == 0:
                st.write(f"**{algo}:** No Heart Disease Detected")
            else:
                st.write(f"**{algo}:** Heart Disease is Detected")

# ==========================================
# TAB 2: BULK PREDICTION (CSV UPLOAD)
# ==========================================
with tab2:
    st.markdown("""
    ### Instructions before uploading the file:
    1. No NaN values are allowed (every cell must be filled).
    2. Total 11 features must be in the same order as Single Prediction.
    3. Verify feature naming conventions & match spelling carefully.
    4. Numeric mappings (e.g., Male = 0, Female = 1) must be configured in advance inside the CSV file.
    """)

    uploaded_file = st.file_uploader("Upload CSV file for batch inference", type=["csv"])

    expected_columns = ["Age", "Sex", "ChestPainType", "RestingBP", "Cholesterol", "FastingBS", "RestingECG", "MaxHR", "ExerciseAngina", "Oldpeak", "ST_Slope"]

    if uploaded_file is not None:
        bulk_data = pd.read_csv(uploaded_file)
        
        # Validating columns structure
        if list(bulk_data.columns) == expected_columns:
            # Loading primary model (Logistic Regression model used in the tutorial video demo)
            with open("logistic_regression.pkl", 'rb') as f:
                lr_model = pickle.load(f)
            
            # Initializing empty column
            bulk_data["Prediction_LR"] = np.nan
            
            # Batch looping logic
            for i in range(len(bulk_data)):
                # Exclude last prediction column during testing slice
                row_features = bulk_data.iloc[i, :11].values.reshape(1, -1)
                pred_val = lr_model.predict(row_features)
                bulk_data.at[i, "Prediction_LR"] = int(pred_val[0])
            
            # Local backup saving
            bulk_data.to_csv("predicted_heart_disease_lr.csv", index=False)
            
            # Rendering output metrics front-end layout
            st.subheader("Predictions Preview")
            st.write(bulk_data)
            
            # Creating download widget link template
            download_link = get_binary_file_downloader_html(bulk_data, "Download Prediction CSV")
            st.markdown(download_link, unsafe_allow_html=True)
        else:
            st.warning("Please make sure the uploaded CSV file has correct columns and formatting matching expected properties.")
    else:
        st.info("Upload a CSV file to get batch predictions.")

# ==========================================
# TAB 3: MODEL INFORMATION (CHARTS VISUALIZATION)
# ==========================================
with tab3:
    st.subheader("Model Performance Evaluation Chart")
    
    # Accuracy mappings collected from the training stages referenced
    data_dict = {
        "Decision Tree": 83.97,
        "Logistic Regression": 85.00,
        "Random Forest": 88.90,
        "Support Vector Machine": 86.20,
        "Grid RF": 89.90
    }
    
    models_list = list(data_dict.keys())
    accuracy_list = list(data_dict.values())
    
    chart_df = pd.DataFrame({
        "Models": models_list,
        "Accuracy": accuracy_list
    })
    
    # Constructing a dynamic interactive chart with Plotly
    fig = px.bar(chart_df, x="Models", y="Accuracy", labels={"Accuracy": "Accuracy Percentage (%)"})
    st.plotly_chart(fig)