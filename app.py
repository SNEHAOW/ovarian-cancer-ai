import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load("xgboost_ovarian_model.pkl")

st.set_page_config(page_title="Ovarian Cancer AI Predictor", layout="wide")

st.title("AI Assisted Ovarian Cancer Risk Prediction")

st.markdown("""
This system evaluates ovarian cancer risk using symptoms, clinical history,
lab tests, and imaging findings.

⚠️ Educational demonstration only.
""")

# -------------------------
# SIDEBAR INPUT
# -------------------------

st.sidebar.header("Patient Case Information")

age = st.sidebar.number_input("Age", 18, 100, 50)

pcos = st.sidebar.selectbox("PCOS", ["No", "Yes"])
pcos = 1 if pcos == "Yes" else 0

endometriosis = st.sidebar.selectbox("Endometriosis", ["No", "Yes"])
endometriosis = 1 if endometriosis == "Yes" else 0

# -------------------------
# SYMPTOMS
# -------------------------

st.sidebar.subheader("Symptoms")

def yn(label):
    return 1 if st.sidebar.checkbox(label) else 0

abdominal_pain = yn("Abdominal Pain")
nausea = yn("Nausea")
pelvic_pressure = yn("Pelvic Pressure")
decreased_appetite = yn("Decreased Appetite")
abdominal_distension = yn("Abdominal Distension")
intermittent_lower_back_pain = yn("Lower Back Pain")
urinary_frequency = yn("Urinary Frequency")
previous_surgery = yn("Previous Surgery")
bloating = yn("Bloating")
trouble_breathing = yn("Trouble Breathing")
pelvic_pain = yn("Pelvic Pain")
early_satiety = yn("Early Satiety")

# -------------------------
# MENSTRUAL CYCLE
# -------------------------

menstrual = st.sidebar.selectbox(
    "Menstrual Cycle",
    ["Low Flow", "Regular Flow", "Heavy Flow"]
)

if menstrual == "Low Flow":
    menstrual_cycles = 0
elif menstrual == "Regular Flow":
    menstrual_cycles = 1
else:
    menstrual_cycles = 2

# -------------------------
# GENETIC RISK
# -------------------------

st.sidebar.subheader("Genetic Risk")

brca = st.sidebar.selectbox("BRCA Mutation Risk", ["Low Risk", "High Risk"])
BRCA_mutation = 1 if brca == "High Risk" else 0

colon_cancer_in_family = yn("Colon Cancer in Family")

# -------------------------
# VITALS
# -------------------------

st.sidebar.subheader("Vitals")

systolic_bp = st.sidebar.number_input("Systolic BP", 80, 200, 120)
diastolic_bp = st.sidebar.number_input("Diastolic BP", 40, 120, 80)

HR = st.sidebar.number_input("Heart Rate", 40, 200, 80)
BMI = st.sidebar.number_input("BMI", 10.0, 60.0, 25.0)

hypertension = st.sidebar.selectbox("Hypertension", ["No", "Yes"])
hypertension = 1 if hypertension == "Yes" else 0

# -------------------------
# LAB TESTS
# -------------------------

st.sidebar.subheader("Lab Tests")

hemoglobin = st.sidebar.number_input("Hemoglobin", 0.0, 25.0, 12.0)
WBC = st.sidebar.number_input("WBC", 0.0, 50.0, 7.0)
platelets = st.sidebar.number_input("Platelets", 0.0, 1000.0, 300.0)

CA125 = st.sidebar.number_input("CA-125", 0.0, 5000.0, 35.0)
CEA = st.sidebar.number_input("CEA", 0.0, 100.0, 2.0)

# -------------------------
# IMAGING
# -------------------------

st.sidebar.subheader("Imaging Findings")

mass_size = st.sidebar.number_input("Mass Size", 0.0, 30.0, 5.0)

solid_nodule_mass = yn("Solid Nodule Mass")
ascites = yn("Ascites")
omental_nodularity = yn("Omental Nodularity")
omental_thickening = yn("Omental Thickening")
liver_metastasis = yn("Liver Metastasis")
omental_adhesion = yn("Omental Adhesion")
peritoneal_implants = yn("Peritoneal Implants")
peritoneal_disease = yn("Peritoneal Disease")
papillary_projections = yn("Papillary Projections")
omental_metastasis = yn("Omental Metastasis")

# -------------------------
# COMORBIDITIES
# -------------------------

hyperlipidemia = yn("Hyperlipidemia")

# -------------------------
# MODEL INPUT
# -------------------------

input_data = pd.DataFrame([{
    "age": age,
    "PCOS": pcos,
    "endometriosis": endometriosis,
    "abdominal_pain": abdominal_pain,
    "nausea": nausea,
    "pelvic_pressure": pelvic_pressure,
    "decreased_appetite": decreased_appetite,
    "abdominal_distension": abdominal_distension,
    "intermittent_lower_back_pain": intermittent_lower_back_pain,
    "urinary_frequency": urinary_frequency,
    "previous_surgery": previous_surgery,
    "bloating": bloating,
    "trouble_breathing": trouble_breathing,
    "pelvic_pain": pelvic_pain,
    "menstrual_cycles": menstrual_cycles,
    "early_satiety": early_satiety,
    "BRCA_mutation": BRCA_mutation,
    "colon_cancer_in_family": colon_cancer_in_family,
    "hypertension": hypertension,
    "systolic_bp": systolic_bp,
    "diastolic_bp": diastolic_bp,
    "HR": HR,
    "BMI": BMI,
    "hemoglobin": hemoglobin,
    "WBC": WBC,
    "platelets": platelets,
    "CA125": CA125,
    "CEA": CEA,
    "mass_size": mass_size,
    "solid_nodule_mass": solid_nodule_mass,
    "ascites": ascites,
    "omental_nodularity": omental_nodularity,
    "omental_thickening": omental_thickening,
    "liver_metastasis": liver_metastasis,
    "omental_adhesion": omental_adhesion,
    "peritoneal_implants": peritoneal_implants,
    "peritoneal_disease": peritoneal_disease,
    "papillary_projections": papillary_projections,
    "omental_metastasis": omental_metastasis,
    "hyperlipidemia": hyperlipidemia
}])

# -------------------------
# PREDICTION
# -------------------------

st.subheader("Prediction")

if st.button("Run AI Prediction"):

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    if prediction == 1:
        st.error("High Risk: Ovarian Cancer Likely")
    else:
        st.success("Low Risk: Ovarian Cancer Unlikely")

    st.write(f"Predicted Cancer Probability: **{probability:.2%}**")