import streamlit as st
import pandas as pd
import joblib

# -----------------------
# LOAD MODEL
# -----------------------

model = joblib.load("xgboost_ovarian_model.pkl")

st.set_page_config(page_title="Ovarian Cancer AI Predictor", layout="wide")

st.title("AI Assisted Ovarian Cancer Risk Prediction")

st.markdown("""
This system estimates ovarian cancer risk using symptoms, clinical history,
lab tests, and imaging findings.

⚠️ Educational use only. Any subtype shown below is a **possible pattern suggestion only**
and not a confirmed diagnosis.
""")

# -----------------------
# SIDEBAR INPUTS
# -----------------------

st.sidebar.header("Patient Information")

age = st.sidebar.number_input("Age", 18, 100, 50)

PCOS = 1 if st.sidebar.checkbox("PCOS") else 0
endometriosis = 1 if st.sidebar.checkbox("Endometriosis") else 0

# -----------------------
# SYMPTOMS
# -----------------------

st.sidebar.subheader("Symptoms")

abdominal_pain = 1 if st.sidebar.checkbox("Abdominal Pain") else 0
nausea = 1 if st.sidebar.checkbox("Nausea") else 0
pelvic_pressure = 1 if st.sidebar.checkbox("Pelvic Pressure") else 0
decreased_appetite = 1 if st.sidebar.checkbox("Decreased Appetite") else 0
abdominal_distension = 1 if st.sidebar.checkbox("Abdominal Distension") else 0
intermittent_lower_back_pain = 1 if st.sidebar.checkbox("Lower Back Pain") else 0
urinary_frequency = 1 if st.sidebar.checkbox("Urinary Frequency") else 0
previous_surgery = 1 if st.sidebar.checkbox("Previous Surgery") else 0
bloating = 1 if st.sidebar.checkbox("Bloating") else 0
trouble_breathing = 1 if st.sidebar.checkbox("Trouble Breathing") else 0
pelvic_pain = 1 if st.sidebar.checkbox("Pelvic Pain") else 0
early_satiety = 1 if st.sidebar.checkbox("Early Satiety") else 0

menstrual_cycles = st.sidebar.selectbox(
    "Menstrual Cycle",
    [0, 1, 2],
    help="0 = low/irregular, 1 = regular, 2 = heavy"
)

# -----------------------
# GENETIC
# -----------------------

st.sidebar.subheader("Genetic Risk")

BRCA_mutation = 1 if st.sidebar.checkbox("BRCA Mutation") else 0
colon_cancer_in_family = 1 if st.sidebar.checkbox("Colon Cancer in Family") else 0

# -----------------------
# COMORBIDITIES
# -----------------------

st.sidebar.subheader("Comorbidities")

hypertension = 1 if st.sidebar.checkbox("Hypertension") else 0
hyperlipidemia = 1 if st.sidebar.checkbox("Hyperlipidemia") else 0

# -----------------------
# VITALS
# -----------------------

st.sidebar.subheader("Vitals")

systolic_bp = st.sidebar.number_input("Systolic BP", 80, 200, 120)
diastolic_bp = st.sidebar.number_input("Diastolic BP", 40, 120, 80)

HR = st.sidebar.number_input("Heart Rate", 40, 200, 80)
BMI = st.sidebar.number_input("BMI", 10.0, 60.0, 25.0)

# -----------------------
# BLOOD TESTS
# -----------------------

st.sidebar.subheader("Blood Tests")

hemoglobin = st.sidebar.number_input("Hemoglobin", 0.0, 25.0, 12.0)
WBC = st.sidebar.number_input("WBC", 0.0, 50.0, 7.0)
platelets = st.sidebar.number_input("Platelets", 0.0, 1000.0, 300.0)

CA125 = st.sidebar.number_input("CA-125", 0.0, 5000.0, 35.0)
CEA = st.sidebar.number_input("CEA", 0.0, 100.0, 2.0)

# -----------------------
# IMAGING
# -----------------------

st.sidebar.subheader("Imaging")

mass_size = st.sidebar.number_input("Mass Size", 0.0, 30.0, 5.0)

solid_nodule_mass = 1 if st.sidebar.checkbox("Solid Nodule Mass") else 0
ascites = 1 if st.sidebar.checkbox("Ascites") else 0
omental_nodularity = 1 if st.sidebar.checkbox("Omental Nodularity") else 0
omental_thickening = 1 if st.sidebar.checkbox("Omental Thickening") else 0
liver_metastasis = 1 if st.sidebar.checkbox("Liver Metastasis") else 0
omental_adhesion = 1 if st.sidebar.checkbox("Omental Adhesion") else 0
peritoneal_implants = 1 if st.sidebar.checkbox("Peritoneal Implants") else 0
peritoneal_disease = 1 if st.sidebar.checkbox("Peritoneal Disease") else 0
papillary_projections = 1 if st.sidebar.checkbox("Papillary Projections") else 0
omental_metastasis = 1 if st.sidebar.checkbox("Omental Metastasis") else 0

# -----------------------
# CREATE DATAFRAME
# -----------------------

input_data = pd.DataFrame([{
    "age": age,
    "PCOS": PCOS,
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

# -----------------------
# FORCE FEATURE ORDER
# -----------------------

model_features = [
    'age','PCOS','endometriosis','abdominal_pain','nausea','pelvic_pressure',
    'decreased_appetite','abdominal_distension','intermittent_lower_back_pain',
    'urinary_frequency','previous_surgery','bloating','trouble_breathing',
    'pelvic_pain','menstrual_cycles','early_satiety','BRCA_mutation',
    'colon_cancer_in_family','hypertension','systolic_bp','diastolic_bp','HR',
    'BMI','hemoglobin','WBC','platelets','CA125','CEA','mass_size',
    'solid_nodule_mass','ascites','omental_nodularity','omental_thickening',
    'liver_metastasis','omental_adhesion','peritoneal_implants',
    'peritoneal_disease','papillary_projections','omental_metastasis',
    'hyperlipidemia'
]

input_data = input_data[model_features]

# -----------------------
# SUBTYPE HEURISTIC
# -----------------------

def suggest_ovarian_cancer_subtype(row):
    hgsc = 0
    endometrioid = 0
    mucinous = 0
    clear_cell = 0

    # High-grade serous carcinoma
    if row["age"] >= 50:
        hgsc += 2
    if row["BRCA_mutation"] == 1:
        hgsc += 3
    if row["CA125"] >= 200:
        hgsc += 3
    elif row["CA125"] >= 35:
        hgsc += 1
    if row["ascites"] == 1:
        hgsc += 2
    if row["omental_nodularity"] == 1:
        hgsc += 2
    if row["omental_thickening"] == 1:
        hgsc += 2
    if row["peritoneal_implants"] == 1:
        hgsc += 2
    if row["peritoneal_disease"] == 1:
        hgsc += 2
    if row["papillary_projections"] == 1:
        hgsc += 2
    if row["omental_metastasis"] == 1:
        hgsc += 2
    if row["liver_metastasis"] == 1:
        hgsc += 2
    if row["trouble_breathing"] == 1:
        hgsc += 1

    # Endometrioid carcinoma
    if row["endometriosis"] == 1:
        endometrioid += 3
    if row["pelvic_pain"] == 1:
        endometrioid += 2
    if row["abdominal_pain"] == 1:
        endometrioid += 1
    if row["menstrual_cycles"] != 1:
        endometrioid += 2
    if row["CA125"] >= 35:
        endometrioid += 1
    if row["ascites"] == 0:
        endometrioid += 1
    if row["omental_nodularity"] == 0:
        endometrioid += 1
    if row["peritoneal_disease"] == 0:
        endometrioid += 1

    # Mucinous carcinoma
    if row["mass_size"] >= 10:
        mucinous += 3
    elif row["mass_size"] >= 7:
        mucinous += 2
    if row["CEA"] >= 5:
        mucinous += 3
    elif row["CEA"] >= 2.5:
        mucinous += 1
    if row["CA125"] < 200:
        mucinous += 1
    if row["ascites"] == 0:
        mucinous += 1
    if row["omental_nodularity"] == 0:
        mucinous += 1
    if row["peritoneal_disease"] == 0:
        mucinous += 1
    if row["solid_nodule_mass"] == 0:
        mucinous += 1
    if row["abdominal_distension"] == 1:
        mucinous += 1

    # Clear cell carcinoma
    if row["endometriosis"] == 1:
        clear_cell += 3
    if row["pelvic_pain"] == 1:
        clear_cell += 2
    if row["solid_nodule_mass"] == 1:
        clear_cell += 2
    if row["mass_size"] >= 6:
        clear_cell += 1
    if row["CA125"] >= 35:
        clear_cell += 1
    if row["ascites"] == 0:
        clear_cell += 1
    if row["papillary_projections"] == 0:
        clear_cell += 1
    if row["age"] >= 40:
        clear_cell += 1
    if row["hemoglobin"] < 12:
        clear_cell += 1

    scores = {
        "High-grade serous carcinoma": hgsc,
        "Endometrioid carcinoma": endometrioid,
        "Mucinous carcinoma": mucinous,
        "Clear cell carcinoma": clear_cell
    }

    best_type = max(scores, key=scores.get)
    return best_type, scores

# -----------------------
# PREDICTION
# -----------------------

st.subheader("Prediction")

if st.button("Run AI Prediction"):
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    if prediction == 1:
        st.error("High Risk: Ovarian Malignancy Likely")

        subtype, scores = suggest_ovarian_cancer_subtype(input_data.iloc[0])

        st.markdown("### Possible Ovarian Cancer Type")
        st.info(subtype)

        st.caption(
            "This is a symptom/imaging-based subtype suggestion only for demo purposes. "
            "Actual subtype confirmation requires pathology and specialist evaluation."
        )

    else:
        st.success("Low Risk: Ovarian Malignancy Unlikely")

    st.write(f"Predicted Cancer Probability: **{probability:.2%}**")