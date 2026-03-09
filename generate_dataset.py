import pandas as pd
import numpy as np

np.random.seed(42)

n = 20000
data = []

for i in range(n):
    cancer = np.random.choice([0, 1], p=[0.7, 0.3])

    age = int(np.clip(np.random.normal(55 if cancer else 40, 12), 18, 90))

    PCOS = np.random.binomial(1, 0.15)
    endometriosis = np.random.binomial(1, 0.10)

    abdominal_pain = np.random.binomial(1, 0.7 if cancer else 0.2)
    nausea = np.random.binomial(1, 0.5 if cancer else 0.15)
    pelvic_pressure = np.random.binomial(1, 0.65 if cancer else 0.2)
    decreased_appetite = np.random.binomial(1, 0.6 if cancer else 0.15)
    abdominal_distension = np.random.binomial(1, 0.8 if cancer else 0.1)
    intermittent_lower_back_pain = np.random.binomial(1, 0.6 if cancer else 0.25)
    urinary_frequency = np.random.binomial(1, 0.55 if cancer else 0.2)
    previous_surgery = np.random.binomial(1, 0.25)
    bloating = np.random.binomial(1, 0.75 if cancer else 0.2)
    trouble_breathing = np.random.binomial(1, 0.3 if cancer else 0.05)
    pelvic_pain = np.random.binomial(1, 0.7 if cancer else 0.2)
    early_satiety = np.random.binomial(1, 0.7 if cancer else 0.1)

    menstrual_cycles = np.random.choice([0, 1, 2])

    BRCA_mutation = np.random.binomial(1, 0.08)
    colon_cancer_in_family = np.random.binomial(1, 0.10)

    hypertension = np.random.binomial(1, 0.35)
    hyperlipidemia = np.random.binomial(1, 0.30)

    systolic_bp = int(np.random.normal(135 if hypertension else 120, 10))
    diastolic_bp = int(np.random.normal(85 if hypertension else 75, 8))
    HR = int(np.random.normal(85 if cancer else 75, 10))
    BMI = round(np.random.normal(27, 5), 1)

    hemoglobin = round(np.random.normal(11 if cancer else 13.5, 1.2), 1)
    WBC = round(np.random.normal(9 if cancer else 7, 2), 1)
    platelets = round(np.random.normal(350 if cancer else 250, 70), 1)
    CA125 = max(0, round(np.random.normal(400 if cancer else 20, 120), 1))
    CEA = max(0, round(np.random.normal(6 if cancer else 2, 1.5), 1))

    mass_size = round(abs(np.random.normal(7 if cancer else 2, 2)), 1)
    solid_nodule_mass = np.random.binomial(1, 0.7 if cancer else 0.05)
    ascites = np.random.binomial(1, 0.6 if cancer else 0.02)
    omental_nodularity = np.random.binomial(1, 0.65 if cancer else 0.03)
    omental_thickening = np.random.binomial(1, 0.65 if cancer else 0.03)
    liver_metastasis = np.random.binomial(1, 0.15 if cancer else 0.005)
    omental_adhesion = np.random.binomial(1, 0.6 if cancer else 0.03)
    peritoneal_implants = np.random.binomial(1, 0.6 if cancer else 0.02)
    peritoneal_disease = np.random.binomial(1, 0.6 if cancer else 0.02)
    papillary_projections = np.random.binomial(1, 0.7 if cancer else 0.02)
    omental_metastasis = np.random.binomial(1, 0.6 if cancer else 0.01)

    row = {
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
        "hyperlipidemia": hyperlipidemia,
        "ovarian_cancer": cancer
    }

    data.append(row)

df = pd.DataFrame(data)
df.to_csv("ovarian_cancer_training_dataset_20000.csv", index=False)

print("Dataset created successfully")
print(df.shape)
print("\nCancer distribution:")
print(df["ovarian_cancer"].value_counts())