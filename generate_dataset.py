import pandas as pd
import numpy as np

np.random.seed(42)

n = 1000

data = []

for i in range(n):

    age = np.random.randint(20,85)

    CA125 = np.random.normal(30,20)
    CEA = np.random.normal(3,2)

    mass_size = abs(np.random.normal(3,3))

    ascites = np.random.binomial(1,0.2)
    omental_metastasis = np.random.binomial(1,0.1)
    peritoneal_implants = np.random.binomial(1,0.1)
    papillary_projections = np.random.binomial(1,0.15)

    pelvic_pain = np.random.binomial(1,0.3)
    bloating = np.random.binomial(1,0.25)
    early_satiety = np.random.binomial(1,0.2)
    urinary_frequency = np.random.binomial(1,0.2)

    BRCA_mutation = np.random.binomial(1,0.1)

    # cancer probability logic
    cancer_score = 0

    if age > 50:
        cancer_score += 1
    if CA125 > 35:
        cancer_score += 2
    if mass_size > 5:
        cancer_score += 2
    if ascites == 1:
        cancer_score += 2
    if omental_metastasis == 1:
        cancer_score += 3
    if peritoneal_implants == 1:
        cancer_score += 3
    if papillary_projections == 1:
        cancer_score += 2
    if BRCA_mutation == 1:
        cancer_score += 2

    cancer = 1 if cancer_score >= 4 else 0

    row = {

        "age": age,
        "PCOS": np.random.binomial(1,0.1),
        "endometriosis": np.random.binomial(1,0.15),
        "abdominal_pain": pelvic_pain,
        "nausea": np.random.binomial(1,0.2),
        "pelvic_pressure": np.random.binomial(1,0.2),
        "decreased_appetite": np.random.binomial(1,0.2),
        "abdominal_distension": bloating,
        "intermittent_lower_back_pain": np.random.binomial(1,0.2),
        "urinary_frequency": urinary_frequency,
        "previous_surgery": np.random.binomial(1,0.2),
        "bloating": bloating,
        "trouble_breathing": np.random.binomial(1,0.1),
        "pelvic_pain": pelvic_pain,
        "menstrual_cycles": np.random.randint(0,3),
        "early_satiety": early_satiety,
        "BRCA_mutation": BRCA_mutation,
        "colon_cancer_in_family": np.random.binomial(1,0.1),
        "hypertension": np.random.binomial(1,0.2),
        "HR": np.random.normal(75,10),
        "BMI": np.random.normal(25,5),
        "hemoglobin": np.random.normal(12,2),
        "WBC": np.random.normal(7,2),
        "platelets": np.random.normal(300,80),
        "CA125": CA125,
        "CEA": CEA,
        "mass_size": mass_size,
        "solid_nodule_mass": np.random.binomial(1,0.2),
        "ascites": ascites,
        "omental_nodularity": np.random.binomial(1,0.1),
        "omental_thickening": np.random.binomial(1,0.1),
        "liver_metastasis": np.random.binomial(1,0.05),
        "omental_adhesion": np.random.binomial(1,0.1),
        "peritoneal_implants": peritoneal_implants,
        "omental_metastasis": omental_metastasis,
        "peritoneal_disease": np.random.binomial(1,0.1),
        "papillary_projections": papillary_projections,
        "hyperlipidemia": np.random.binomial(1,0.2),
        "Cancer": cancer
    }

    data.append(row)

df = pd.DataFrame(data)

df.to_csv("ovarian_cancer_training_dataset_1000.csv",index=False)

print("Dataset generated successfully.")
print(df["Cancer"].value_counts())