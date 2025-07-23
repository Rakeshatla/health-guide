# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import pandas as pd
import pickle
import numpy as np
import difflib
import streamlit as st
#fv
# load model
svc = pickle.load(open('svc.pkl','rb'))
sym_des = pd.read_csv("archive/symtoms_df.csv")
precautions = pd.read_csv("archive/precautions_df.csv")
workout = pd.read_csv("archive/workout_df.csv")
description = pd.read_csv("archive/description.csv")
medications = pd.read_csv('archive/medications.csv')
diets = pd.read_csv("archive/diets.csv")
import difflib

def resolve_symptom(symptom, symptom_dict, cutoff=0.7):
    """Return closest known symptom or raise error if not found."""
    # Lowercase all comparisons for robustness
    symptom = symptom.strip().lower()
    valid_symptoms = list(symptom_dict.keys())
    # Try exact match first
    if symptom in valid_symptoms:
        return symptom
    # Fuzzy match
    match = difflib.get_close_matches(symptom, valid_symptoms, n=1, cutoff=cutoff)
    if match:
        # Optionally: ask user to confirm if you build CLI/app
        # e.g.: input(f"Did you mean '{match[0]}'? (y/n): ")
        return match[0]  # return the matched symptom
    else:
        raise ValueError(f"Symptom '{symptom}' not recognized. Please re-enter.")
def get_predicted_value_with_fuzzy_matching(patient_symptoms):
    input_vector = np.zeros(len(symptoms_dict))
    resolved_symptoms = []
    for item in patient_symptoms:
        try:
            resolved_symptom = resolve_symptom(item, symptoms_dict)
            input_vector[symptoms_dict[resolved_symptom]] = 1
            resolved_symptoms.append(resolved_symptom)
        except ValueError as e:
            print(e)  # Or handle in your app logic
            # Optionally, prompt user again here
    if not any(input_vector):
        print("No valid symptoms entered.")
        return None
    return diseases_list[svc.predict([input_vector])[0]], resolved_symptoms
#============================================================
# custome and helping functions
#==========================helper funtions================
def helper(dis):
    desc = description[description['Disease'] == dis]['Description']
    desc = " ".join([w for w in desc])

    pre = precautions[precautions['Disease'] == dis][['Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4']]
    pre = [col for col in pre.values]

    med = medications[medications['Disease'] == dis]['Medication']
    med = [med for med in med.values]

    die = diets[diets['Disease'] == dis]['Diet']
    die = [die for die in die.values]

    wrkout = workout[workout['disease'] == dis]['workout']

    return desc, pre, med, die, wrkout


symptoms_dict = {'itching': 0, 'skin_rash': 1, 'nodal_skin_eruptions': 2, 'continuous_sneezing': 3, 'shivering': 4, 'chills': 5, 'joint_pain': 6, 'stomach_pain': 7, 'acidity': 8, 'ulcers_on_tongue': 9, 'muscle_wasting': 10, 'vomiting': 11, 'burning_micturition': 12, 'spotting_ urination': 13, 'fatigue': 14, 'weight_gain': 15, 'anxiety': 16, 'cold_hands_and_feets': 17, 'mood_swings': 18, 'weight_loss': 19, 'restlessness': 20, 'lethargy': 21, 'patches_in_throat': 22, 'irregular_sugar_level': 23, 'cough': 24, 'high_fever': 25, 'sunken_eyes': 26, 'breathlessness': 27, 'sweating': 28, 'dehydration': 29, 'indigestion': 30, 'headache': 31, 'yellowish_skin': 32, 'dark_urine': 33, 'nausea': 34, 'loss_of_appetite': 35, 'pain_behind_the_eyes': 36, 'back_pain': 37, 'constipation': 38, 'abdominal_pain': 39, 'diarrhoea': 40, 'mild_fever': 41, 'yellow_urine': 42, 'yellowing_of_eyes': 43, 'acute_liver_failure': 44, 'fluid_overload': 45, 'swelling_of_stomach': 46, 'swelled_lymph_nodes': 47, 'malaise': 48, 'blurred_and_distorted_vision': 49, 'phlegm': 50, 'throat_irritation': 51, 'redness_of_eyes': 52, 'sinus_pressure': 53, 'runny_nose': 54, 'congestion': 55, 'chest_pain': 56, 'weakness_in_limbs': 57, 'fast_heart_rate': 58, 'pain_during_bowel_movements': 59, 'pain_in_anal_region': 60, 'bloody_stool': 61, 'irritation_in_anus': 62, 'neck_pain': 63, 'dizziness': 64, 'cramps': 65, 'bruising': 66, 'obesity': 67, 'swollen_legs': 68, 'swollen_blood_vessels': 69, 'puffy_face_and_eyes': 70, 'enlarged_thyroid': 71, 'brittle_nails': 72, 'swollen_extremeties': 73, 'excessive_hunger': 74, 'extra_marital_contacts': 75, 'drying_and_tingling_lips': 76, 'slurred_speech': 77, 'knee_pain': 78, 'hip_joint_pain': 79, 'muscle_weakness': 80, 'stiff_neck': 81, 'swelling_joints': 82, 'movement_stiffness': 83, 'spinning_movements': 84, 'loss_of_balance': 85, 'unsteadiness': 86, 'weakness_of_one_body_side': 87, 'loss_of_smell': 88, 'bladder_discomfort': 89, 'foul_smell_of urine': 90, 'continuous_feel_of_urine': 91, 'passage_of_gases': 92, 'internal_itching': 93, 'toxic_look_(typhos)': 94, 'depression': 95, 'irritability': 96, 'muscle_pain': 97, 'altered_sensorium': 98, 'red_spots_over_body': 99, 'belly_pain': 100, 'abnormal_menstruation': 101, 'dischromic _patches': 102, 'watering_from_eyes': 103, 'increased_appetite': 104, 'polyuria': 105, 'family_history': 106, 'mucoid_sputum': 107, 'rusty_sputum': 108, 'lack_of_concentration': 109, 'visual_disturbances': 110, 'receiving_blood_transfusion': 111, 'receiving_unsterile_injections': 112, 'coma': 113, 'stomach_bleeding': 114, 'distention_of_abdomen': 115, 'history_of_alcohol_consumption': 116, 'fluid_overload.1': 117, 'blood_in_sputum': 118, 'prominent_veins_on_calf': 119, 'palpitations': 120, 'painful_walking': 121, 'pus_filled_pimples': 122, 'blackheads': 123, 'scurring': 124, 'skin_peeling': 125, 'silver_like_dusting': 126, 'small_dents_in_nails': 127, 'inflammatory_nails': 128, 'blister': 129, 'red_sore_around_nose': 130, 'yellow_crust_ooze': 131}
diseases_list = {15: 'Fungal infection', 4: 'Allergy', 16: 'GERD', 9: 'Chronic cholestasis', 14: 'Drug Reaction', 33: 'Peptic ulcer diseae', 1: 'AIDS', 12: 'Diabetes ', 17: 'Gastroenteritis', 6: 'Bronchial Asthma', 23: 'Hypertension ', 30: 'Migraine', 7: 'Cervical spondylosis', 32: 'Paralysis (brain hemorrhage)', 28: 'Jaundice', 29: 'Malaria', 8: 'Chicken pox', 11: 'Dengue', 37: 'Typhoid', 40: 'hepatitis A', 19: 'Hepatitis B', 20: 'Hepatitis C', 21: 'Hepatitis D', 22: 'Hepatitis E', 3: 'Alcoholic hepatitis', 36: 'Tuberculosis', 10: 'Common Cold', 34: 'Pneumonia', 13: 'Dimorphic hemmorhoids(piles)', 18: 'Heart attack', 39: 'Varicose veins', 26: 'Hypothyroidism', 24: 'Hyperthyroidism', 25: 'Hypoglycemia', 31: 'Osteoarthristis', 5: 'Arthritis', 0: '(vertigo) Paroymsal  Positional Vertigo', 2: 'Acne', 38: 'Urinary tract infection', 35: 'Psoriasis', 27: 'Impetigo'}

# Model Prediction function
def get_predicted_value(patient_symptoms):
    input_vector = np.zeros(len(symptoms_dict))
    for item in patient_symptoms:
        input_vector[symptoms_dict[item]] = 1
    return diseases_list[svc.predict([input_vector])[0]]
# Use the improved get_predicted_value_with_fuzzy_matching from earlier

def disease_prediction(symptoms):
    symptoms = input("Enter your symptoms (comma-separated): ")
    user_symptoms = [s.strip().strip("[]' ") for s in symptoms.split(',')]

    predicted_disease, resolved_symptoms = get_predicted_value_with_fuzzy_matching(user_symptoms)

    if predicted_disease and resolved_symptoms:
        desc, pre, med, die, wrkout = helper(predicted_disease)

        print("\n=================predicted disease============")
        print(predicted_disease)
        print("Symptom(s) used:", ", ".join(resolved_symptoms))
        print("=================description==================")
        print(desc)
        print("=================precautions==================")
        for idx, p_i in enumerate(pre[0], 1):
            print(f"{idx}: {p_i}")

        print("=================medications==================")
        for idx, m_i in enumerate(med, len(pre[0]) + 1):
            print(f"{idx}: {m_i}")

        print("=================workout==================")
        for idx, w_i in enumerate(wrkout, len(pre[0]) + len(med) + 1):
            print(f"{idx}: {w_i}")

        print("=================diets==================")
        for idx, d_i in enumerate(die, len(pre[0]) + len(med) + len(wrkout) + 1):
            print(f"{idx}: {d_i}")
    else:
        print("No valid prediction was made due to invalid or unrecognized symptoms.")
def main():
    # ---- Stylish Header ----
    st.markdown("""
    # 🩺 Disease Prediction Health Guide
    #### Get instant health insights and lifestyle recommendations
    """)

    # --- Input Options in Columns ---
    col1, col2 = st.columns(2)
    with col1:
        user_input = st.text_input("📝 Enter symptoms (comma-separated):")
    with col2:
        symptoms_available = list(symptoms_dict.keys())
        selected_symptoms = st.multiselect("🔍 Or select symptoms:", symptoms_available)

    # --- Merge and Deduplicate Inputs ---
    input_set = set()
    if user_input:
        input_set.update([sym.strip().lower() for sym in user_input.split(",") if sym.strip()])
    if selected_symptoms:
        input_set.update([sym.strip().lower() for sym in selected_symptoms])
    symptoms_raw = list(input_set)

    # --- Predict Button and Spinner ---
    if st.button("🚦 Predict Disease"):
        if symptoms_raw:
            with st.spinner("Analyzing your symptoms..."):
                try:
                    predicted_disease, resolved = get_predicted_value_with_fuzzy_matching(symptoms_raw)
                    desc, pre, med, die, wrkout = helper(predicted_disease)
                    st.markdown(f"""<div style='background:linear-gradient(90deg,#6dd5ed,#2193b0);padding:1em 2em;border-radius:8px'>
                    <h2 style='color:#fff;'>{predicted_disease}</h2>
                    </div>""", unsafe_allow_html=True)
                    st.write(f"**Symptoms resolved:** {', '.join(resolved)}")
                    
                    with st.expander("📋 Description"):
                        st.write(desc)
                    with st.expander("⚠️ Precautions"):
                        for i, p in enumerate(pre[0], 1):
                            st.write(f"{i}. {p}")
                    with st.expander("💊 Medications"):
                        for i, m in enumerate(med, 1):
                            st.write(f"{i}. {m}")
                    with st.expander("🏃 Workouts"):
                        for i, w in enumerate(wrkout, 1):
                            st.write(f"{i}. {w}")
                    with st.expander("🥗 Diet"):
                        for i, d in enumerate(die, 1):
                            st.write(f"{i}. {d}")
                    st.success("Recommendation generated! For critical symptoms, consult a doctor.")
                except Exception as e:
                    st.error(f"An error occurred: {e}")
        else:
            st.warning("Please enter or select at least one symptom.")

    # --- Footer ---
    st.markdown("""
    ---
 
    """, unsafe_allow_html=True)

# To run the main function
if __name__ == "__main__":
    main()

    
