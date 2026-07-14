
# --- Import necessary libraries ---
import streamlit as st # Streamlit for creating interactive web applications
import joblib # joblib for loading pre-trained models and encoders
import pandas as pd # pandas for data manipulation
import numpy as np # numpy for numerical operations
from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder # For handling categorical features
from sklearn.impute import SimpleImputer # For handling missing values

# --- Load Model and Encoder ---
try:
    # Load the pre-trained machine learning model from 'disease_model.pkl'
    # This model will be used to make predictions based on user input.
    model = joblib.load('disease_model.pkl')
    # Load the pre-fitted LabelEncoder from 'label_encoder.pkl'
    # This encoder is crucial for converting numerical predictions back into human-readable disease names.
    label_encoder = joblib.load('label_encoder.pkl')
except FileNotFoundError:
    st.error("Error: Model or Label Encoder files not found. Make sure 'disease_model.pkl' and 'label_encoder.pkl' are in the same directory.")
    st.stop() # Stop the app if essential files are missing
except Exception as e:
    st.error(f"An error occurred while loading model or label encoder: {e}")
    st.stop()

# --- Data Preparation for MultiLabelBinarizer and Imputer ---
# The MultiLabelBinarizer and SimpleImputer objects were not saved with the model.
# We need to re-create and re-fit them using the original dataset to ensure
# that input processing for the app matches the training pipeline.

try:
    # Load the original dataset again to get the data needed for fitting the binarizers and imputer.
    # Make sure 'dataset.csv' is accessible by the Streamlit application (e.g., in the same directory).
   import os

dataset_path = os.path.join(os.path.dirname(__file__), "dataset.csv")
df_original = pd.read_csv(dataset_path)
except FileNotFoundError:
    st.error("Error: Original dataset file not found. Make sure 'dataset.csv' is in the same directory.")
    st.stop() # Stop the app if the dataset is missing
except Exception as e:
    st.error(f"An error occurred while loading the dataset: {e}")
    st.stop()

# Define the clean_and_split function (must be identical to the one used during training)
# This function processes comma-separated strings into lists of individual items.
def clean_and_split(text):
    if isinstance(text, str): # Check if the input is a string
        # Split by comma, remove leading/trailing whitespace, and filter out empty strings
        return [item.strip() for item in text.replace('nan', '').split(',') if item.strip()]
    return [] # Return an empty list for non-string inputs

# Re-apply the function to create lists for MultiLabelBinarizer for symptoms, cures, and doctors
df_original['symptoms_list'] = df_original['symptoms'].apply(clean_and_split)
df_original['cures_list'] = df_original['cures'].apply(clean_and_split)
df_original['doctor_list'] = df_original['doctor'].apply(clean_and_split)

# Re-create and fit MultiLabelBinarizer for symptoms
# This object converts a list of symptoms into a binary feature vector.
mlb_symptoms = MultiLabelBinarizer()
mlb_symptoms.fit(df_original['symptoms_list']) # Fit on the original symptoms list to learn all unique symptom classes

# Re-create and fit MultiLabelBinarizer for cures
# This object converts a list of cures into a binary feature vector.
mlb_cures = MultiLabelBinarizer()
mlb_cures.fit(df_original['cures_list']) # Fit on the original cures list to learn all unique cure classes

# Re-create and fit MultiLabelBinarizer for doctors
# This object converts a list of doctors into a binary feature vector.
mlb_doctor = MultiLabelBinarizer()
mlb_doctor.fit(df_original['doctor_list']) # Fit on the original doctor list to learn all unique doctor classes

# Re-create and fit the imputer for 'risk level_percentage'
# First, re-extract 'risk level_percentage' as it was during training
# This function extracts numerical percentages from the 'risk level' string.
def extract_risk_percentage(risk_str):
    if isinstance(risk_str, str):
        if 'varies' in risk_str.lower(): # If the string contains 'varies', treat as missing
            return np.nan # Use np.nan for consistency with pandas
        elif '(' in risk_str: # If parentheses are present, extract the number inside
            try:
                value_str = risk_str.split('(')[1].replace('%)', '').replace('%', '').strip()
                return float(value_str)
            except (IndexError, ValueError): # Handle potential parsing errors
                return np.nan
    return np.nan # Return NaN for other unexpected formats or non-string inputs

# Apply the extraction function to get the numerical risk level percentages
df_original['risk level_percentage'] = df_original['risk level'].apply(extract_risk_percentage)

# Initialize and fit the SimpleImputer with a median strategy
# This imputer fills in any None/NaN values with the median from the training data.
imputer = SimpleImputer(strategy='median')
# Fit only on non-null values to get a robust median, then store it.
# Use .values.reshape(-1, 1) to ensure the input is 2D for fit.
imputer.fit(df_original[['risk level_percentage']].dropna()) 
median_risk_level = imputer.statistics_[0] # Store the calculated median risk level

# Combine all unique feature names (columns) that the model was trained on.
# This list defines the expected input format for the model.
all_feature_names = (
    mlb_symptoms.classes_.tolist() +
    mlb_cures.classes_.tolist() +
    mlb_doctor.classes_.tolist() +
    ['risk level_percentage'] # Add the risk level percentage feature name
)

# --- Streamlit App Layout ---
st.set_page_config(layout="wide") # Configure the Streamlit page to use a wide layout
st.title('Medical Symptom Disease Prediction') # Set the main title of the application

st.markdown("""
This application uses a trained Machine Learning model to predict potential diseases based on user-selected symptoms, cures, doctors consulted, and an estimated risk level.
It's designed to demonstrate an ML workflow and should not be used for actual medical diagnosis.
""")

# --- Disclaimer ---
# Display a clear warning that the app is for educational purposes only.
st.warning('Disclaimer: This application is for educational purposes only and should NOT be used for medical diagnosis. Always consult a qualified healthcare professional for any health concerns.')

st.header('Provide Your Information:') # Section header for user input

# --- Input Widgets ---

# Multi-select widget for symptoms
selected_symptoms = st.multiselect(
    'Select Symptoms you are experiencing (choose all that apply):',
    options=mlb_symptoms.classes_.tolist(), # Options are all unique symptoms learned during training
    help='Choose from the list of symptoms that match your condition.'
)

# Multi-select widget for cures/treatments
selected_cures = st.multiselect(
    'Select Cures/Treatments you are currently undergoing or have used (choose all that apply):',
    options=mlb_cures.classes_.tolist(), # Options are all unique cures learned during training
    help='Choose from the list of treatments that apply to you.'
)

# Multi-select widget for doctor types consulted
selected_doctors = st.multiselect(
    'Select Doctor types you have consulted (choose all that apply):',
    options=mlb_doctor.classes_.tolist(), # Options are all unique doctor types learned during training
    help='Choose from the list of medical professionals you have seen.'
)

# Slider for Risk Level Percentage
# Determine dynamic min, max, and step values for the slider based on the original data
min_risk_val = df_original['risk level_percentage'].min()
max_risk_val = df_original['risk level_percentage'].max()

# Handle potential NaN or empty range for risk level
if pd.isna(min_risk_val): min_risk_val = 0.0
if pd.isna(max_risk_val): max_risk_val = 100.0

range_risk = max_risk_val - min_risk_val # Calculate the range of risk values
if range_risk == 0: # If range is zero (all values are the same), use a default step
    step_risk = 0.1
elif range_risk < 1: # If the range is very small, use a finer step
    step_risk = 0.01
else: # Otherwise, calculate a step that provides roughly 100 increments
    step_risk = round(range_risk / 100, 2)
    if step_risk == 0: step_risk = 0.1 # Ensure step is not zero after rounding

input_risk_level = st.slider(
    'Select Risk Level Percentage (e.g., from a medical report):',
    min_value=float(min_risk_val), # Minimum value for the slider
    max_value=float(max_risk_val), # Maximum value for the slider
    value=float(median_risk_level), # Default value is the median from the training data
    step=float(step_risk), # Step size for the slider
    help='This represents a numerical risk percentage associated with a condition, e.g., from a medical test. Defaults to the median value if not changed.'
)

st.write('---') # Horizontal rule for visual separation

# --- Prediction Logic ---
if st.button('Predict Disease'): # Button to trigger the prediction
    # Create a new DataFrame for input features, initialized with zeros
    # This DataFrame must have the exact same columns as the data the model was trained on.
    input_data = pd.DataFrame(0, index=[0], columns=all_feature_names)

    # Populate the input_data DataFrame based on user selections
    # For each selected symptom, set its corresponding column to 1
    for symptom in selected_symptoms:
        if symptom in input_data.columns: # Check if the symptom is a known feature
            input_data.loc[0, symptom] = 1

    # For each selected cure, set its corresponding column to 1
    for cure in selected_cures:
        if cure in input_data.columns: # Check if the cure is a known feature
            input_data.loc[0, cure] = 1

    # For each selected doctor type, set its corresponding column to 1
    for doctor_type in selected_doctors:
        if doctor_type in input_data.columns: # Check if the doctor type is a known feature
            input_data.loc[0, doctor_type] = 1

    # Set the 'risk level_percentage' column to the value selected by the user
    input_data.loc[0, 'risk level_percentage'] = input_risk_level

    try:
        # Make prediction using the loaded model
        prediction_encoded = model.predict(input_data) # Get the numerical prediction (encoded label)
        prediction_proba = model.predict_proba(input_data) # Get prediction probabilities for all classes

        # Convert the numerical prediction back to a human-readable disease name
        predicted_disease = label_encoder.inverse_transform(prediction_encoded)[0]

        # Get the confidence score for the predicted disease
        # Find the index of the predicted class (highest probability)
        predicted_class_idx = np.argmax(prediction_proba)
        # Retrieve the probability corresponding to the predicted class
        confidence_score = prediction_proba[0, predicted_class_idx]

        # --- Display Results ---
        st.success(f'## Predicted Disease: {predicted_disease}') # Display the predicted disease prominently
        st.info(f'### Confidence Score: {confidence_score:.2f}') # Display the confidence score (formatted to 2 decimal places)

        st.markdown('---')
        st.subheader('Understanding the Prediction:')
        st.write('Based on the features you provided (symptoms, cures, doctors consulted, and risk level), the model has identified the most probable disease.')
        st.write('The **Confidence Score** indicates the model\'s certainty in its prediction. A higher score means the model is more confident. However, remember this is a machine learning prediction and not a medical diagnosis.')
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
