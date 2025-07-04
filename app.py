import streamlit as st
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report

# Set page config for better appearance
st.set_page_config(page_title="Bank Transaction Fraud Detection", layout="wide")

st.title("Bank Transaction Fraud Detection")
st.write("This application uses a hybrid machine learning model to predict fraudulent transactions.")

# --- Model Loading ---
@st.cache_resource # Cache the model loading for efficiency
def load_models():
    """
    Loads the trained models and preprocessors from the 'trained_models' directory.
    Uses st.cache_resource to avoid reloading on each app rerun.
    """
    model_dir = "trained_models" # Assuming models are in a 'trained_models' directory relative to app.py
    try:
        # Load the best Random Forest model pipeline (includes preprocessor)
        best_rf_model = joblib.load(f"{model_dir}/best_random_forest_model.joblib")
        # Load the scaler used for Isolation Forest (needed for anomaly score calculation on new data)
        iso_forest_scaler = joblib.load(f"{model_dir}/scaler.joblib")
         # Load the Isolation Forest model
        best_iso_forest = joblib.load(f"{model_dir}/best_isolation_forest_model.joblib")

        st.success("Models and preprocessors loaded successfully!")
        return best_rf_model, iso_forest_scaler, best_iso_forest
    except FileNotFoundError:
        st.error(f"Error: Model files not found in '{model_dir}'. Please ensure they are in the correct directory.")
        return None, None, None
    except Exception as e:
        st.error(f"An error occurred while loading models: {e}")
        return None, None, None,
# Load models and preprocessors
best_rf_model, iso_forest_scaler, best_iso_forest = load_models()


# --- Prediction Function ---
def predict_fraud(features, rf_model, iso_scaler, iso_model):
    """
    Makes a fraud prediction using the loaded hybrid model pipeline.
    First calculates anomaly features using the Isolation Forest model and scaler,
    then uses the Random Forest pipeline for the final prediction.
    """
    if rf_model is None or iso_scaler is None or iso_model is None:
        return "Model not loaded."


    expected_columns = [
        'Gender', 'Age', 'State', 'City', 'Bank_Branch', 'Account_Type',
        'Transaction_Date', 'Transaction_Time', 'Transaction_Amount',
        'Merchant_ID', 'Transaction_Type', 'Merchant_Category', 'Account_Balance',
        'Transaction_Device', 'Transaction_Location', 'Device_Type',
        'Transaction_Currency', 'Transaction_Description'
    ]

    # Create a DataFrame from input features
    input_data = pd.DataFrame([features], columns=expected_columns)

    # --- Calculate Anomaly Features for the new transaction ---

    iso_numerical_cols = ['Age', 'Transaction_Amount', 'Account_Balance'] # Based on how Isolation Forest was trained

    # Check if all required numerical columns are in the input data
    if not all(col in input_data.columns for col in iso_numerical_cols):
        st.error("Input data is missing required numerical columns for anomaly detection.")
        return "Error during prediction."

    try:
        # Scale the numerical data using the loaded scaler
        input_numerical_scaled = iso_scaler.transform(input_data[iso_numerical_cols])
        input_numerical_scaled_df = pd.DataFrame(input_numerical_scaled, columns=iso_numerical_cols)

        # Predict anomaly score and outlier label using the loaded Isolation Forest model
        anomaly_score = iso_model.decision_function(input_numerical_scaled_df)[0]
        is_outlier_iso_forest = iso_model.predict(input_numerical_scaled_df)[0]

        # Add the calculated anomaly features to the input DataFrame
        input_data['anomaly_score'] = anomaly_score
        input_data['is_outlier_iso_forest'] = is_outlier_iso_forest

    except Exception as e:
        st.error(f"An error occurred during anomaly feature calculation: {e}")
        return "Error during prediction."


    # --- Make final prediction using the hybrid Random Forest pipeline ---

    try:
        prediction = rf_model.predict(input_data)
        return prediction[0]
    except Exception as e:
        st.error(f"An error occurred during final prediction: {e}")
        return "Error during prediction."


# --- User Interface for Input ---
st.sidebar.header('Input Transaction Details')

# Define the features to be collected from user input (excluding identifiers and generated anomaly features)
input_features_list = [
    'Gender', 'Age', 'State', 'City', 'Bank_Branch', 'Account_Type',
    'Transaction_Date', 'Transaction_Time', 'Transaction_Amount',
    'Merchant_ID', 'Transaction_Type', 'Merchant_Category', 'Account_Balance',
    'Transaction_Device', 'Transaction_Location', 'Device_Type',
    'Transaction_Currency', 'Transaction_Description'
]

input_features = {}
for feature in input_features_list:
    if feature == 'Gender':
        input_features[feature] = st.sidebar.selectbox(f'Select {feature}', ['Male', 'Female'])
    elif feature in ['Age', 'Transaction_Amount', 'Account_Balance']:
        input_features[feature] = st.sidebar.number_input(f'Enter {feature}', value=0.0, format="%.2f")
    # Add more specific widget types for other features as needed (e.g., date input, time input)
    else:
        input_features[feature] = st.sidebar.text_input(f'Enter {feature}')

# Add a button to trigger prediction
if st.sidebar.button('Predict Fraud'):
    # Pass the loaded models and scaler to the predict_fraud function
    prediction = predict_fraud(input_features, best_rf_model, iso_forest_scaler, best_iso_forest)

    # Display the prediction result
    st.header('Prediction Result')
    if isinstance(prediction, (np.integer, int)):
        if prediction == 1:
            st.error("This transaction is predicted as FRAUDULENT.")
        elif prediction == 0:
            st.success("This transaction is predicted as NOT FRAUDULENT.")
        else:
             st.warning(f"Prediction result: {prediction}")

    else:
        st.warning(f"Prediction result: {prediction}")


# --- Evaluation Visualizations ---
st.header("Model Evaluation Visualizations")

@st.cache_data # Cache the evaluation data and results
def prepare_evaluation_data(data_path, model):
    """
    Loads evaluation data, prepares it, and makes predictions for visualization.
    Uses st.cache_data to avoid re-running data processing on each app rerun.
    """
    try:
        hybrid_data_eval = pd.read_csv(data_path)
        st.success("Hybrid data loaded for evaluation visualization.")

        features_to_exclude_eval = ['Is_Fraud', 'Customer_ID', 'Customer_Name', 'Transaction_ID', 'Customer_Contact', 'Customer_Email']
        X_hybrid_eval = hybrid_data_eval.drop(columns=features_to_exclude_eval)
        y_hybrid_eval = hybrid_data_eval['Is_Fraud']

        # Split data for evaluation (using the same random_state as training for consistency)
        X_train_eval, X_test_hybrid_eval, y_train_eval, y_test_hybrid_eval = train_test_split(
            X_hybrid_eval, y_hybrid_eval, test_size=0.2, random_state=42, stratify=y_hybrid_eval
        )

        if model:
            # The loaded model is a pipeline, just apply it to the test data
            y_pred_hybrid_eval = model.predict(X_test_hybrid_eval)
            y_pred_prob_hybrid_eval = model.predict_proba(X_test_hybrid_eval)[:, 1]
            return y_test_hybrid_eval, y_pred_hybrid_eval, y_pred_prob_hybrid_eval
        else:
            st.warning("Model not loaded, cannot prepare evaluation data.")
            return None, None, None

    except FileNotFoundError:
        st.error(f"Error loading hybrid data for evaluation visualization from {data_path}.")
        return None, None, None
    except Exception as e:
        st.error(f"An error occurred during evaluation data preparation: {e}")
        return None, None, None

# Define the path to the hybrid dataset for evaluation
hybrid_data_path_eval = "data/undersampled_fraud_data_with_anomalies.csv" # Adjust path if needed

# Prepare the evaluation data and get predictions
y_test_eval, y_pred_eval, y_pred_prob_eval = prepare_evaluation_data(hybrid_data_path_eval, best_rf_model)

if y_test_eval is not None and y_pred_eval is not None and y_pred_prob_eval is not None:
    # Define plotting functions within the app for clarity or import them
    def plot_confusion_matrix(y_true, y_pred):
        """Generates and displays a confusion matrix."""
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        return plt.gcf() # Return the current figure

    def plot_roc_curve(y_true, y_pred_prob):
        """Generates and displays an ROC curve."""
        fpr, tpr, thresholds = roc_curve(y_true, y_pred_prob)
        roc_auc = auc(fpr, tpr)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        return plt.gcf() # Return the current figure

    # Display the confusion matrix
    st.subheader("Confusion Matrix")
    fig_cm = plot_confusion_matrix(y_test_eval, y_pred_eval)
    st.pyplot(fig_cm)

    # Display the ROC curve
    st.subheader("ROC Curve")
    fig_roc = plot_roc_curve(y_test_eval, y_pred_prob_eval)
    st.pyplot(fig_roc)

    # Display classification report as text
    st.subheader("Classification Report")
    report_text = classification_report(y_test_eval, y_pred_eval)
    st.text(report_text)

else:
    st.warning("Could not load data or model for evaluation visualizations.")

# --- Add a footer or additional info ---
st.markdown("---")
st.markdown("Developed as a demonstration for Bank Transaction Fraud Detection.")
import joblib
import pandas as pd
import numpy as np

# Load the trained models and preprocessors
try:
    best_rf_model = joblib.load("trained_models/best_random_forest_model.joblib")
    preprocessor = joblib.load("trained_models/preprocessor.joblib")
    scaler = joblib.load("trained_models/scaler.joblib")
    st.success("Models and preprocessors loaded successfully!")
except FileNotFoundError:
    st.error("Error loading models. Make sure 'trained_models' directory and its contents exist.")
    best_rf_model = None
    preprocessor = None
    scaler = None


def predict_fraud(features, preprocessor, scaler, model):
    """Makes a fraud prediction using the loaded models."""
    if model is None or preprocessor is None or scaler is None:
        return "Model not loaded."

    # Create a DataFrame from input features
    input_data = pd.DataFrame([features])


    # Separate numerical and categorical columns for scaling and one-hot encoding
    numerical_cols_loaded = [col for col in input_data.columns if input_data[col].dtype in [np.int64, np.float64]]
    categorical_cols_loaded = [col for col in input_data.columns if input_data[col].dtype == 'object']

    # Apply scaling to numerical columns
    input_data_numerical_scaled = scaler.transform(input_data[numerical_cols_loaded])
    input_data[numerical_cols_loaded] = input_data_numerical_scaled

    input_data_processed = preprocessor.transform(input_data)

    # Make prediction
    prediction = model.predict(input_data_processed)

    return prediction[0]

st.sidebar.header('Input Transaction Details')

# Define the features to be collected from user input, excluding those handled automatically
input_features_list = [
    'Gender', 'Age', 'State', 'City', 'Bank_Branch', 'Account_Type',
    'Transaction_Amount', 'Merchant_Category', 'Account_Balance',
    'Transaction_Device', 'Device_Type', 'Transaction_Currency',
    'anomaly_score', 'is_outlier_iso_forest'
]

# Create input widgets for each feature
input_features = {}
for feature in input_features_list:
    # Create a base key by replacing spaces/special chars
    base_key = feature.lower().replace(' ', '_').replace('-', '_')
    
    if feature == 'Gender':
        input_features[feature] = st.sidebar.selectbox(
            f'Select {feature}', 
            ['Male', 'Female'],
            key=f"gender_{base_key}"  # Unique key
        )
    elif feature in ['Age', 'Transaction_Amount', 'Account_Balance', 'anomaly_score']:
        input_features[feature] = st.sidebar.number_input(
            f'Enter {feature}', 
            value=0.0,
            key=f"number_{base_key}"  # Unique key
        )
    elif feature == 'is_outlier_iso_forest':
        input_features[feature] = st.sidebar.selectbox(
            f'Is Outlier (Isolation Forest)?', 
            [-1, 1], 
            format_func=lambda x: 'Yes' if x == 1 else 'No',
            key=f"outlier_{base_key}"  # Unique key
        )
    else:
        input_features[feature] = st.sidebar.text_input(
            f'Enter {feature}',
            key=f"text_{base_key}"  # Unique key
        )

# Add a button to trigger prediction
if st.sidebar.button('Predict Fraud'):
    # Make prediction
    prediction = predict_fraud(input_features, preprocessor, scaler, best_rf_model)

    # Display the prediction result
    st.header('Prediction Result')
    if prediction == 1:
        st.error("This transaction is predicted as FRAUDULENT.")
    else:
        st.success("This transaction is predicted as NOT FRAUDULENT.")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc

def plot_confusion_matrix(y_true, y_pred):
    """Generates and displays a confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    return plt.gcf() # Return the current figure


def plot_roc_curve(y_true, y_pred_prob):
    """Generates and displays an ROC curve."""
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_prob)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    return plt.gcf() # Return the current figure


# --- Add a section to display evaluation metrics ---
st.header("Model Evaluation Visualizations")

try:
    hybrid_data_eval = pd.read_csv("/content/undersampled_fraud_data_with_anomalies.csv")
    st.success("Hybrid data loaded for evaluation visualization.")

    # Prepare data for evaluation visualization
    features_to_exclude_eval = ['Is_Fraud', 'Customer_ID', 'Customer_Name', 'Transaction_ID', 'Customer_Contact', 'Customer_Email']
    X_hybrid_eval = hybrid_data_eval.drop(columns=features_to_exclude_eval)
    y_hybrid_eval = hybrid_data_eval['Is_Fraud']

    X_train_eval, X_test_hybrid_eval, y_train_eval, y_test_hybrid_eval = train_test_split(
        X_hybrid_eval, y_hybrid_eval, test_size=0.2, random_state=42, stratify=y_hybrid_eval
    )

    # Make predictions on the test set using the loaded model
    if best_rf_model and preprocessor:
        hybrid_numerical_cols_eval = X_test_hybrid_eval.select_dtypes(include=np.number).columns.tolist()
        hybrid_categorical_cols_eval = X_test_hybrid_eval.select_dtypes(include=['object']).columns.tolist()

        hybrid_preprocessor_eval = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), hybrid_numerical_cols_eval),
                ('cat', OneHotEncoder(handle_unknown='ignore'), hybrid_categorical_cols_eval)
            ],
            remainder='passthrough'
        )

        hybrid_preprocessor_eval.fit(X_train_eval)

        # Transform the test data
        X_test_hybrid_processed = hybrid_preprocessor_eval.transform(X_test_hybrid_eval)


        y_pred_hybrid_eval = best_rf_model.predict(X_test_hybrid_processed)
        y_pred_prob_hybrid_eval = best_rf_model.predict_proba(X_test_hybrid_processed)[:, 1]

        # Display the confusion matrix
        st.subheader("Confusion Matrix")
        fig_cm = plot_confusion_matrix(y_test_hybrid_eval, y_pred_hybrid_eval)
        st.pyplot(fig_cm)

        # Display the ROC curve
        st.subheader("ROC Curve")
        fig_roc = plot_roc_curve(y_test_hybrid_eval, y_pred_prob_hybrid_eval)
        st.pyplot(fig_roc)

    else:
        st.warning("Models not loaded, cannot display evaluation visualizations.")

except FileNotFoundError:
    st.error("Error loading hybrid data for evaluation visualization.")
except Exception as e:
    st.error(f"An error occurred during evaluation visualization: {e}")

# --- Add a section to display evaluation metrics ---
st.header("Model Evaluation Visualizations")

try:
    hybrid_data_eval = pd.read_csv("/content/undersampled_fraud_data_with_anomalies.csv")
    st.success("Hybrid data loaded for evaluation visualization.")

    features_to_exclude_eval = ['Is_Fraud', 'Customer_ID', 'Customer_Name', 'Transaction_ID', 'Customer_Contact', 'Customer_Email']
    X_hybrid_eval = hybrid_data_eval.drop(columns=features_to_exclude_eval)
    y_hybrid_eval = hybrid_data_eval['Is_Fraud']

    X_train_eval, X_test_hybrid_eval, y_train_eval, y_test_hybrid_eval = train_test_split(
        X_hybrid_eval, y_hybrid_eval, test_size=0.2, random_state=42, stratify=y_hybrid_eval
    )

    if best_rf_model and preprocessor:
        hybrid_numerical_cols_eval = X_test_hybrid_eval.select_dtypes(include=np.number).columns.tolist()
        hybrid_categorical_cols_eval = X_test_hybrid_eval.select_dtypes(include=['object']).columns.tolist()

        hybrid_preprocessor_eval = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), hybrid_numerical_cols_eval),
                ('cat', OneHotEncoder(handle_unknown='ignore'), hybrid_categorical_cols_eval)
            ],
            remainder='passthrough'
        )


        hybrid_preprocessor_eval.fit(X_train_eval)

        # Transform the test data
        X_test_hybrid_processed = hybrid_preprocessor_eval.transform(X_test_hybrid_eval)


        # Predict using the loaded best Random Forest model
        y_pred_hybrid_eval = best_rf_model.predict(X_test_hybrid_processed)
        # Get prediction probabilities for ROC curve
        y_pred_prob_hybrid_eval = best_rf_model.predict_proba(X_test_hybrid_processed)[:, 1]

        # Display the confusion matrix
        st.subheader("Confusion Matrix")
        fig_cm = plot_confusion_matrix(y_test_hybrid_eval, y_pred_hybrid_eval)
        st.pyplot(fig_cm)

        # Display the ROC curve
        st.subheader("ROC Curve")
        fig_roc = plot_roc_curve(y_test_hybrid_eval, y_pred_prob_hybrid_eval)
        st.pyplot(fig_roc)

    else:
        st.warning("Models not loaded, cannot display evaluation visualizations.")

except FileNotFoundError:
    st.error("Error loading hybrid data for evaluation visualization.")
except Exception as e:
    st.error(f"An error occurred during evaluation visualization: {e}")
