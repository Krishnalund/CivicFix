import streamlit as st
import pandas as pd
import joblib
import os

# Page configuration
st.set_page_config(page_title="CivicFix", page_icon="🏛️", layout="wide")

# Project paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "nyc_311_cleaned.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models")

# Theme

# Theme
st.markdown("""
<style>
/* Main background */
.stApp {
    background-color: #E8EDF2;
}

/* Sidebar background */
section[data-testid="stSidebar"] {
    background-color: #351C40;
}

/* Sidebar labels and headings */
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2 {
    color: white !important;
}

/* Model dropdown background */
section[data-testid="stSidebar"] div[data-baseweb="select"] > div {
    background-color: white !important;
    border-color: #D0D0D0 !important;
}

/* Selected model text */
section[data-testid="stSidebar"] div[data-baseweb="select"] span,
section[data-testid="stSidebar"] div[data-baseweb="select"] div {
    color: #351C40 !important;
    -webkit-text-fill-color: #351C40 !important;
}

/* Dropdown arrow */
section[data-testid="stSidebar"] div[data-baseweb="select"] svg {
    fill: #351C40 !important;
}

/* Dropdown menu options */
[data-baseweb="popover"] [role="option"] {
    color: #351C40 !important;
    background-color: white !important;
}

/* Headings */
h1, h2, h3 {
    color: #351C40;
}

/* Buttons */
.stButton > button {
    background-color: #351C40;
    color: white;
    border-radius: 8px;
}
</style>
""", unsafe_allow_html=True)


# Load dataset
@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

# Load models
@st.cache_resource
def load_models():
    return {
        "Logistic Regression": joblib.load(
            os.path.join(MODEL_PATH, "logistic_regression.pkl")
        ),
        "Random Forest": joblib.load(
            os.path.join(MODEL_PATH, "random_forest.pkl")
        ),
        "Gradient Boosting": joblib.load(
            os.path.join(MODEL_PATH, "gradient_boosting.pkl")
        )
    }

df = load_data()
models = load_models()

# Header
st.title("🏛️ CivicFix")
st.write(
    "NYC 311 Complaint Resolution Prediction System. "
    "Predict whether a complaint will be resolved within 7 days."
)

# Sidebar
st.sidebar.title("CivicFix")
st.sidebar.write("Complaint Prediction")

# Model selection
model_name = st.sidebar.selectbox(
    "Select Machine Learning Model",
    list(models.keys()),
    key="model_name"
)

def reset_inputs():
    st.session_state["model_name"] = "Logistic Regression"
    st.session_state["agency"] = get_options("agency")[0]
    st.session_state["complaint_type"] = get_options("complaint_type")[0]
    st.session_state["descriptor"] = get_options("descriptor")[0]
    st.session_state["incident_zip"] = get_options("incident_zip")[0]
    st.session_state["borough"] = get_options("borough")[0]
    st.session_state["location_type"] = get_options("location_type")[0]
    st.session_state["created_month"] = 6
    st.session_state["created_weekday"] = get_options("created_weekday")[0]
    st.session_state["created_hour"] = 12
# Input section
st.subheader("Enter Complaint Details")

col1, col2, col3 = st.columns(3)

# Categorical options from dataset
def get_options(column):
    values = df[column].dropna().astype(str).unique().tolist()
    return sorted(values)

with col1:
    agency = st.selectbox("Agency", get_options("agency"), key="agency")
    complaint_type = st.selectbox("Complaint Type", get_options("complaint_type"), key="complaint_type")
    descriptor = st.selectbox("Descriptor", get_options("descriptor"), key="descriptor")

with col2:
    incident_zip = st.selectbox("Incident ZIP", get_options("incident_zip"), key="incident_zip")
    borough = st.selectbox("Borough", get_options("borough"), key="borough")
    location_type = st.selectbox("Location Type", get_options("location_type"), key="location_type")

with col3:
    created_month = st.number_input(
        "Created Month",
        min_value=1,
        max_value=12,
        value=6,
        key="created_month"
    )

    created_weekday = st.selectbox(
        "Created Weekday",
        get_options("created_weekday"),
        key="created_weekday"
    )

    created_hour = st.number_input(
        "Created Hour",
        min_value=0,
        max_value=23,
        value=12,
        key="created_hour"
    )

# Prediction
if st.button("Predict Resolution Time"):
    input_data = pd.DataFrame([{
        "agency": agency,
        "complaint_type": complaint_type,
        "descriptor": descriptor,
        "incident_zip": incident_zip,
        "borough": borough,
        "location_type": location_type,
        "created_month": created_month,
        "created_weekday": created_weekday,
        "created_hour": created_hour
    }])

    model = models[model_name]
    prediction = model.predict(input_data)[0]

    st.subheader("Prediction Result")

    if prediction == 0:
        st.success("Complaint is predicted to be resolved within 7 days.")
    else:
        st.warning("Complaint is predicted to take more than 7 days.")

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(input_data)[0]
        classes = list(model.classes_)

        if 0 in classes:
            st.write(
                f"Within 7 days probability: "
                f"{probabilities[classes.index(0)] * 100:.2f}%"
            )

        if 1 in classes:
            st.write(
                f"More than 7 days probability: "
                f"{probabilities[classes.index(1)] * 100:.2f}%"
            )


st.button(
    "Clear / Reset",
    on_click=reset_inputs
)