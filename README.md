# CivicFix – NYC 311 Complaint Resolution Prediction

## Project Overview

CivicFix is a Machine Learning project that uses NYC 311 public-service complaint data to predict whether a complaint will be resolved within 7 days or take more than 7 days.

The system uses information available when a complaint is submitted to make predictions.

### Prediction Categories

- **0 – Within 7 Days:** Complaint resolved within 7 days.
- **1 – More than 7 Days:** Complaint takes more than 7 days to resolve.

## Project Objectives

- Analyze NYC 311 public-service complaints.
- Identify patterns affecting complaint resolution time.
- Train machine learning classification models.
- Compare model performance using appropriate evaluation metrics.
- Provide an interactive prediction interface using Streamlit.

## Dataset

**Source:** NYC Open Data – 311 Service Requests (2020–Present)

The project uses a sample of 20,000 records collected from 2020–2025.

The models use nine features available at complaint submission, including agency, complaint type, descriptor, borough, ZIP code, location type, and complaint creation date/time information.

## Project Workflow

1. Data Extraction
2. Data Cleaning and Exploratory Data Analysis (EDA)
3. Feature Engineering and Preprocessing
4. Model Training
5. Model Evaluation
6. Streamlit Application

## Machine Learning Models

Three classification algorithms were trained and evaluated:

- Logistic Regression
- Random Forest
- Gradient Boosting

## Model Evaluation Results

The models were evaluated on a chronological test set of 4,000 records.

| Model | Accuracy | Precision | Recall | F1-Score |
|---|---:|---:|---:|---:|
| Logistic Regression | 85.40% | 51.39% | 90.07% | 65.44% |
| Random Forest | 88.08% | 57.88% | 81.92% | 67.84% |
| Gradient Boosting | 88.78% | 60.87% | 75.24% | 67.30% |

**Key Findings:**
- Random Forest achieved the highest F1-score (67.84%).
- Logistic Regression achieved the highest recall (90.07%).
- Gradient Boosting achieved the highest accuracy (88.78%) and precision (60.87%).

## Streamlit Application

CivicFix provides an interactive interface where users can:

- Select a trained machine learning model.
- Enter complaint information.
- Predict the expected resolution category.
- View prediction probabilities.

## Repository Structure

```text
CivicFix/
│
├── app/
│   └── streamlit_app.py
│
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
│   ├── logistic_regression.pkl
│   ├── random_forest.pkl
│   └── gradient_boosting.pkl
│
├── notebooks/
│   ├── 01_data_extraction.ipynb
│   ├── 02_data_cleaning_EDA.ipynb
│   ├── 03_model_training.ipynb
│   └── 04_model_evaluation.ipynb
│
├── README.md
├── requirements.txt
└── .gitignore
```

## Installation and Execution

### 1. Clone the Repository

```bash
git clone https://github.com/Krishnalund/CivicFix.git
cd CivicFix
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit Application

```bash
streamlit run app/streamlit_app.py
```

The application will open in your browser.

## Evaluation Metrics

The project uses:

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix

## Project Objective

CivicFix aims to demonstrate how machine learning can help analyze public-service complaint patterns and estimate resolution delays, supporting better understanding of public-service operations.

## Team

- Krishna Lund – Data Extraction , Data Cleaning and Exploratory Data Analysis (EDA)
- Sneha Kanjwani – Feature Engineering and Model Training
- Subhash Bajaj – Model Evaluation and Streamlit Application
