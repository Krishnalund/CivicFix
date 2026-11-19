# CivicFix – NYC 311 Complaint Resolution Prediction

## Project Overview
CivicFix is a Machine Learning project that analyzes NYC 311 public-service complaints to predict their resolution time.

The project classifies complaints into two categories:
- **Within 7 Days:** Complaints resolved within 7 days.
- **After 7 Days:** Complaints taking more than 7 days.

## Project Workflow
1. Data Extraction
2. Data Cleaning and Exploratory Data Analysis (EDA)
3. Model Training
4. Model Evaluation

## Machine Learning Models
We trained and evaluated three classification models:
- Logistic Regression
- Random Forest
- Gradient Boosting

## Evaluation Metrics
Models were evaluated using:
- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix

## Repository Structure
```text
CivicFix/
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

## Objective
To use machine learning to predict NYC public-service complaint resolution times and support better service management.
