# CRM Lead Intelligence System

A machine learning application that predicts the probability of a lead converting and helps sales teams prioritize follow-ups.

## Project Overview

The CRM Lead Intelligence System uses historical marketing campaign data to estimate lead conversion probabilities. It ranks leads by their predicted likelihood of conversion, helping sales teams focus their efforts.

This project was developed as a practical machine learning portfolio project.

## Features

- Predicts lead conversion probability
- Assigns lead priority based on a configurable threshold
- Ranks leads by predicted conversion probability
- Uses a preprocessing and prediction pipeline
- Provides a Streamlit interface for scoring leads

## Machine Learning

**Dataset:** UCI Bank Marketing dataset

**Models explored:**
- Logistic Regression
- K-Nearest Neighbors (KNN)

**Selected model:** Logistic Regression

The feature `duration` was excluded because it is only known after a call and would cause data leakage for pre-call lead scoring.

### Evaluation

| Metric | Logistic Regression |
|---|---:|
| Accuracy | 90.09% |
| Precision | 69.05% |
| Recall | 21.88% |
| F1-score | 33.22% |
| ROC-AUC | 0.8008 |
| Brier score | 0.0769 |

The dataset is imbalanced, so accuracy alone does not describe model performance. The model's default classification threshold is 0.5; a lower threshold can increase recall depending on the sales team's priorities.

## Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- Joblib

## Project Structure

```text
crm-lead-intelligence-system/
├── app.py
├── requirements.txt
└── models/
    └── crm_lead_model.pkl
```

## Run Locally

1. Clone this repository:

   ```bash
   git clone https://github.com/santoshkumar1649/CRM-Lead-Intelligence-system.git
   cd CRM-Lead-Intelligence-system
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Start the app:

   ```bash
   streamlit run app.py
   ```

## Dataset

The project uses the [UCI Bank Marketing dataset](https://archive.ics.uci.edu/dataset/222/bank+marketing), which contains data from a Portuguese bank's direct marketing campaigns.

## Limitations

- The dataset represents a specific bank and campaign context; predictions may not generalize to other businesses.
- Model probabilities are estimates, not guarantees.
- The priority threshold should be selected based on business costs and sales capacity.
- The model excludes call duration to support scoring before a call takes place.

## Future Improvements

- Add batch lead scoring from CSV uploads
- Improve probability calibration
- Compare additional models and ranking metrics
- Deploy the Streamlit app
