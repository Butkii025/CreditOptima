# 🏦 Loan Approval Predictor

An interactive Streamlit app that predicts whether a loan application would be
approved, based on the analysis and models built in the original notebook
(`notebook.ipynb`).

The app trains three classifiers on startup — **Logistic Regression**,
**K-Nearest Neighbors**, and **Naive Bayes** — on `loan_approval_data.csv`,
lets you compare their performance, and lets you enter an applicant's details
to get a live prediction.

## Features

- **Predict tab** — fill in an applicant's details and get an approval
  prediction with an estimated probability, using the model of your choice.
- **Model Comparison tab** — accuracy, precision, recall, F1 score, and
  confusion matrices for all three models side by side.
- **Dataset tab** — a quick look at the underlying data and class balance.

## Project structure

```
loan_app/
├── app.py                    # Streamlit app (UI)
├── model_utils.py            # Data loading, preprocessing, training, prediction
├── loan_approval_data.csv    # Training data
├── requirements.txt          # Python dependencies
└── README.md
```
