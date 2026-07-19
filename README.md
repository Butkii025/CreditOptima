<div align="center">

# CreditOptima

**Instant, transparent loan-approval predictions system - powered by machine learning.**

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://creditoptima.streamlit.app/)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4%2B-orange)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

**Try to check your loan approval [ CreditOptima ](https://creditoptima.streamlit.app/)**

</div>

---

## Overview

**CreditOptima** is an interactive Streamlit application that predicts whether
a loan application is likely to be **approved or rejected**, based on
applicant details like income, credit score, existing debt, and employment
status.

It trains three classic machine learning classifiers on startup **Logistic
Regression**, **K-Nearest Neighbors**, and **Naive Bayes** lets you compare
how each one performs, and gives you a live, explainable prediction with a
confidence score for any applicant profile you enter.

## Problem Statement

* XYZ is a mid-sized financial company that offers personal and home loans to customers across urban and rural regions of India. Every day, hundreds of customers apply for loans through online and branch applications.
* Until now, XYZ has relied on a manual verification process, where loan officers evaluate applications by checking income proofs, employment details, credit history, and other supporting documents. This process is time-consuming, inconsistent, and prone to human bias.
* As a result, the bank faces two major challenges:
  
*Good customers sometimes get rejected, leading to loss of business.*

*High-risk customers sometimes get approved, leading to financial losses.*

* To solve this problem, the bank wants to introduce an intelligent loan approval system powered by Machine Learning that can automatically analyze applicant details and predict whether a loan should be Approved or Rejected, before final human verification.
* As the Machine Learning Engineer on this project, the goal is to design and develop this intelligent system using historical loan application data — one that learns hidden patterns from previous customer records and delivers accurate, fast, and unbiased loan approval decisions.


## Features

| | |
|---|---|
| **Live Prediction** | Fill in an applicant's financial and personal details and get an instant Approved / Not Approved verdict with an estimated approval probability. |
| **Model Choice** | Switch between Logistic Regression, K-Nearest Neighbors, and Naive Bayes to see how each model calls the same application. |
| **Model Comparison** | Side-by-side accuracy, precision, recall, and F1 score, plus confusion matrices for every model. |
| **Dataset Explorer** | A quick look at the underlying training data, including class balance and credit score distribution. |

## Live Demo

The app is deployed and publicly available on Streamlit Community Cloud:

### [creditoptima.streamlit.app](https://creditoptima.streamlit.app/)

No installation needed — just open the link and start predicting.

## How it works

1. **Preprocessing** — missing numeric values are mean-imputed, missing
   categorical values are imputed with the most frequent category.
2. **Feature engineering** — `Credit_Score` and `DTI_Ratio` are transformed
   into squared features (`Credit_Score_sq`, `DTI_Ratio_sq`) to better
   capture non-linear effects on approval likelihood.
3. **Encoding & scaling** — categorical fields (employment status, marital
   status, loan purpose, property area, education, gender, employer category)
   are one-hot encoded; all numeric features are standard-scaled.
4. **Modeling** — three classifiers are trained on an 80/20 train-test split
   and evaluated on accuracy, precision, recall, and F1 score.
5. **Prediction** — new applicant data is passed through the same
   preprocessing pipeline and scored by the selected model.

## Project structure

```
CreditOptima/
├── app.py                    # Streamlit app (UI)
├── model_utils.py            # Data loading, preprocessing, training, prediction logic
├── loan_approval_data.csv    # Training dataset
├── requirements.txt          # Dependencies
└── README.md
```
## Visuals

<img width="392" height="293" alt="Screenshot 2026-07-17 155102" src="https://github.com/user-attachments/assets/8fbb0dea-d647-4992-ae4a-5b19ca973a4f" />
<img width="410" height="248" alt="Screenshot 2026-07-17 155149" src="https://github.com/user-attachments/assets/db26b85d-fbff-467e-9a21-7af813c50d23" />
<img width="417" height="100" alt="Screenshot 2026-07-17 155129" src="https://github.com/user-attachments/assets/b491edbb-22e0-40f6-8d78-eae05bb97003" />

---

## Run it locally

Clone the repo and set up a virtual environment:

```bash
git clone https://github.com/Butkii025/CreditOptima.git
cd CreditOptima
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
```

Install dependencies and launch:

```bash
pip install -r requirements.txt
streamlit run app.py
```

The app will open automatically at `http://localhost:8501`

### Using Anaconda

```bash
conda create -n creditoptima python=3.11 -y
conda activate creditoptima
pip install -r requirements.txt
streamlit run app.py
```

## Deployment

CreditOptima is deployed on **Streamlit Community Cloud**, which auto-builds
from this repository:

1. Push changes to the `main` branch.
2. Streamlit Cloud detects the update, reinstalls `requirements.txt`, and
   redeploys automatically.
3. No servers, Docker files, or secrets are required — the models train in
   memory on startup (and are cached, so it only happens once per instance).

## Model performance

| Model | Accuracy | Precision | Recall | F1 Score |
|---|---:|---:|---:|---:|
| Logistic Regression | ~0.87 | ~0.82 | ~0.75 | ~0.78 |
| Naive Bayes | ~0.86 | ~0.82 | ~0.73 | ~0.77 |
| K-Nearest Neighbors | ~0.76 | ~0.63 | ~0.60 | ~0.62 |

*(Figures vary slightly by random seed / train-test split — see the **Model
Comparison** tab in the live app for current numbers.)*

## Tech stack

- [Streamlit](https://streamlit.io/) — web app framework
- [scikit-learn](https://scikit-learn.org/) — ML models & preprocessing
- [pandas](https://pandas.pydata.org/) / [NumPy](https://numpy.org/) — data handling

## Contributing

Issues and pull requests are welcome. If you'd like to add a new model,
improve the UI, or extend the dataset, feel free to fork and open a PR.

## License

This project is available under the [MIT License](LICENSE).

## [ Note ]

CreditOptima is a demonstration project built for educational purposes. Its
predictions are based on a sample dataset and should **not** be used as the
sole basis for real lending or credit decisions.

---
