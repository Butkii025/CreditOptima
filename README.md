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

## Run it locally

1. **Clone the repo / open this folder**, then create a virtual environment
   (optional but recommended):

   ```bash
   python -m venv venv
   source venv/bin/activate        # Windows: venv\Scripts\activate
   ```

2. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

3. **Run the app:**

   ```bash
   streamlit run app.py
   ```

   It will open automatically at `http://localhost:8501`.

## Deploy on Streamlit Community Cloud

1. Push this folder to a **public (or private) GitHub repository**, making
   sure `app.py`, `model_utils.py`, `requirements.txt`, and
   `loan_approval_data.csv` are all committed at the same level (or update
   the path in `app.py` if you nest them in a subfolder).
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with
   GitHub.
3. Click **"New app"**, then select:
   - **Repository:** your repo
   - **Branch:** `main` (or whichever branch you pushed to)
   - **Main file path:** `app.py`
4. Click **Deploy**. Streamlit Cloud will install everything from
   `requirements.txt` and launch the app — you'll get a public URL like
   `https://your-app-name.streamlit.app`.
5. Any time you push new commits to that branch, the deployed app
   auto-updates.

### Notes on deployment

- The app trains the models **in memory on startup** (cached with
  `st.cache_resource`, so it only retrains once per server instance, not on
  every interaction) — there's no separate model file to manage. Keep
  `loan_approval_data.csv` in the repo so the app has data to train on.
- No secrets, API keys, or external services are required.
- If you rename or move the CSV, update `DATA_PATH` in `app.py` accordingly.

## Deploy elsewhere (Docker / Render / Hugging Face Spaces, etc.)

Any platform that can run `pip install -r requirements.txt` followed by
`streamlit run app.py --server.port $PORT --server.address 0.0.0.0` will
work — the app has no other infrastructure dependencies.

## Retraining on new data

To use a different dataset, replace `loan_approval_data.csv` with a file
that has the same columns, or adjust the column lists at the top of
`model_utils.py` (`CATEGORICAL_COLS`, `RAW_NUMERIC_COLS`) to match your data.

## Model performance (on this dataset's held-out test set)

| Model               | Accuracy | Precision | Recall | F1 Score |
|---------------------|---------:|----------:|-------:|---------:|
| Logistic Regression |   ~0.87  |   ~0.82   |  ~0.75 |   ~0.78  |
| Naive Bayes         |   ~0.86  |   ~0.82   |  ~0.73 |   ~0.77  |
| K-Nearest Neighbors |   ~0.76  |   ~0.63   |  ~0.60 |   ~0.62  |

*(Exact numbers vary slightly by random seed / train-test split; see the
Model Comparison tab in the app for live figures.)*
