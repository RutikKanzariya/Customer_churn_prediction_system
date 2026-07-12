# 📉 ChurnGuard — Customer Churn Prediction System

An end-to-end machine learning system that predicts customer churn using **XGBoost**,
tuned with **Optuna**, balanced with **SMOTE**, explained with **SHAP**, and deployed
as an interactive **Streamlit** web app.

### 🚀 Live Demo
*(Deploy to Streamlit Cloud and add your link here)*

---

## 🎯 Project Overview

Customer churn — when a customer stops using a company's service — is one of the most
costly problems for subscription-based businesses. This project builds a full pipeline
that identifies customers at high risk of churning, so retention teams can intervene
early.

The dataset used is the **Telco Customer Churn** dataset (Kaggle), containing ~7,000
customer records with demographic info, account details, and service subscriptions.

---

## 🧠 What This Project Demonstrates

- **Data preprocessing & feature engineering**: cleaning, encoding, engineered features
  (`AvgMonthlySpend`, `TenureGroup`)
- **Handling class imbalance** with SMOTE (oversampling the minority "churn" class)
- **Baseline comparison**: Logistic Regression vs. Random Forest vs. XGBoost
- **Boosting technique**: XGBoost as the primary model
- **Hyperparameter tuning**: Optuna (Bayesian/TPE-based search, 30 trials) over 9
  hyperparameters (`max_depth`, `learning_rate`, `subsample`, `colsample_bytree`,
  `reg_alpha`, `reg_lambda`, `gamma`, `min_child_weight`, `n_estimators`)
- **Model explainability**: SHAP values (global feature importance + per-prediction
  waterfall plots)
- **Deployment**: interactive Streamlit app for real-time predictions

---

## 📊 Results

| Model                  | F1 Score | ROC-AUC |
|-------------------------|----------|---------|
| Logistic Regression     | 0.750    | 0.866   |
| Random Forest            | 0.838    | 0.945   |
| XGBoost (default)        | 0.827    | 0.941   |
| **XGBoost (Optuna-tuned)** | **0.863** | **0.951** |

*(Results shown are from the included sample dataset — retrain on the real Kaggle CSV
for production numbers.)*

Hyperparameter tuning improved ROC-AUC over the default XGBoost baseline and outperformed
all other models tested.

---

## 🗂️ Project Structure

```
churn_project/
│
├── data/
│   ├── telco_churn.csv          # dataset (replace with real Kaggle download)
│   └── generate_sample_data.py  # generates a schema-matching synthetic dataset
│
├── models/                      # generated after running train_model.py
│   ├── xgb_churn_model.pkl
│   ├── scaler.pkl
│   ├── label_encoders.pkl
│   ├── metadata.json
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   ├── feature_importance.png
│   └── shap_summary.png
│
├── train_model.py               # full training pipeline
├── app.py                       # Streamlit deployment app
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup & Usage

### 1. Clone and install dependencies
```bash
git clone <your-repo-url>
cd churn_project
pip install -r requirements.txt
```

### 2. Get the real dataset (recommended)
Download **Telco Customer Churn** from Kaggle:
👉 https://www.kaggle.com/datasets/blastchar/telco-customer-churn

Save it as `data/telco_churn.csv` (same filename, same columns — no code changes
needed).

*Or*, to just test the pipeline immediately without downloading anything:
```bash
python data/generate_sample_data.py
```
This creates a synthetic dataset with the exact same schema as the real one.

### 3. Train the model
```bash
python train_model.py
```
This runs the full pipeline: preprocessing → SMOTE → baseline models → XGBoost →
Optuna tuning → SHAP → saves all artifacts to `models/`.

### 4. Run the app
```bash
streamlit run app.py
```
Open `http://localhost:8501` in your browser.

---

## 🚀 Deploying to Streamlit Cloud

1. Push this repo to GitHub (make sure `models/*.pkl` are committed, or add a build
   step that runs `train_model.py` first)
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repo and set the main file to `app.py`
4. Deploy

---

## 🔮 Future Improvements

- Add LightGBM/CatBoost comparison
- SHAP force plots per prediction, saved as downloadable reports
- Batch prediction via CSV upload
- Model monitoring / drift detection
- REST API (FastAPI) alongside the Streamlit UI

---

## 🛠️ Tech Stack

**Language:** Python
**ML/Boosting:** Scikit-learn, XGBoost
**Imbalance Handling:** imbalanced-learn (SMOTE)
**Hyperparameter Tuning:** Optuna
**Explainability:** SHAP
**Deployment:** Streamlit, Streamlit Cloud
**Data:** Pandas, NumPy
**Visualization:** Matplotlib, Seaborn

---

## 👨‍💻 Author

Built as part of a machine learning portfolio project demonstrating an end-to-end
pipeline from raw data to a deployed, explainable ML application.
