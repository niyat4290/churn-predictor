# 📡 Telco Customer Churn Predictor

An interactive machine learning app that predicts whether a telecom customer will churn — and **explains why** using SHAP values.

> Built with XGBoost · SHAP · Streamlit

---

## 🚀 Live Demo
**[Click here to open the Live App](https://churn-predictor-5e4w3mwnmsy8z66usxkstp.streamlit.app/)**

---

## 🧠 What It Does

- Predicts **churn probability** for a customer based on 20 features (contract type, tenure, charges, services, etc.)
- Shows a **SHAP waterfall chart** explaining exactly which factors push toward or away from churn for that specific customer
- Displays **global feature importance** across the full dataset
- Includes a raw dataset preview

---

## 📁 Project Structure

```
churn-predictor/
├── app.py                    # Streamlit app
├── train_model.py            # Model training script (run once)
├── Telco-Customer-Churn.csv  # Dataset
├── requirements.txt          # Python dependencies
├── model.pkl                 # Trained XGBoost model  (generated)
├── encoders.pkl              # Label encoders         (generated)
├── explainer.pkl             # SHAP TreeExplainer     (generated)
├── feature_names.pkl         # Feature name list      (generated)
└── README.md
```

---

## ⚙️ Run Locally

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/churn-predictor.git
cd churn-predictor

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Train the model (generates .pkl files)
python train_model.py

# 5. Launch the app
streamlit run app.py
```

App opens at **http://localhost:8501**

---

## 📊 Model Performance

| Metric     | Score  |
|------------|--------|
| ROC-AUC    | ~0.85  |
| Precision  | ~0.67  |
| Recall     | ~0.55  |
| F1 (Churn) | ~0.60  |

Class imbalance handled via `scale_pos_weight` in XGBoost.

---

## 🗃️ Dataset

[Telco Customer Churn — IBM Sample Data](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

7,043 customers · 21 features · Binary target: `Churn` (Yes/No)

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| XGBoost | Gradient boosted tree classifier |
| SHAP | Model explainability |
| Streamlit | Interactive web app |
| scikit-learn | Preprocessing & evaluation |
| Matplotlib | Visualizations |

---

## 📌 Key Learnings

- SHAP TreeExplainer provides both local (per-customer) and global (dataset-wide) explanations
- `scale_pos_weight` is critical for imbalanced churn datasets (73% No / 27% Yes)
- Streamlit's `@st.cache_resource` keeps heavy model artifacts in memory across reruns
