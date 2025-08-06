
# 🍄 Mushroom Classification – Anomaly Detection App

[![Streamlit App](https://img.shields.io/badge/Live%20App-Click%20Here-brightgreen)](https://mushroom-detector-app-uocer8scfn2dnw27fkofyw.streamlit.app/)

This project demonstrates a **machine learning-based anomaly detection system** for classifying mushrooms as **edible** or **poisonous** using two models: **Random Forest** and **Logistic Regression**. The focus is on **maximizing recall** for poisonous mushrooms to ensure safety.

---


## 🔍 Dataset

- Source: [Kaggle – UCI Mushroom Dataset](https://www.kaggle.com/datasets/uciml/mushroom-classification)
- Features: 22 categorical variables describing mushroom characteristics
- Target: `class` — `p` (poisonous) or `e` (edible)

---

## 💡 Approach

### ✅ Classification Models:
- **Logistic Regression** with class weighting and threshold tuning
- **Random Forest** with calibrated decision threshold

### ✅ Key Techniques:
- **Label Encoding** of categorical features
- **Train/test split** with stratification
- **Custom threshold (0.3)** to optimize **recall**
- **Streamlit App** for interactive prediction and model comparison

---

## 📊 Metrics

Since false negatives are dangerous (e.g., mislabeling a poisonous mushroom as edible), we optimized for:

- **Recall (Poisonous Class)** – maximize
- **Precision** – monitor
- **Confusion Matrix** – interpret safety trade-offs

---

## 🧪 Streamlit Web App

🚀 Try it here:  
👉 [**Live App on Streamlit Cloud**](https://mushroom-detector-app-uocer8scfn2dnw27fkofyw.streamlit.app/)

### Features:
- Select a mushroom sample from the test set
- Run predictions using both models
- View:
  - Predicted class (edible or poisonous)
  - Probability/confidence
  - Actual label for comparison


---

## 📦 Installation (Local)

```bash
git clone https://github.com/your-username/mushroom-detector-app.git
cd mushroom-detector-app
pip install -r requirements.txt
streamlit run app.py
