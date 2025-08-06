import streamlit as st
import pandas as pd
import joblib

# Load model and encoders
model_forest = joblib.load("mushroom_detector_forest.pkl")
model_log = joblib.load("mushroom_detector_log.pkl")
encoders = joblib.load("label_encoders.pkl")
test_data = pd.read_csv("test_data.csv")

# Decode class labels
class_map = {0: "edible", 1: "poisonous"}

st.title("🍄 Mushroom Classifier")

# Select a row by index
selected_index = st.selectbox("Select a mushroom sample:", test_data.index)

# Extract sample
sample = test_data.iloc[selected_index:selected_index+1]
sample_features = sample.drop(columns=["class"])
actual_class = class_map[sample["class"].values[0]]

st.write("🧬 Selected Features:")
st.write(sample_features)


threshold = 0.3

def pred_with_model(model, features, threshold):
    prob = model.predict_proba(features)[0][1]
    pred = int(prob >= threshold)
    return class_map[pred], prob

# Inference

rf_pred, rf_prob = pred_with_model(model_forest, sample_features, threshold)
lg_pred, lg_prob = pred_with_model(model_log, sample_features, threshold)


# Show results
st.markdown("## 🧪 Predictions")
col1, col2 = st.columns(2)

with col1:
    st.markdown("### 🌲 Random Forest")
    st.write(f"**Prediction:** {rf_pred}")
    st.write(f"**Probability (poisonous):** {rf_prob:.4f}")

with col2:
    st.markdown("### ➕ Logistic Regression")
    st.write(f"**Prediction:** {lg_pred}")
    st.write(f"**Probability (poisonous):** {lg_prob:.4f}")

# Actual label
st.markdown("### ✅ Actual Label")
st.write(f"**Actual:** {actual_class}")