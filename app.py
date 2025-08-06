import streamlit as st
import pandas as pd
import joblib

# Load model and encoders
model = joblib.load("mushroom_detector_forest.pkl")
encoders = joblib.load("label_encoders.pkl")
test_data = pd.read_csv("test_data.csv")

# Decode class labels
class_map = {0: "edible", 1: "poisonous"}

st.title("🍄 Mushroom Classifier")

# Select a row by index
selected_index = st.selectbox("Select a mushroom sample:", test_data.index)

# Display selected row
sample = test_data.iloc[selected_index:selected_index+1]
st.write("Selected Mushroom Features:")
st.write(sample.drop(columns=["class"]))

# Inference
proba = model.predict_proba(sample.drop(columns=["class"]))[0][1]
threshold = 0.3
pred = int(proba >= threshold)

st.markdown(f"**🔍 Predicted:** {class_map[pred]}")
st.markdown(f"**✅ Actual:** {class_map[sample['class'].values[0]]}")
st.markdown(f"**🧠 Poisonous probability:** {proba:.4f}")
