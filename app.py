import streamlit as st
import pickle
import numpy as np

model = pickle.load(open("model.pkl", "rb"))

st.title("📚 Smart Study Planner")

study_hours = st.slider("Study Hours", 1, 12)
sleep_hours = st.slider("Sleep Hours", 4, 10)
break_time = st.slider("Break Time (hrs)", 0.5, 3.0)

if st.button("Predict Productivity"):
    input_data = np.array([[study_hours, sleep_hours, break_time]])
    prediction = model.predict(input_data)

    st.success(f"Predicted Productivity Score: {round(prediction[0],2)}")

    if prediction > 8:
        st.write("🔥 You're in peak performance zone!")
    elif prediction > 5:
        st.write("👍 Good, but can improve!")
    else:
        st.write("⚠️ Try adjusting your schedule!")
