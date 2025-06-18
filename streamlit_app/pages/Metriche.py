import streamlit as st
from utils.mock_data import load_mock_metrics

st.title("📊 Metriche QA")
st.write("Qui verranno mostrati i grafici relativi alle metriche.")

df = load_mock_metrics()
st.dataframe(df)

st.line_chart(df.set_index("Prompt ID")[["EM Score", "F1 Score"]])