import streamlit as st
from utils.mock_data import load_mock_metrics

st.title("📊 Metriche di valutazione")

df = load_mock_metrics()

st.subheader("🔹 EM Score (Exact Match)")
st.line_chart(
    data=df,
    x="Step Fine-Tuning",
    y="EM Score"
)

st.subheader("🔹 F1 Score")
st.line_chart(
    data=df,
    x="Step Fine-Tuning",
    y="F1 Score"
)
