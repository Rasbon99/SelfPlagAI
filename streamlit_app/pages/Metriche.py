import streamlit as st
from utils.mock_data import load_mock_metrics

st.title("📊 Metriche di valutazione")

df = load_mock_metrics()

# Mostra le metriche EM Score e F1 Score in un unico grafico a linee
st.line_chart(
    data=df.set_index("Step Fine-Tuning")[["EM Score", "F1 Score"]],
    use_container_width=True
)

