# streamlit_app/main.py

import streamlit as st
from utils.mock_data import load_all_mock_data

# Configura la pagina
st.set_page_config(page_title="SelfPlagAI", layout="wide")
st.title("🧠 Dashboard SelfPlagAI")

# Carica tutti i dati mock
data = load_all_mock_data()

# ─────────────────────────────────────────────
# Prompt selezionabile
st.header("📌 Seleziona un prompt predefinito")

prompt_options = list(data.keys())
selected_prompt = st.selectbox("Prompt:", prompt_options)

# ─────────────────────────────────────────────
# Selezione delle metriche da visualizzare
st.header("📊 Metriche di valutazione")

available_metrics = ["EM Score", "F1 Score"]
selected_metrics = st.multiselect(
    "Scegli le metriche da visualizzare:",
    options=available_metrics,
    default=available_metrics
)

metrics_df = data[selected_prompt]["metrics"]

if selected_metrics:
    st.line_chart(
        data=metrics_df.set_index("Step Fine-Tuning")[selected_metrics],
        use_container_width=True
    )
else:
    st.info("⬅️ Seleziona almeno una metrica per vedere il grafico.")

st.divider()

# ─────────────────────────────────────────────
# Visualizzazione risposte
st.header("💬 Risposte Fine-Tuned")

responses_df = data[selected_prompt]["responses"]
st.dataframe(responses_df, use_container_width=True)

