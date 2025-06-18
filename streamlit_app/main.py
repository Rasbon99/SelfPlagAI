import streamlit as st

# Configurazione della pagina
st.set_page_config(page_title="SelfPlagAI - Prompt Inference", layout="centered")

st.title("🧠 Invia un Prompt al Modello")
st.write("Inserisci una domanda e scegli il modello per generare una risposta.")

# Input dell'utente
prompt = st.text_area("📥 Inserisci il tuo prompt:", placeholder="Es. Spiega il principio di funzionamento di un motore a combustione interna.")

# Selezione del modello
model_options = [
    "Phi-3-mini Instruct",
    "Mistral 7B Instruct v0.2",
    "Qwen 1.5 7B Chat",
    "Gemma 2 7B",
    "LLaMA 3 8B Instruct",
    "Mixtral 8x7B MoE",
    "OpenChat 3.5 7B"
]

selected_model = st.selectbox("🤖 Scegli un modello:", model_options)

# Pulsante per inviare
if st.button("Genera risposta"):
    if not prompt.strip():
        st.warning("⚠️ Inserisci un prompt prima di continuare.")
    else:
        # Per ora simula una risposta
        st.info(f"✅ Prompt ricevuto: `{prompt}`")
        st.info(f"📌 Modello selezionato: `{selected_model}`")
        st.success("🚧 La risposta sarà generata non appena l'inference sarà collegata.")

