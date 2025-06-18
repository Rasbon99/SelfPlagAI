import streamlit as st
from utils.mock_data import load_mock_responses
st.title("💬 Risposte LLM")


df = load_mock_responses()
st.dataframe(df)