# streamlit_app/main.py

import streamlit as st
import utils.mongo as mn
import sys
import os
import pandas as pd

# Aggiunge la cartella precedente al percorso
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Configura la pagina
st.set_page_config(page_title="SelfPlagAI", layout="wide")
st.title("🧠 Dashboard SelfPlagAI")

df_metrics = mn.get_eval_metrics()

col1, col2, col3 = st.columns(3)


with col1:

    model_options = df_metrics['base_model_name'].unique()
    selected_model = st.selectbox("Model:", model_options) 


with col2: 
    
    database_options = df_metrics['database_name'].unique()
    selected_database = st.selectbox("Database:", database_options)
with col3:
    plot_options = ['Generations', 'Question']
    selected_gen = st.selectbox("Plot per:", plot_options) 
st.subheader("📊 Evaluation Metrics")

metric_map = {
    "Exact Match": "exact_match",
    "F1 Score": "f1_score",
    "BERTScore F1": "bert_score_f1",
    "Semantic Similarity": "semantic_similarity"
}
available_metrics = list(metric_map.keys())
selected_metrics = st.multiselect(
    "Select metrics:",
    options=available_metrics,
    default=available_metrics
)

# Filtra per modello e database selezionati
filtered_df = df_metrics[
    (df_metrics['base_model_name'] == selected_model) &
    (df_metrics['database_name'] == selected_database)
]

if selected_gen == "Generations":
    # Plotta le metriche aggregate per generazione
    plot_df = filtered_df.copy()
    plot_df['generation_num'] = plot_df['generation'].str.extract(r'(\d+)').astype(int)
    plot_df = plot_df.sort_values('generation_num')
    st.line_chart(
        plot_df.set_index('generation')[[metric_map[m] for m in selected_metrics]],
        use_container_width=True
    )
else:
    # Seleziona la domanda
    all_questions = filtered_df.iloc[0]['questions']
    question_idx = st.selectbox(
        "Select a Question:",
        options=list(range(len(all_questions))),
        format_func=lambda i: all_questions[i][:80] + ("..." if len(all_questions[i]) > 80 else "")
    )
    # Costruisci il dataframe per la domanda selezionata
    data = {
        "generation": [],
    }
    for m in selected_metrics:
        data[m] = []
    plot_df = filtered_df.copy()
    plot_df['generation_num'] = plot_df['generation'].str.extract(r'(\d+)').astype(int)
    plot_df = plot_df.sort_values('generation_num')
    for _, row in plot_df.iterrows():
        data["generation"].append(row["generation"])
        if "Exact Match" in selected_metrics:
            data["Exact Match"].append(row["individual_exact_match"][question_idx])
        if "F1 Score" in selected_metrics:
            data["F1 Score"].append(row["individual_token_f1"][question_idx])
        if "BERTScore F1" in selected_metrics:
            data["BERTScore F1"].append(row["individual_bert_f1"][question_idx])
        if "Semantic Similarity" in selected_metrics:
            data["Semantic Similarity"].append(row["individual_semantic_similarity"][question_idx])
    chart_df = pd.DataFrame(data).set_index("generation")
    st.line_chart(chart_df, use_container_width=True)

    # Visualizza le predictions in tabella
    st.subheader("📝 Predictions per la domanda selezionata")

    predictions_data = []
    for _, row in plot_df.iterrows():
        gen_name = row["generation"]
        preds = row["predictions"]
        if preds and len(preds) > question_idx:
            prediction_text = preds[question_idx]
        else:
            prediction_text = "N/A"
        predictions_data.append({"Generation": gen_name, "Prediction": prediction_text})

    pred_df = pd.DataFrame(predictions_data)
    st.dataframe(pred_df, use_container_width=True, hide_index=True)

st.divider()








