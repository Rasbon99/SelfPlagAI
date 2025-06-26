# streamlit_app/main.py

import streamlit as st
import utils.mongo as mn
import sys
import os
import pandas as pd
import plotly.express as px

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

    metric_map = {
    "Exact Match": "exact_match",
    "F1 Score": "f1_score",
    "BERTScore F1": "bert_score_f1",
    "Semantic Similarity": "semantic_similarity",
    "Type Token Ration" : "ttr"
    }
    selected_metrics = ["Exact Match", "F1 Score", "BERTScore F1", "Semantic Similarity", "Type Token Ration"]

# Filtra per modello e database selezionati
filtered_df = df_metrics[
    (df_metrics['base_model_name'] == selected_model) &
    (df_metrics['database_name'] == selected_database)
]

if selected_gen == "Generations":
    plot_df = filtered_df.copy()
    plot_df['generation_num'] = plot_df['generation'].str.extract(r'(\d+)').astype(int)
    plot_df = plot_df.sort_values('generation_num')
    # Melt per plotly
    plotly_df = plot_df.melt(
        id_vars=['generation'],
        value_vars=[metric_map[m] for m in selected_metrics],
        var_name='Metrica',
        value_name='Valore'
    )
    fig = px.line(
        plotly_df,
        x='generation',
        y='Valore',
        color='Metrica',
        markers=True,
        labels={'generation': 'Generation', 'Valore': 'Valore', 'Metrica': 'Metrica'}
    )
    st.plotly_chart(fig, use_container_width=True)
else:
    # Seleziona la domanda
    all_questions = filtered_df.iloc[0]['questions']
    question_idx = st.selectbox(
        "Select a Question:",
        options=list(range(len(all_questions))),
        format_func=lambda i: all_questions[i][:80] + ("..." if len(all_questions[i]) > 80 else "")
    )

    col4, col5 = st.columns([3,2])

    with col4:
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
            if "Type Token Ration" in selected_metrics:
                data["Type Token Ration"].append(row["individual_ttr"][question_idx])
        chart_df = pd.DataFrame(data).set_index("generation")
        # Trasformazione per plotly
        plotly_df = chart_df.reset_index().melt(
            id_vars=['generation'],
            value_vars=selected_metrics,
            var_name='Metrica',
            value_name='Valore'
        )
        fig = px.line(
            plotly_df,
            x='generation',
            y='Valore',
            color='Metrica',
            markers=True,
            labels={'generation': 'Generation', 'Valore': 'Valore', 'Metrica': 'Metrica'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col5:
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
        # Stampa Question, Context e Reference della domanda selezionata
        selected_question = all_questions[question_idx]
        selected_context = filtered_df.iloc[0]['contexts'][question_idx]
        selected_reference = filtered_df.iloc[0]['references'][question_idx]
        st.info(
            f"**Question:** {selected_question}\n\n"
            f"**Context:** {selected_context}\n\n"
            f"**Reference:** {selected_reference}"
        )
        st.dataframe(pred_df, use_container_width=True, hide_index=True)

st.divider()








