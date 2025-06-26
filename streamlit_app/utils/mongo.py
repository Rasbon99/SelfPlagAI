import sys
import os
import pandas as pd
import numpy as np
import streamlit as st

# Aggiunge la cartella precedente al percorso
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..\..')))

import db_utils as db
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), '../../key.env'))

@st.cache_data
def get_eval_metrics(collection: str = 'evaluation_results'):
    username = os.getenv("MONGO_USERNAME")
    password = os.getenv("MONGO_PASSWORD")

    if not username or not password:
        raise ValueError("MONGO_USERNAME o MONGO_PASSWORD non trovati nelle variabili d'ambiente")

    client = db.get_mongo_client(username, password)
    df_metrics = db.read_collection(client, collection, as_dataframe=True)
    df_ttr_metrics = db.read_collection(client, collection_name='ttr_results', as_dataframe=True)

    # Prendi la prima riga (assumendo che sia quella che ti interessa)
    row = df_metrics.iloc[0]
    ttr_row = df_ttr_metrics.iloc[0].to_dict() if not df_ttr_metrics.empty else {}

    # Trova tutte le colonne che rappresentano una generazione
    generations = [col for col in df_metrics.columns if col.startswith('generation_')]

    # Prepara accesso rapido a TTR per domanda e generazione
    by_question = ttr_row.get('by_question', {}) if ttr_row else {}
    by_generation = ttr_row.get('by_generation', {}) if ttr_row else {}

    records = []
    for gen_idx, gen_col in enumerate(generations):
        gen = row[gen_col]
        if gen is None or not isinstance(gen, dict):
            continue

        individual_scores = gen.get('individual_scores', {})
        generation_num = gen.get('generation', gen_col)
        generation_name = f"Generation {generation_num}"

        # TTR individuale: lista di ttr per ogni domanda di questa generazione
        ttr_individual = []
        if by_question:
            # by_question è un dict con chiavi stringa degli indici domanda
            for q_idx in range(len(gen.get('questions', []))):
                q_obj = by_question.get(str(q_idx), {})
                gen_obj = q_obj.get(f'gen_{gen_idx}', {})
                ttr_val = gen_obj.get('ttr_metrics', {}).get('ttr', np.nan)
                ttr_individual.append(ttr_val)
        else:
            ttr_individual = [np.nan] * len(gen.get('questions', []))

        # TTR aggregato per generazione
        ttr_mean = np.nan
        if by_generation:
            gen_obj = by_generation.get(str(gen_idx), {})
            ttr_mean = gen_obj.get('mean_ttr', np.nan)

        record = {
            'generation': generation_name,
            'test_size': gen.get('test_size', np.nan),
            'exact_match': gen.get('exact_match', np.nan),
            'f1_score': gen.get('f1_score', np.nan),
            'bert_score_f1': gen.get('bert_score_f1', np.nan),
            'semantic_similarity': gen.get('semantic_similarity', np.nan),
            'avg_prediction_length': gen.get('avg_prediction_length', np.nan),
            'avg_reference_length': gen.get('avg_reference_length', np.nan),
            'predictions': gen.get('predictions', []),
            'references': gen.get('references', []),
            'questions': gen.get('questions', []),
            'contexts': gen.get('contexts', []),
            'individual_bert_f1': individual_scores.get('bert_f1', []),
            'individual_token_f1': individual_scores.get('token_f1', []),
            'individual_exact_match': individual_scores.get('exact_match', []),
            'individual_semantic_similarity': individual_scores.get('semantic_similarity', []),
            'database_name': gen.get('database_name', None),
            'test_collection': gen.get('test_collection', None),
            'base_model_name': gen.get('base_model_name', None),
            'model_nick': gen.get('model_nick', None),
            # Nuove colonne TTR
            'individual_ttr': ttr_individual,
            'ttr': ttr_mean,
        }
        records.append(record)

    df_generations = pd.DataFrame(records)
    return df_generations