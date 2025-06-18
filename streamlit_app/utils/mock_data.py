import pandas as pd

def load_mock_metrics():
    """
    Simula l'andamento delle metriche EM/F1 per una singola domanda,
    con risposte migliorate in più step di fine-tuning.
    """
    return pd.DataFrame({
        "Step Fine-Tuning": [1, 2, 3, 4, 5],
        "EM Score": [0.55, 0.65, 0.75, 0.88, 0.97],
        "F1 Score": [0.60, 0.72, 0.82, 0.91, 0.99]
    })

def load_mock_responses():
    """
    Simula una domanda complessa con risposte sempre più dettagliate
    attraverso vari step di fine-tuning.
    """
    return pd.DataFrame({
        "Prompt": [
            "Quali sono le principali differenze tra apprendimento supervisionato e non supervisionato nel machine learning?"
        ] * 5,
        "Step Fine-Tuning": [1, 2, 3, 4, 5],
        "Risposta Base": [
            "L'apprendimento supervisionato usa etichette, quello non supervisionato no."
        ] * 5,
        "Risposta Fine-Tuned": [
            "Il supervisionato usa etichette, il non supervisionato no.",
            "Nel supervisionato i dati sono etichettati, nel non supervisionato no.",
            "L'apprendimento supervisionato richiede dati etichettati, mentre il non supervisionato lavora con dati non etichettati.",
            "Il supervisionato prevede l'uso di set di dati con etichette note per addestrare un modello, mentre il non supervisionato cerca di trovare strutture nei dati senza etichette.",
            "Il supervisionato si basa su dati etichettati per apprendere una funzione obiettivo (es. classificazione), mentre il non supervisionato analizza dati grezzi per scoprire pattern nascosti (es. clustering, riduzione dimensionale)."
        ]
    })

