import pandas as pd

def load_mock_metrics():
    """
    Simula l'andamento delle metriche EM/F1 per una singola domanda,
    con risposte migliorate in più step di fine-tuning.
    """
    return pd.DataFrame({
        "Step Fine-Tuning": [1, 2, 3, 4, 5],
        "EM Score": [0.70, 0.80, 0.88, 0.95, 1.00],
        "F1 Score": [0.75, 0.85, 0.90, 0.96, 1.00]
    })

def load_mock_responses():
    """
    Simula una singola domanda con la risposta base
    e 5 versioni di risposta fine-tuned progressivamente migliori.
    """
    return pd.DataFrame({
        "Prompt": ["Qual è la capitale della Francia?"] * 5,
        "Step Fine-Tuning": [1, 2, 3, 4, 5],
        "Risposta Base": ["Parigi"] * 5,
        "Risposta Fine-Tuned": [
            "parigi",
            "Parigi è la capitale.",
            "La capitale della Francia è Parigi.",
            "La capitale della Francia è Parigi, una città europea.",
            "Parigi è la capitale della Francia, situata in Europa."
        ]
    })
