import pandas as pd

def load_mock_metrics():
    return pd.DataFrame({
        "Prompt ID": range(1, 6),
        "EM Score": [0.87, 0.76, 0.92, 0.65, 0.80],
        "F1 Score": [0.91, 0.78, 0.95, 0.70, 0.83]
    })

def load_mock_responses():
    return pd.DataFrame({
        "Prompt": [
            "Qual è la capitale della Francia?",
            "Chi ha scritto '1984'?",
            "Cosa fa il DNA?",
            "Quando è scoppiata la WWII?",
            "Dove si trova il Nilo?"
        ],
        "Risposta Base": [
            "Parigi", "George Orwell", "Trasporta informazioni genetiche",
            "Nel 1939", "In Africa"
        ],
        "Risposta Fine-Tuned": [
            "Parigi (Francia)", "Orwell", "Contiene il codice genetico",
            "1939", "Africa (lungo diversi paesi)"
        ]
    })