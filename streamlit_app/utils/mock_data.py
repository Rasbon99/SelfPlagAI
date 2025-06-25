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


def load_all_mock_data():
    """
    Restituisce un dizionario con prompt, metriche e risposte associati.
    Ogni chiave è un prompt, e il valore è un dict con 'metrics' e 'responses'.
    """
    data = {
        "Differenze tra apprendimento supervisionato e non supervisionato": {
            "metrics": pd.DataFrame({
                "Step Fine-Tuning": [1, 2, 3, 4, 5],
                "EM Score": [0.55, 0.65, 0.75, 0.88, 0.97],
                "F1 Score": [0.60, 0.72, 0.82, 0.91, 0.99]
            }),
            "responses": pd.DataFrame({
                "Step Fine-Tuning": [1, 2, 3, 4, 5],
                "Risposta Fine-Tuned": [
                    "Il supervisionato usa etichette, il non supervisionato no.",
                    "Dati etichettati vs non etichettati.",
                    "Il supervisionato richiede etichette; il non supervisionato trova pattern.",
                    "Uno prevede target noti, l'altro scopre struttura nei dati.",
                    "Il supervisionato apprende da dati etichettati, il non supervisionato cerca relazioni nei dati grezzi."
                ]
            })
        },

        "Funzionamento della blockchain in parole semplici": {
            "metrics": pd.DataFrame({
                "Step Fine-Tuning": [1, 2, 3, 4, 5],
                "EM Score": [0.50, 0.60, 0.70, 0.83, 0.92],
                "F1 Score": [0.58, 0.68, 0.77, 0.88, 0.95]
            }),
            "responses": pd.DataFrame({
                "Step Fine-Tuning": [1, 2, 3, 4, 5],
                "Risposta Fine-Tuned": [
                    "Una blockchain è un insieme di dati.",
                    "La blockchain è un registro di dati distribuiti.",
                    "È una catena di blocchi crittografici che registra transazioni.",
                    "Registro distribuito, immutabile e sicuro tramite crittografia.",
                    "La blockchain è un sistema distribuito che registra transazioni in modo trasparente, sicuro e immutabile tramite blocchi collegati e verificati."
                ]
            })
        }
    }

    return data

