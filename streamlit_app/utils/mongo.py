# Importa MongoClient per la connessione a MongoDB
from pymongo import MongoClient

# Importa os per leggere variabili d'ambiente dal sistema (es. MONGO_URI)
import os

# Carica automaticamente le variabili definite in un file .env
from dotenv import load_dotenv
load_dotenv()

def get_mongo_client():
    """
    Crea e restituisce un client MongoDB usando l'URI specificato nella variabile d'ambiente MONGO_URI.
    Se MONGO_URI non è presente, viene usato di default 'mongodb://localhost:27017' (istanza locale).
    
    Questo client rappresenta la connessione al server MongoDB (locale o remoto).
    """
    uri = os.getenv("MONGO_URI", "mongodb://localhost:27017")
    return MongoClient(uri)

def get_db():
    """
    Ritorna il database principale del progetto.
    Per convenzione abbiamo scelto 'selfplagai_db', che identifica chiaramente il progetto SelfPlagAI.
    
    MongoDB non richiede di creare esplicitamente un database: basta usarlo e verrà generato.
    """
    client = get_mongo_client()
    return client["selfplagai_db"]  # Puoi rinominarlo ad es. in "qa_evaluation_db" se preferisci

def get_collection(name="results"):
    """
    Restituisce una collezione specifica dal database, di default 'results'.

    Esempi:
    - 'results': per salvare le risposte generate dai modelli (prompt, base, fine-tuned)
    - 'metrics': per salvare le metriche di valutazione (EM, F1, SAS)
    - 'logs': per salvare cronologia, timestamp o meta-informazioni

    Il nome può essere cambiato dinamicamente quando si chiama la funzione.
    """
    return get_db()[name]
