import streamlit as st
import pandas as pd
import joblib
import os
import gdown

# IDs de Drive
ID_SIMILARIDAD = 'https://drive.google.com/file/d/1QeJXG9lsECN77SgAKkQpYAh6mEEPefjV/view?usp=drive_link' 

DATA_DIR = 'data_processed'
SIM_PATH = os.path.join(DATA_DIR, 'similarity_matrix.joblib')
CLUS_PATH = os.path.join(DATA_DIR, 'clustering_model.pkl') 
MOVIES_PATH = 'data/movies.csv'

@st.cache_resource
def load_models():
    # Crea la carpeta si no existe en el servidor de la nube
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
    
    # Descarga solo si el archivo no está presente (útil para la nube)
    if not os.path.exists(SIM_PATH):
        with st.spinner('Descargando Matriz de Similitud desde Drive...'):
            url = f'https://drive.google.com/uc?id={ID_SIMILARIDAD}&confirm=t'
            gdown.download(url, SIM_PATH, quiet=False)
            
    # Carga local
    movies = pd.read_csv(MOVIES_PATH)
    similarity = joblib.load(SIM_PATH)
    # Carga el clustering si ya lo tienes en la carpeta o súbelo a GitHub si es pequeño
    clustering = joblib.load(CLUS_PATH) if os.path.exists(CLUS_PATH) else None
    
    return movies, similarity, clustering