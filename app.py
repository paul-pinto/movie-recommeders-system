import streamlit as st
import pandas as pd
import joblib
import os
import gdown

# 1. Configuración inmediata (Lo primero que ve el navegador)
st.set_page_config(page_title="Movie Recommender", layout="wide")
st.title("🎬 Sistema de Inteligencia de Películas")

# --- RUTAS E IDS ---
ID_SIMILARIDAD = '1QeJXG9lsECN77SgAKkQpYAh6mEEPefjV'
DATA_DIR = 'data_processed'
SIM_PATH = os.path.join(DATA_DIR, 'similarity_matrix.joblib')
MOVIES_PATH = 'data/movies.csv'

# 2. Función de carga con mensajes visuales
@st.cache_resource
def load_all():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
    
    if not os.path.exists(SIM_PATH):
        # Usamos st.info para que el usuario sepa que algo está pasando
        st.info("Descargando base de conocimientos desde Drive (119MB)...")
        url = f'https://drive.google.com/uc?id={ID_SIMILARIDAD}&confirm=t'
        gdown.download(url, SIM_PATH, quiet=False)
    
    # Carga de datos
    df = pd.read_csv(MOVIES_PATH)
    sim = joblib.load(SIM_PATH)
    return df, sim

# 3. Ejecución principal
try:
    movies, similarity = load_all()
    st.success("✅ Modelos cargados correctamente.")
    
    # Aquí pones el selector de películas y el resto de tu lógica de columnas
    selected_movie = st.selectbox("Selecciona una película:", movies['title'].values)
    
    # ... resto del código (col1, col2, etc.)
    
except Exception as e:
    st.error(f"Error crítico al iniciar: {e}")
    st.info("Revisa si el archivo CSV está en la carpeta 'data/'.")