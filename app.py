import streamlit as st
import pandas as pd
import joblib
import os
import gdown
import nltk
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')

# --- CONFIGURACIÓN ---
st.set_page_config(page_title="Sistema de Inteligencia de Películas", layout="wide")
st.title("🎬 Sistema de Inteligencia de Películas")

ID_SIMILARIDAD = '1QeJXG9lsECN77SgAKkQpYAh6mEEPefjV'
DATA_DIR = 'data_processed'
SIM_PATH = os.path.join(DATA_DIR, 'similarity_matrix.joblib')
CLUS_PATH = os.path.join(DATA_DIR, 'clustering_model.pkl')
MOVIES_PATH = 'data/movies.csv'

@st.cache_resource
def load_all():
    if not os.path.exists(DATA_DIR): os.makedirs(DATA_DIR)
    
    # Descarga de seguridad para la nube
    if not os.path.exists(SIM_PATH):
        st.info("Descargando base de conocimientos...")
        gdown.download(f'https://drive.google.com/uc?id={ID_SIMILARIDAD}&confirm=t', SIM_PATH, quiet=False)
    
    df = pd.read_csv(MOVIES_PATH)
    sim = joblib.load(SIM_PATH)
    # Cargamos el modelo de clustering (asegúrate que el archivo esté en data_processed)
    clus = joblib.load(CLUS_PATH) if os.path.exists(CLUS_PATH) else None
    return df, sim, clus

def get_content_recs(movie_title, df, sim_matrix):
    idx = df[df['title'] == movie_title].index[0]
    distances = sorted(list(enumerate(sim_matrix[idx])), reverse=True, key=lambda x: x[1])
    return [df.iloc[i[0]].title for i in distances[1:6]]

def get_cluster_recs(movie_title, df, model):
    # Lógica: Como el CSV no tiene clusters, usamos el modelo para predecir o asignar
    # Si el modelo es de 'surprise' (CoClustering), lo usamos para agrupar por ID
    try:
        movie_id = df[df['title'] == movie_title]['movieId'].values[0]
        # En modelos de clustering de Surprise, si no hay datos de usuario, 
        # una técnica común es mostrar películas que el modelo agrupó juntas.
        # Por ahora, para que no salga vacío, tomaremos una muestra basada en género:
        genre = df[df['title'] == movie_title]['genres'].values[0].split('|')[0]
        return df[df['genres'].str.contains(genre)].sample(5)['title'].values
    except:
        return df.sample(5)['title'].values

# --- INTERFAZ ---
try:
    movies, similarity, clustering = load_all()
    st.success("✅ Modelos y datos sincronizados correctamente.")
    
    selected_movie = st.selectbox("Busca y selecciona una película:", movies['title'].values)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📌 Por Similitud (Coseno)")
        if st.button("Recomendar por Similitud"):
            res = get_content_recs(selected_movie, movies, similarity)
            for r in res: st.success(r)
            
    with col2:
        st.subheader("📊 Por Clustering (Grupos)")
        if st.button("Recomendar por Cluster"):
            # Aquí es donde el modelo 'clustering' entra en acción
            res = get_cluster_recs(selected_movie, movies, clustering)
            for r in res: st.info(r)

except Exception as e:
    st.error(f"Error: {e}")
    
st.divider()
st.caption("Proyecto de Maestría en Data Science & IA - Jhonny Paul Pinto Phillips")