import streamlit as st
import pandas as pd
import pickle
import os
import gdown

# --- CONFIGURACIÓN ---
st.set_page_config(page_title="Movie Recommender", page_icon="🎬")

# ID de tu matriz en Drive (Este no cambia)
FILE_ID = '139RiukkDCJchMOEiwEm4EQMIu9Reqhpj'
URL = f'https://drive.google.com/uc?id={FILE_ID}'

DATA_DIR = 'data_processed'
SIMILARITY_PATH = os.path.join(DATA_DIR, 'similarity_matrix.pkl')

# USANDO EL NOMBRE EXACTO QUE VEO EN TU FOTO
MOVIES_CSV_PATH = 'data/movies.csv' 

@st.cache_resource
def load_data():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

    # 1. Descargar matriz si no existe
    if not os.path.exists(SIMILARITY_PATH):
        with st.spinner('Descargando matriz de similitud...'):
            gdown.download(URL, SIMILARITY_PATH, quiet=False)
    
    try:
        # 2. Cargar tu archivo movies.csv
        movies = pd.read_csv(MOVIES_CSV_PATH)
        
        # 3. Cargar la matriz
        with open(SIMILARITY_PATH, 'rb') as f:
            similarity = pickle.load(f)
            
        return movies, similarity
    except Exception as e:
        st.error(f"Error al cargar archivos: {e}")
        return None, None

def recommend(movie, movies_df, similarity_matrix):
    try:
        # En tu movies.csv la columna debe ser 'title' o el nombre exacto que tengas
        index = movies_df[movies_df['title'] == movie].index[0]
        distances = sorted(list(enumerate(similarity_matrix[index])), reverse=True, key=lambda x: x[1])
        return [movies_df.iloc[i[0]].title for i in distances[1:7]]
    except:
        return ["No se encontraron recomendaciones."]

# --- INTERFAZ ---
def main():
    st.title('🎬 Movie Recommender System')

    # Verificación de seguridad
    if not os.path.exists(MOVIES_CSV_PATH):
        st.error(f"No encuentro el archivo '{MOVIES_CSV_PATH}'. Revisa que esté en GitHub.")
        return

    movies, similarity = load_data()
    
    if movies is not None:
        # Asumo que la columna se llama 'title'. Si es 'Title', cámbialo abajo:
        movie_list = movies['title'].values 
        selected_movie = st.selectbox("Selecciona una película:", movie_list)

        if st.button('Recomendar'):
            recs = recommend(selected_movie, movies, similarity)
            for r in recs:
                st.write(f"- {r}")

if __name__ == '__main__':
    main()