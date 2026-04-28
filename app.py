import streamlit as st
import pandas as pd
import pickle
import os
import gdown

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="Movie Recommender System", page_icon="🎬", layout="centered")

# --- CONSTANTES ---
# ID de tu archivo similarity_matrix.pkl en Drive
FILE_ID = '139RiukkDCJchMOEiwEm4EQMIu9Reqhpj'
URL = f'https://drive.google.com/uc?id={FILE_ID}'

DATA_DIR = 'data_processed'
SIMILARITY_PATH = os.path.join(DATA_DIR, 'similarity_matrix.pkl')

# Archivo de películas que YA TIENES en tu carpeta data/ de GitHub
MOVIES_CSV_PATH = 'data/movies.csv' 

@st.cache_resource
def load_data():
    """Descarga la matriz y carga los datos."""
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

    # 1. Descargar la matriz de similitud si no existe
    if not os.path.exists(SIMILARITY_PATH):
        with st.spinner('Descargando base de conocimientos desde Drive (758MB)...'):
            # fuzzy=True es vital para saltar la advertencia de virus de Google Drive
            gdown.download(URL, SIMILARITY_PATH, quiet=False, fuzzy=True)
    
    # 2. Cargar los archivos
    try:
        # Cargar CSV de películas
        if not os.path.exists(MOVIES_CSV_PATH):
            st.error(f"Error: No se encuentra el archivo '{MOVIES_CSV_PATH}' en el repositorio.")
            return None, None
            
        movies = pd.read_csv(MOVIES_CSV_PATH)
        
        # Intentar cargar la matriz con pickle
        try:
            with open(SIMILARITY_PATH, 'rb') as f:
                similarity = pickle.load(f)
        except (pickle.UnpicklingError, AttributeError, EOFError, ImportError, IndexError):
            # Si el archivo está corrupto (invalid load key), lo borramos para forzar nueva descarga
            os.remove(SIMILARITY_PATH)
            st.warning("Se detectó una descarga incompleta. Por favor, recarga la página (F5) para reintentar.")
            st.stop()
            
        return movies, similarity

    except Exception as e:
        st.error(f"Error inesperado: {e}")
        return None, None

def recommend(movie, movies_df, similarity_matrix):
    """Lógica para obtener recomendaciones."""
    try:
        # Asegúrate de que la columna se llame 'title' en tu movies.csv
        index = movies_df[movies_df['title'] == movie].index[0]
        distances = sorted(list(enumerate(similarity_matrix[index])), reverse=True, key=lambda x: x[1])
        
        recommended_movies = []
        # Obtenemos las 6 películas más similares (saltando la primera que es ella misma)
        for i in distances[1:7]:
            recommended_movies.append(movies_df.iloc[i[0]].title)
        return recommended_movies
    except Exception as e:
        return [f"Lo siento, hubo un problema al buscar recomendaciones: {e}"]

# --- INTERFAZ DE USUARIO ---
def main():
    st.title('🎬 Movie Recommender System')
    st.write("Selecciona una película y te recomendaremos otras similares basándonos en sus características.")

    movies, similarity = load_data()
    
    if movies is not None and similarity is not None:
        # Selector de películas
        movie_list = movies['title'].values
        selected_movie = st.selectbox(
            "Escribe o selecciona una película que te guste:",
            movie_list
        )

        if st.button('Recomendar'):
            recommendations = recommend(selected_movie, movies, similarity)
            
            st.subheader(f"Si te gustó '{selected_movie}', deberías ver:")
            
            # Mostrar recomendaciones en tarjetas/columnas
            cols = st.columns(3)
            for idx, movie_name in enumerate(recommendations):
                with cols[idx % 3]:
                    st.success(movie_name)

if __name__ == '__main__':
    main()