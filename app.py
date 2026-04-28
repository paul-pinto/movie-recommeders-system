import streamlit as st
import pandas as pd
import pickle
import os
import gdown

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="Movie Recommender System", page_icon="🎬", layout="centered")

# --- CONSTANTES ---
# ID de tu matriz en Drive (758MB)
FILE_ID = '139RiukkDCJchMOEiwEm4EQMIu9Reqhpj'
# La URL incluye &confirm=t para saltar la advertencia de archivos grandes
URL = f'https://drive.google.com/uc?id={FILE_ID}&confirm=t'

DATA_DIR = 'data_processed'
SIMILARITY_PATH = os.path.join(DATA_DIR, 'similarity_matrix.pkl')
# Ruta exacta al archivo que tienes en tu carpeta data/
MOVIES_CSV_PATH = 'data/movies.csv' 

@st.cache_resource
def load_data():
    """Gestiona la descarga de la matriz y la carga de datos."""
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

    # 1. Descargar la matriz de similitud si no existe
    if not os.path.exists(SIMILARITY_PATH):
        with st.spinner('Descargando base de conocimientos (758MB). Por favor, ten paciencia...'):
            try:
                gdown.download(URL, SIMILARITY_PATH, quiet=False)
            except Exception as e:
                st.error(f"Error al descargar desde Drive: {e}")
                return None, None
    
    # 2. Cargar los archivos
    try:
        # Verificar que el CSV de películas existe en el repo
        if not os.path.exists(MOVIES_CSV_PATH):
            st.error(f"No se encuentra el archivo '{MOVIES_CSV_PATH}' en la carpeta data/ de GitHub.")
            return None, None
            
        movies = pd.read_csv(MOVIES_CSV_PATH)
        
        # Intentar cargar la matriz con pickle
        try:
            with open(SIMILARITY_PATH, 'rb') as f:
                similarity = pickle.load(f)
        except Exception:
            # Si el archivo está corrupto (ej. error de 'invalid load key'), lo borramos
            if os.path.exists(SIMILARITY_PATH):
                os.remove(SIMILARITY_PATH)
            st.warning("Se detectó una descarga incompleta o corrupta. Por favor, recarga la página (F5) para reintentar.")
            st.stop()
            
        return movies, similarity

    except Exception as e:
        st.error(f"Error inesperado durante la carga: {e}")
        return None, None

def recommend(movie, movies_df, similarity_matrix):
    """Lógica principal de recomendación."""
    try:
        # Buscamos el índice de la película por título
        index = movies_df[movies_df['title'] == movie].index[0]
        # Obtenemos las distancias de la matriz
        distances = sorted(list(enumerate(similarity_matrix[index])), reverse=True, key=lambda x: x[1])
        
        # Seleccionamos las 6 más parecidas (excluyendo la primera que es la misma película)
        return [movies_df.iloc[i[0]].title for i in distances[1:7]]
    except Exception as e:
        return [f"Error al procesar: {e}"]

# --- INTERFAZ PRINCIPAL ---
def main():
    st.title('🎬 Movie Recommender System')
    st.write("Selecciona una película para ver recomendaciones inteligentes basadas en contenido.")

    movies, similarity = load_data()
    
    if movies is not None and similarity is not None:
        # Creamos la lista desplegable usando la columna 'title' del CSV
        movie_list = movies['title'].values
        selected_movie = st.selectbox(
            "Busca o selecciona una película:",
            movie_list
        )

        if st.button('Obtener Recomendaciones'):
            recommendations = recommend(selected_movie, movies, similarity)
            
            st.subheader("Películas recomendadas para ti:")
            
            # Mostramos los resultados de forma elegante
            cols = st.columns(2)
            for idx, movie_name in enumerate(recommendations):
                with cols[idx % 2]:
                    st.success(movie_name)

if __name__ == '__main__':
    main()