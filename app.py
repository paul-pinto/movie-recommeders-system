import streamlit as st
import pandas as pd
import pickle
import os
import gdown

# --- CONFIGURACIÓN ---
st.set_page_config(page_title="Movie Recommender", page_icon="🎬")

# Revisamos el Drive
FILE_ID = '139RiukkDCJchMOEiwEm4EQMIu9Reqhpj'
URL = f'https://drive.google.com/uc?id={FILE_ID}'

DATA_DIR = 'data_processed'
SIMILARITY_PATH = os.path.join(DATA_DIR, 'similarity_matrix.pkl')
# Ruta al CSV que está en el GitHub
MOVIES_CSV_PATH = 'data/movies_cleaned.csv' 

@st.cache_resource
def load_data():
    # Crear carpeta para la matriz si no existe
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

    # 1. Descargar la matriz de similitud desde Drive si no existe
    if not os.path.exists(SIMILARITY_PATH):
        with st.spinner('Descargando matriz de similitud (723MB). Esto solo ocurre una vez...'):
            gdown.download(URL, SIMILARITY_PATH, quiet=False)
    
    # 2. Cargar los archivos
    try:
        # Cargamos el CSV que está en la carpeta data/
        movies = pd.read_csv(MOVIES_CSV_PATH)
        
        # Cargamos la matriz descargada
        with open(SIMILARITY_PATH, 'rb') as f:
            similarity = pickle.load(f)
            
        return movies, similarity
    except Exception as e:
        st.error(f"Error al cargar los archivos: {e}")
        return None, None

def recommend(movie, movies_df, similarity_matrix):
    try:
        # Encontrar el índice de la película
        index = movies_df[movies_df['title'] == movie].index[0]
        # Calcular distancias
        distances = sorted(list(enumerate(similarity_matrix[index])), reverse=True, key=lambda x: x[1])
        
        # Obtener los títulos de las 6 más parecidas
        recommended_movie_names = []
        for i in distances[1:7]:
            recommended_movie_names.append(movies_df.iloc[i[0]].title)
        return recommended_movie_names
    except Exception as e:
        return [f"Error en recomendación: {e}"]

# --- INTERFAZ DE USUARIO ---
def main():
    st.title('🎬 Movie Recommender System')
    st.markdown("Selecciona una película para ver recomendaciones basadas en contenido.")

    # Verificar que el CSV existe antes de cargar todo
    if not os.path.exists(MOVIES_CSV_PATH):
        st.error(f"No se encuentra el archivo {MOVIES_CSV_PATH} en la carpeta data/")
        return

    movies, similarity = load_data()
    
    if movies is not None and similarity is not None:
        # Crear el buscador con la columna 'title' del CSV
        movie_list = movies['title'].values
        selected_movie = st.selectbox("Escribe o selecciona una película:", movie_list)

        if st.button('Obtener Recomendaciones'):
            recommendations = recommend(selected_movie, movies, similarity)
            
            st.subheader("Te recomendamos:")
            cols = st.columns(3)
            for idx, name in enumerate(recommendations):
                with cols[idx % 3]:
                    st.success(name)

if __name__ == '__main__':
    main()