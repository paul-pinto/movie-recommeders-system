import streamlit as st
import pandas as pd
import pickle
import os
import gdown

# --- CONFIGURACIÓN ---
st.set_page_config(page_title="Movie Recommender", page_icon="🎬")

FILE_ID = '139RiukkDCJchMOEiwEm4EQMIu9Reqhpj'
URL = f'https://drive.google.com/uc?id={FILE_ID}'
DATA_DIR = 'data_processed'
SIMILARITY_PATH = os.path.join(DATA_DIR, 'similarity_matrix.pkl')
# Asegúrate de que este archivo sí lo subas a GitHub (pesa poco)
MOVIES_DICT_PATH = os.path.join(DATA_DIR, 'movie_dict.pkl') 

@st.cache_resource
def load_data():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

    if not os.path.exists(SIMILARITY_PATH):
        with st.spinner('Descargando matriz de similitud (723MB)...'):
            gdown.download(URL, SIMILARITY_PATH, quiet=False)
    
    try:
        movies_dict = pickle.load(open(MOVIES_DICT_PATH, 'rb'))
        movies = pd.DataFrame(movies_dict)
        similarity = pickle.load(open(SIMILARITY_PATH, 'rb'))
        return movies, similarity
    except Exception as e:
        st.error(f"Error al cargar datos: {e}")
        return None, None

def recommend(movie, movies_df, similarity_matrix):
    index = movies_df[movies_df['title'] == movie].index[0]
    distances = sorted(list(enumerate(similarity_matrix[index])), reverse=True, key=lambda x: x[1])
    return [movies_df.iloc[i[0]].title for i in distances[1:7]]

# --- INTERFAZ ---
def main():
    st.title('🎬 Movie Recommender System')
    movies, similarity = load_data()
    
    if movies is not None:
        selected_movie = st.selectbox("Elige una película:", movies['title'].values)
        if st.button('Recomendar'):
            recs = recommend(selected_movie, movies, similarity)
            cols = st.columns(3)
            for i, name in enumerate(recs):
                with cols[i % 3]:
                    st.success(name)

if __name__ == '__main__':
    main()