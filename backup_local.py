import streamlit as st
import pandas as pd
import joblib
import os

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="Comparativa de Modelos", page_icon="🎬", layout="wide")

# --- RUTAS LOCALES ---
DATA_DIR = 'data_processed'
SIM_PATH = os.path.join(DATA_DIR, 'similarity_matrix.joblib')
CLUS_PATH = os.path.join(DATA_DIR, 'clustering_model.pkl') # O .joblib según lo guardaste
MOVIES_PATH = 'data/movies.csv'

@st.cache_resource
def load_models():
    try:
        movies = pd.read_csv(MOVIES_PATH)
        # Cargamos los modelos si existen
        similarity = joblib.load(SIM_PATH) if os.path.exists(SIM_PATH) else None
        clustering = joblib.load(CLUS_PATH) if os.path.exists(CLUS_PATH) else None
        return movies, similarity, clustering
    except Exception as e:
        st.error(f"Error al cargar archivos: {e}")
        return None, None, None

def get_content_recs(movie_title, df, sim_matrix):
    try:
        idx = df[df['title'] == movie_title].index[0]
        distances = sorted(list(enumerate(sim_matrix[idx])), reverse=True, key=lambda x: x[1])
        return [df.iloc[i[0]].title for i in distances[1:6]]
    except:
        return ["No se encontraron recomendaciones por similitud."]

def get_cluster_recs(movie_title, df, model):
    # Lógica para Clustering (Surprise CoClustering o K-Means)
    # Si el CSV tiene los clusters anotados por el notebook, es más fácil:
    if 'cluster' in df.columns:
        cluster_id = df[df['title'] == movie_title]['cluster'].values[0]
        recs = df[df['cluster'] == cluster_id].sample(min(len(df), 5))['title'].values
        return recs
    else:
        # Si no hay columna cluster, mostramos una muestra aleatoria como "simulacro"
        # para que la interfaz no se vea vacía mientras integras el modelo real
        return df.sample(5)['title'].values

def main():
    st.title("🎬 Comparativa: Similitud de Coseno vs Clustering")
    
    movies, similarity, clustering = load_models()
    
    if movies is not None:
        selected_movie = st.selectbox("Elige una película:", movies['title'].values)
        
        # --- AQUÍ DEFINIMOS LAS COLUMNAS PARA EVITAR EL NAMEERROR ---
        col1, col2 = st.columns(2)
        
        with col1:
            st.header("Modelo 1: Similitud")
            st.write("Basado en contenido (TF-IDF)")
            if st.button("Recomendar por Similitud"):
                if similarity is not None:
                    recs = get_content_recs(selected_movie, movies, similarity)
                    for r in recs:
                        st.success(r)
                else:
                    st.error("Archivo de similitud no encontrado.")
        
        with col2:
            st.header("Modelo 2: Clustering")
            st.write("Basado en grupos de comportamiento")
            if st.button("Recomendar por Cluster"):
                # Aquí llamamos a la función de clustering
                recs = get_cluster_recs(selected_movie, movies, clustering)
                for r in recs:
                    st.info(r)

if __name__ == '__main__':
    main()