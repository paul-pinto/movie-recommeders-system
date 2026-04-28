import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import joblib
import os

def generar_matriz():
    print("Iniciando preparacion de datos...")
    path_csv = 'data/movies.csv'
    
    if not os.path.exists(path_csv):
        print(f"Error: No se encuentra {path_csv}")
        return

    # Cargamos el CSV
    df = pd.read_csv(path_csv)
    
    # Usamos 'genres' que es la columna que si existe en tu archivo
    columna_texto = 'genres'
    
    print(f"Usando la columna: '{columna_texto}' para calcular similitudes...")
    
    # 2. Procesamiento TF-IDF
    # El token_pattern ayuda a que generos como "Sci-Fi" no se rompan
    tfidf = TfidfVectorizer(stop_words='english', token_pattern=r'[a-zA-Z0-9\-]+')
    tfidf_matrix = tfidf.fit_transform(df[columna_texto].fillna(''))
    
    # 3. Calcular similitud
    print("Calculando matriz de similitud de cosenos...")
    similarity = cosine_similarity(tfidf_matrix)
    
    # 4. Guardar
    if not os.path.exists('data_processed'):
        os.makedirs('data_processed')
        
    print("Guardando similarity_matrix.joblib...")
    joblib.dump(similarity, 'data_processed/similarity_matrix.joblib', compress=3)
    print("Exito: Archivo generado correctamente.")

if __name__ == "__main__":
    generar_matriz()