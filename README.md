# 🎬 MovieAdvisor AI: Sistema de Recomendación Híbrido

Este proyecto presenta un sistema de recomendación de películas de alto rendimiento que combina estrategias de **Filtrado Colaborativo (Clustering)** y **Filtrado Basado en Contenido (NLP)**. Desarrollado como parte de la Maestría en Ciencia de Datos e IA.

## 🚀 Características Principales

- **Motor de Clustering:** Implementación de *Co-Clustering* (bi-clustering) para agrupar usuarios y películas simultáneamente, optimizando la precisión en las predicciones de calificación.
- **Motor de Contenido:** Algoritmo de procesamiento de lenguaje natural (NLP) con *TF-IDF* y *Similitud de Coseno* para recomendar títulos basados en géneros y etiquetas personalizadas.
- **Ranking Corregido:** Algoritmo de post-procesamiento que penaliza ítems con pocas interacciones para asegurar la confiabilidad de las recomendaciones.
- **Interfaz Interactiva:** Despliegue en *Streamlit* para una exploración de datos en tiempo real.

## 🛠️ Tecnologías Utilizadas

- **Lenguaje:** Python 3.11+
- **Modelado:** Scikit-Surprise, Scikit-Learn
- **NLP:** NLTK (Tokenización, Lemmatización)
- **Manipulación de Datos:** Pandas, NumPy
- **Despliegue:** Streamlit, GitHub Actions

## 📊 Metodología y Resultados

1. **Análisis Exploratorio (EDA):** Procesamiento de más de 100,000 calificaciones y 9,000 títulos únicos.
2. **Optimización:** Ajuste de hiperparámetros mediante *GridSearchCV* para minimizar el RMSE en el modelo de Clustering.
3. **Validación:** El sistema basado en contenido demostró una alta cohesión temática, recomendando géneros similares (ej. Crimen/Misterio) para títulos del mismo perfil.

## 📂 Estructura del Proyecto

- `app.py`: Interfaz de usuario.
- `src/`: Lógica modular (preprocesamiento y recomendadores).
- `notebooks/`: Investigación y entrenamiento de modelos.
- `data/`: Datasets procesados y modelos serializados (.pkl).

## ▶️ Cómo Ejecutar El Proyecto

### 1) Crear y activar entorno virtual

```powershell
python -m venv .venv
.\.venv\Scripts\activate
```

### 2) Instalar dependencias

```powershell
python -m pip install -r requirements.txt
```

### 3) Exportar artefactos requeridos por `app.py`

Ejecuta esta celda al final de tu notebook `notebooks/proyecto_3.ipynb` (después de entrenar y crear las variables):

```python
import os
import joblib

os.makedirs("data_processed", exist_ok=True)

# Ajuste de nombres para que coincidan con app.py
ratings_stats = final_rating              # DataFrame indexado por movieId con columna rating_count
similarity_matrix = similar_movies        # Matriz de similitud de coseno
algo_clustering = clust_tuned             # Modelo entrenado CoClustering
final_ratings_content = final_ratings     # DataFrame indexado por title (motor contenido)

joblib.dump(ratings_stats, "data_processed/final_rating_stats.pkl")
joblib.dump(similarity_matrix, "data_processed/similarity_matrix.pkl")
joblib.dump(algo_clustering, "data_processed/clustering_model.pkl")
joblib.dump(final_ratings_content, "data_processed/final_ratings_content.pkl")

print("Artefactos exportados en ./data_processed")
```

### 4) Ejecutar Streamlit

```powershell
python -m streamlit run app.py
```

Si aparece un error de NLTK la primera vez, ejecuta:

```powershell
python -c "import nltk; [nltk.download(x) for x in ['omw-1.4','punkt','punkt_tab','stopwords','wordnet']]"
```

---
**Autor:** Jhonny Paul Pinto Phillips  
**Ubicación:** Guayaramerín, Bolivia | 2026
