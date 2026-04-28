# src/recommender.py
import numpy as np
import pandas as pd

def get_recommendations(data, user_id, top_n, algo):
    """Genera recomendaciones base por Clustering."""
    user_item_matrix = data.pivot(index='userId', columns='movieId', values='rating')
    if user_id not in user_item_matrix.index:
        return []
    user_data = user_item_matrix.loc[user_id]
    non_interacted_movies = user_data[user_data.isnull()].index.tolist()
    
    recommendations = []
    for item_id in non_interacted_movies:
        est = algo.predict(user_id, item_id).est
        recommendations.append((item_id, est))
    
    recommendations.sort(key=lambda x: x[1], reverse=True)
    return recommendations[:top_n]

def ranking_movies(recommendations, final_rating_stats, movies_df):
    """Añade títulos y aplica la fórmula de ranking corregido."""
    movie_ids = [item[0] for item in recommendations]
    existing_ids = [m_id for m_id in movie_ids if m_id in final_rating_stats.index]
    
    ranked_movies = final_rating_stats.loc[existing_ids].copy().reset_index()
    preds_df = pd.DataFrame(recommendations, columns=['movieId', 'predicted_ratings'])
    ranked_movies = ranked_movies.merge(preds_df, on='movieId')
    ranked_movies = ranked_movies.merge(movies_df[['movieId', 'title', 'genres']], on='movieId')
    
    ranked_movies['corrected_ratings'] = (
        ranked_movies['predicted_ratings'] - 1 / np.sqrt(ranked_movies['rating_count'])
    )
    
    return ranked_movies.sort_values('corrected_ratings', ascending=False)[
        ['title', 'genres', 'rating_count', 'predicted_ratings', 'corrected_ratings']
    ]

def recommendations(title, similar_movies_matrix, final_ratings_df, top_n=10):
    """Recomendación por similitud de contenido (NLP)."""
    indices = pd.Series(final_ratings_df.index)
    if title not in indices.values:
        return []
    idx = indices[indices == title].index[0]
    score_series = pd.Series(similar_movies_matrix[idx]).sort_values(ascending=False)
    top_indexes = list(score_series.iloc[1 : top_n + 1].index)
    return [list(final_ratings_df.index)[i] for i in top_indexes]