# src/__init__.py

"""
Paquete de Lógica del Sistema de Recomendación de Películas
Contiene las funciones de preprocesamiento de NLP y algoritmos de recomendación.
"""

# Importamos las funciones clave de los módulos internos
# Esto permite hacer: from src import tokenize, get_recommendations
from .preprocessing import tokenize
from .recommender import (
    get_recommendations, 
    ranking_movies, 
    recommendations
)

# Definimos qué funciones se exportan al usar: from src import *
__all__ = [
    'tokenize',
    'get_recommendations',
    'ranking_movies',
    'recommendations'
]

# Versión del paquete
__version__ = "1.0.0"