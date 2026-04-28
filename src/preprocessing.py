# src/preprocessing.py

import re
import nltk
from nltk.tokenize import wordpunct_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

_NLTK_READY = False


def ensure_resources():
    """
    Verifica recursos NLTK de forma perezosa.
    Si no hay internet, no rompe el arranque de la app.
    """
    global _NLTK_READY
    if _NLTK_READY:
        return
    try:
        stopwords.words("english")
        WordNetLemmatizer().lemmatize("movies")
        _NLTK_READY = True
        return
    except LookupError:
        pass

    resources = ['omw-1.4', 'punkt', 'punkt_tab', 'stopwords', 'wordnet']
    for res in resources:
        try:
            nltk.download(res, quiet=True)
        except Exception:
            continue

    try:
        stopwords.words("english")
        _NLTK_READY = True
    except LookupError:
        _NLTK_READY = False

def tokenize(text):
    """
    Limpia, tokeniza y lematiza el texto de entrada.
    
    Pasos:
    1. Convierte a minúsculas y elimina caracteres no alfabéticos.
    2. Divide el texto en tokens.
    3. Elimina 'stopwords' (palabras sin valor semántico).
    4. Aplica Lemmatization para reducir las palabras a su raíz.
    """
    if not isinstance(text, str):
        return []

    ensure_resources()

    # 1. Limpieza: Solo letras y minúsculas
    text = re.sub(r"[^a-zA-Z]", " ", text.lower())

    # 2. Tokenización (usamos wordpunct para mayor compatibilidad)
    tokens = wordpunct_tokenize(text)

    # 3. Eliminar stopwords
    if _NLTK_READY:
        stop_words = set(stopwords.words("english"))
    else:
        stop_words = set()
    words = [word for word in tokens if word not in stop_words]

    # 4. Lemmatization
    if _NLTK_READY:
        lemmatizer = WordNetLemmatizer()
        text_lems = [lemmatizer.lemmatize(lem).strip() for lem in words]
    else:
        text_lems = [w.strip() for w in words]

    return text_lems
