from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer()

def tokenizer_func(client_preferences):
    # Convierte el diccionario a una sola string
    input_text = " ".join(str(value) for value in client_preferences.values())
    return vectorizer.fit_transform([input_text]).toarray()[0]

