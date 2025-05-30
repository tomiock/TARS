from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

vectorizer = TfidfVectorizer()



def tokenizer_func(client_preferences):
    text_fields = [
        client_preferences.get('industry', ''),
        client_preferences.get('task_type', ''),
        client_preferences.get('source_language', ''),
        client_preferences.get('target_language', ''),
        client_preferences.get('pm', ''),
        client_preferences.get('hourly_rate', ''),
        client_preferences.get('forecast', '')

    ]
    input_text = " ".join([str(val).strip().lower() for val in text_fields if val])
    if not input_text:
        raise ValueError("Empty input text for tokenizer.")
    vec = vectorizer.fit_transform([input_text]).toarray()[0]
    if len(vec) < 42:
        vec = np.pad(vec, (0, 42 - len(vec)), mode='constant')
    elif len(vec) > 42:
        vec = vec[:42]  # Truncar
    return vec

