from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer()

def tokenizer_func(client_preferences):
    text_fields = [
        client_preferences.get('manufacturer', ''),
        client_preferences.get('sector', ''),
        client_preferences.get('industry', ''),
        client_preferences.get('industry_group', ''),
        client_preferences.get('sub_industry', ''),
        client_preferences.get('task_type', ''),
        client_preferences.get('original_language', ''),
        client_preferences.get('target_language', ''),
        client_preferences.get('project_id'),
        client_preferences.get('budget'),
        client_preferences.get('pm', ''),
        client_preferences.get('finish_date', '')
    ]
    input_text = " ".join([str(val).strip().lower() for val in text_fields if val])
    if not input_text:
        raise ValueError("Empty input text for tokenizer.")
    return vectorizer.fit_transform([input_text]).toarray()[0]

