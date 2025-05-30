import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'model')))

from flask import Flask, request, render_template, jsonify
import pandas as pd
import torch
from embedding_model import recommend_translators
from model_loader import load_models  
from tokenizer import tokenizer_func

app = Flask(__name__)

# Cargar modelos y CSV una sola vez
model_tasks, model_translators = load_models(
                                                
    task_path="../model/task_ae.pth",
    translator_path="../model/trans_ae.pth",
    task_dim=42,             #PREGUNTAR TASK_AE DIM
    translator_dim=42,       #PREGUNTAR TRANS_AE DIM
    latent_dim=42,           #PREGUNTAR LATENT DIM
    hidden_dim=64
                                            )
translators_df = pd.read_csv("../data/translators_enhanced.csv")

# Página de inicio con formulario
@app.route('/forms.html')
def form():
    return render_template('forms.html')

@app.route('/translators_avail.html')
def translators_avail():
    return render_template('translators_avail.html')

# Endpoint que procesa el formulario y devuelve traductores recomendados
@app.route('/recommend', methods=['POST'])
def recommend():
    # Recoge datos del formulario
    client_preferences = {
        'industry': request.form['industry'],
        'task_type': request.form['task_type'],
        'source_language': request.form['source_language'],
        'target_language': request.form['target_language'],
        'forecast' : request.form['forecast'],
        'hourly_rate': request.form['hourly_rate'],
        'pm': request.form['pm']
    }
    
    # Ejecuta recomendación
    recommended = recommend_translators(
        client_preferences=client_preferences,
        translators_df=translators_df,
        model_tasks=model_tasks,
        model_translators=model_translators,
        tokenizer=tokenizer_func
    )

    print("Contenido de recommended:")
    print(recommended)
    # Retorna página HTML con traductores recomendados
    return jsonify({"translators": recommended.to_dict(orient='records')})



if __name__ == '__main__':
    app.run(debug=True, port=8080)
