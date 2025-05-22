from flask import Flask, request, render_template, jsonify
import pandas as pd
import torch
from embedding_model import recommend_translators
from model_loader import load_models  
from tokenizer import tokenizer_func

app = Flask(__name__)

# Cargar modelos y CSV una sola vez
model_tasks, model_translators = load_models(
                                                
    task_path="models/task_ae.pth",
    translator_path="models/trans_ae.pth",
    task_dim=...,             #PREGUNTAR TASK_AE DIM
    translator_dim=...,       #PREGUNTAR TRANS_AE DIM
    latent_dim=...,           #PREGUNTAR LATENT DIM
                                            )
translators_df = pd.read_csv("translators_enhanced.csv")

# Página de inicio con formulario
@app.route('/forms.html')
def form():
    return render_template('forms.html')

# Endpoint que procesa el formulario y devuelve traductores recomendados
@app.route('/recommend', methods=['POST'])
def recommend():
    # Recoge datos del formulario
    client_preferences = {
        'manufacturer': request.form['manufacturer'],
        'sector': request.form['sector'],
        'industry': request.form['industry'],
        'industry_group': request.form['industry_group'],
        'sub_industry': request.form['sub_industry'],
        'task_type': request.form['task_type'],
        'original_language': request.form['original_language'],
        'target_language': request.form['target_language'],
        'budget': float(request.form['budget']),
        'project_id': request.form['project_id'],
        'finish_date': request.form['finish_date']
    }

    # Ejecuta recomendación
    recommended = recommend_translators(
        client_preferences=client_preferences,
        translators_df=translators_df,
        model_tasks=model_tasks,
        model_translators=model_translators,
        tokenizer=tokenizer_func
    )

    # Retorna página HTML con traductores recomendados
    return render_template('translators_avail.html', translators=recommended)

if __name__ == '__main__':
    app.run(debug=True)
