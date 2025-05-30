
import torch
from embedding_model import Task_AE, Translator_AE

def load_models(task_path="models/task_ae.pth",
                translator_path="models/trans_ae.pth",
                task_dim=8,
                translator_dim=42,
                latent_dim=42,
                hidden_dim=64,
                device=None):
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Crear las arquitecturas
    model_task = Task_AE(task_dim=task_dim, latent_dim=latent_dim, hidden_dim=hidden_dim).to(device)
    model_translator = Translator_AE(translator_dim=translator_dim, latent_dim=latent_dim, hidden_dim=hidden_dim).to(device)

    # Cargar pesos
    model_task.load_state_dict(torch.load(task_path, map_location=device))
    model_translator.load_state_dict(torch.load(translator_path, map_location=device))

    model_task.eval()
    model_translator.eval()

    return model_task, model_translator
