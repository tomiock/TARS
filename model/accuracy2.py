import torch
import pandas as pd
import numpy as np
import wandb

from pytorch_metric_learning.losses import TripletMarginLoss
from pytorch_metric_learning.miners import TripletMarginMiner, BaseMiner
from pytorch_metric_learning.distances import LpDistance, CosineSimilarity

from torch.utils.data import Dataset, DataLoader
from torch import nn

from tqdm import tqdm # For progress bars

from typing import Optional, Callable, Dict, List

# --- Utility Function ---
def create_features(features: List[str], data: pd.Series) -> List[float]:
    """
    Extracts and flattens specified features from a Pandas Series.
    Args:
        features: A list of column names to extract.
        data: A Pandas Series representing a row of data.
    Returns:
        A list of floats representing the flattened features.
    """
    return_data = []
    for col in features:
        col_data = data[col]
        if isinstance(col_data, (list, np.ndarray)):
            return_data.extend(col_data)
        elif isinstance(col_data, (int, float, np.number)):
            return_data.append(float(col_data))
    return return_data

# --- Dataset Definition ---
class PositivesDataset(Dataset):
    """
    Dataset for loading positive task-translator pairs.
    Each item is a (task_features, translator_features, pair_label) tuple.
    pair_label is the original DataFrame index, identifying the positive pair.
    """
    def __init__(self, dataframe: pd.DataFrame):
        self.dataframe = dataframe
        self.pair_labels = self.dataframe.index.tolist()

        self.task_samples = []
        self.translator_samples = []

        self.task_features = [
            "SOURCE_LANG_task", "TARGET_LANG_task", "INDUSTRY_task",
            "FORECAST_task", "HOURLY_RATE_task", "task_categorical_vector",
        ]
        self.translator_features = [
            "SOURCE_LANG_EMBED_translator", "TARGET_LANG_EMBED_translator",
            "INDUSTRY_EMBED_translator", "FORECAST_mean", "HOURLY_RATE_mean",
            "HOURLY_RATE_translator", "QUALITY_EVALUATION_mean",
            "translator_categorical_vector",
        ]

        print("Processing dataset...")
        for i in tqdm(range(len(dataframe)), desc="Loading data"):
            row_data = self.dataframe.iloc[i]
            data_task = create_features(self.task_features, row_data)
            data_translator = create_features(self.translator_features, row_data)

            features_task = np.array(data_task, dtype=np.float32)
            features_translator = np.array(data_translator, dtype=np.float32)

            self.task_samples.append(features_task)
            self.translator_samples.append(features_translator)

        assert len(self.translator_samples) == len(self.task_samples), \
            "Mismatch in number of task and translator samples"

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        task = self.task_samples[idx]
        translator = self.translator_samples[idx]
        pair_label = self.pair_labels[idx]
        return torch.tensor(task), torch.tensor(translator), torch.tensor(pair_label, dtype=torch.long)

# --- Model Definitions (Autoencoders) ---
class Translator_AE(nn.Module):
    def __init__(self, translator_dim: int, latent_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(translator_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, translator_dim),
        )

    def forward(self, batch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        emb = self.encoder(batch)
        recon = self.decoder(emb)
        return emb, recon

class Task_AE(nn.Module):
    def __init__(self, task_dim: int, latent_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(task_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, task_dim),
        )

    def forward(self, batch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        emb = self.encoder(batch)
        recon = self.decoder(emb)
        return emb, recon

# --- Training and Evaluation Steps ---
def train_step(
    train_dataloader: DataLoader,
    model_task: nn.Module,
    model_translator: nn.Module,
    loss_func: Callable,
    miner_func: BaseMiner,
    optimizer_task: torch.optim.Optimizer,
    optimizer_translator: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model_task.train()
    model_translator.train()
    total_loss = 0.0
    for tasks, translators, labels in tqdm(train_dataloader, desc="Training"):
        tasks, translators, labels = tasks.to(device), translators.to(device), labels.to(device)
        optimizer_task.zero_grad()
        optimizer_translator.zero_grad()

        t_emb, _ = model_task(tasks)
        tr_emb, _ = model_translator(translators)
        embeds = torch.cat([t_emb, tr_emb], dim=0)
        lbls = torch.cat([labels, labels], dim=0)

        hard_triplets = miner_func(embeds, lbls)
        loss_val = loss_func(embeds, lbls, hard_triplets)
        loss_val.backward()
        optimizer_task.step()
        optimizer_translator.step()

        total_loss += loss_val.item()
    return total_loss / len(train_dataloader)

def calculate_accuracy_metrics(
    val_dataloader: DataLoader,
    model_task: nn.Module,
    model_translator: nn.Module,
    distance_metric: Callable = LpDistance(p=2),
    k_values: List[int] = [1, 5, 10],
    device: Optional[torch.device] = None
) -> Dict[str, float | Dict[int, float]]:
    model_task.eval()
    model_translator.eval()
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_task.to(device)
    model_translator.to(device)

    # Gather embeddings
    task_embs, tr_embs, labels = [], [], []
    with torch.no_grad():
        for tasks, translators, pair_labels in tqdm(val_dataloader, desc="Embedding Val Set"):
            t, tr = tasks.to(device), translators.to(device)
            te, _ = model_task(t)
            tre, _ = model_translator(tr)
            task_embs.append(te)
            tr_embs.append(tre)
            labels.append(pair_labels)
    task_embs = torch.cat(task_embs, dim=0)
    tr_embs = torch.cat(tr_embs, dim=0)
    labels = torch.cat(labels, dim=0)

    # Vectorized retrieval
    D = torch.cdist(task_embs.to(device), tr_embs.to(device))
    ranks = D.argsort(dim=1)
    # Compute MRR and P@k
    mrr = 0.0
    precision = {k: 0 for k in k_values}
    for i in range(ranks.size(0)):
        true_id = labels[i].item()
        ranked_labels = labels[ranks[i]]
        rank_pos = (ranked_labels == true_id).nonzero(as_tuple=True)[0][0].item() + 1
        mrr += 1.0 / rank_pos
        for k in k_values:
            if rank_pos <= k:
                precision[k] += 1
    mrr /= ranks.size(0)
    precision = {k: v / ranks.size(0) for k, v in precision.items()}
    return {"mrr": mrr, "precision_at_k": precision}

# --- Inference Function ---
def inference(
    model_tasks: nn.Module,
    model_translators: nn.Module,
    batch: tuple,
    device: Optional[torch.device] = None,
):
    model_tasks.eval()
    model_translators.eval()
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tasks, translators, _ = batch
    tasks, translators = tasks.to(device), translators.to(device)
    with torch.no_grad():
        te, _ = model_tasks(tasks)
        tre, _ = model_translators(translators)
    return tre.cpu(), te.cpu()

# --- Main Execution Block ---
if __name__ == "__main__":
    # Initialize wandb
    config = {
        "batch_size": 128,
        "learning_rate": 1e-4,
        "latent_dim": 10,
        "hidden_dim": 36,
        "num_epochs": 10,
        "margin": 0.2,
    }
    wandb.init(project="translation-retrieval", config=config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    wandb.config.update({"device": str(device)})

    # Load data
    positives_df = pd.read_pickle("../data/positives.pkl")
    if not positives_df.index.is_unique:
        positives_df = positives_df.reset_index(drop=True)

    dataset = PositivesDataset(positives_df)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=config["batch_size"], shuffle=True,
                              num_workers=2, pin_memory=device.type=="cuda")
    val_loader = DataLoader(val_set, batch_size=config["batch_size"], shuffle=False,
                            num_workers=2, pin_memory=device.type=="cuda")

    sample_task, sample_translator, _ = dataset[0]
    TASK_DIM = sample_task.shape[0]
    TRANSLATOR_DIM = sample_translator.shape[0]

    model_task = Task_AE(task_dim=TASK_DIM, latent_dim=config["latent_dim"], hidden_dim=config["hidden_dim"]).to(device)
    model_translator = Translator_AE(translator_dim=TRANSLATOR_DIM, latent_dim=config["latent_dim"], hidden_dim=config["hidden_dim"]).to(device)

    wandb.watch(model_task, log="all", log_freq=10)
    wandb.watch(model_translator, log="all", log_freq=10)

    miner = TripletMarginMiner(distance=LpDistance(p=2))
    loss_fn = TripletMarginLoss(distance=LpDistance(p=2), margin=config["margin"])
    optimizer_task = torch.optim.Adam(model_task.parameters(), lr=config["learning_rate"])
    optimizer_translator = torch.optim.Adam(model_translator.parameters(), lr=config["learning_rate"])

    # Training loop
    for epoch in range(config["num_epochs"]):
        train_loss = train_step(train_loader, model_task, model_translator,
                                loss_fn, miner, optimizer_task, optimizer_translator, device)
        metrics = calculate_accuracy_metrics(val_loader, model_task, model_translator,
                                             distance_metric=LpDistance(p=2), k_values=[1,3,5,10], device=device)

        # Log metrics
        wandb.log({
            "epoch": epoch+1,
            "train_loss": train_loss,
            "val_mrr": metrics["mrr"],
            **{f"precision@{k}": p for k,p in metrics["precision_at_k"].items()}
        })

        print(f"Epoch {epoch+1}/{config['num_epochs']} - train_loss: {train_loss:.4f} - val_mrr: {metrics['mrr']:.4f}")

    print("Training finished.")
    wandb.finish()