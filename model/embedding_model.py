import torch
import pandas as pd
import numpy as np
import wandb

from pytorch_metric_learning.losses import TripletMarginLoss
from pytorch_metric_learning.miners import TripletMarginMiner, BaseMiner
from pytorch_metric_learning.distances import LpDistance, CosineSimilarity

from torch.utils.data import Dataset, DataLoader
from torch import nn

from tqdm import tqdm  # For progress bars

from typing import Optional, Callable, Dict, List


def create_features(features: List[str], data: pd.Series) -> List[float]:
    return_data = []
    for col in features:
        col_data = data[col]
        if isinstance(col_data, (list, np.ndarray)):
            return_data.extend(col_data)
        elif isinstance(col_data, (int, float, np.number)):
            return_data.append(float(col_data))
    return return_data


class PositivesDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame):
        self.dataframe = dataframe
        self.pair_labels = dataframe.index.tolist()
        self.task_samples, self.translator_samples = [], []
        self.task_features = [
            "SOURCE_LANG_task",
            "TARGET_LANG_task",
            "INDUSTRY_task",
            "FORECAST_task",
            "HOURLY_RATE_task",
            "task_categorical_vector",
        ]
        self.translator_features = [
            "SOURCE_LANG_EMBED_translator",
            "TARGET_LANG_EMBED_translator",
            "INDUSTRY_EMBED_translator",
            "FORECAST_mean",
            "HOURLY_RATE_mean",
            "HOURLY_RATE_translator",
            "QUALITY_EVALUATION_mean",
            "translator_categorical_vector",
        ]

        for i in tqdm(range(len(dataframe)), desc="Loading data"):
            row_data = dataframe.iloc[i]
            t = create_features(self.task_features, row_data)
            tr = create_features(self.translator_features, row_data)
            self.task_samples.append(np.array(t, dtype=np.float32))
            self.translator_samples.append(np.array(tr, dtype=np.float32))

        assert len(self.task_samples) == len(self.translator_samples), (
            "Mismatch between task and translator samples"
        )

    def __len__(self) -> int:
        return len(self.task_samples)

    def __getitem__(self, idx: int):
        return (
            torch.tensor(self.task_samples[idx]),
            torch.tensor(self.translator_samples[idx]),
            torch.tensor(self.pair_labels[idx], dtype=torch.long),
        )


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

    def forward(self, batch: torch.Tensor):
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

    def forward(self, batch: torch.Tensor):
        emb = self.encoder(batch)
        recon = self.decoder(emb)
        return emb, recon


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

        tasks, translators, labels = (
            tasks.to(device),
            translators.to(device),
            labels.to(device),
        )

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


def eval_step(
    val_dataloader: DataLoader,
    model_task: nn.Module,
    model_translator: nn.Module,
    loss_func: Callable,
    miner_func: BaseMiner,
    device: torch.device,
) -> float:

    model_task.eval()
    model_translator.eval()

    total_loss = 0.0
    count = 0

    with torch.no_grad():
        for tasks, translators, labels in tqdm(val_dataloader, desc="Validation Loss"):
            tasks, translators, labels = (
                tasks.to(device),
                translators.to(device),
                labels.to(device),
            )

            t_emb, _ = model_task(tasks)
            tr_emb, _ = model_translator(translators)

            embeds = torch.cat([t_emb, tr_emb], dim=0)
            lbls = torch.cat([labels, labels], dim=0)

            hard_triplets = miner_func(embeds, lbls)

            if hard_triplets[0].numel() > 0:
                loss_val = loss_func(embeds, lbls, hard_triplets)
                total_loss += loss_val.item()
                count += 1

    return total_loss / count if count > 0 else 0.0


def calculate_accuracy_metrics(
    val_dataloader: DataLoader,
    model_task: nn.Module,
    model_translator: nn.Module,
    distance_metric: Callable = LpDistance(p=2),
    k_values: List[int] = [1, 5, 10],
    device: Optional[torch.device] = None,
) -> Dict[str, float | Dict[int, float]]:

    model_task.eval()
    model_translator.eval()

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_task.to(device)
    model_translator.to(device)
    task_embs, tr_embs, labels = [], [], []

    with torch.no_grad():
        for tasks, translators, pair_labels in tqdm(
            val_dataloader, desc="Embedding Val Set"
        ):

            t, tr = tasks.to(device), translators.to(device)
            te, _ = model_task(t)
            tre, _ = model_translator(tr)

            task_embs.append(te.cpu())
            tr_embs.append(tre.cpu())
            labels.append(pair_labels)

    task_embs = torch.cat(task_embs)
    tr_embs = torch.cat(tr_embs)
    labels = torch.cat(labels)

    num = task_embs.size(0)
    D = torch.cdist(task_embs.to(device), tr_embs.to(device))
    ranks = D.argsort(dim=1)

    precision_at_k = {k: 0 for k in k_values}
    reciprocal_ranks = []
    for i in range(num):
        true = labels[i].item()
        ranked = labels[ranks[i]]

        pos = (ranked == true).nonzero(as_tuple=True)[0][0].item() + 1

        reciprocal_ranks.append(1.0 / pos)
        for k in k_values:
            if pos <= k:
                precision_at_k[k] += 1

    mrr = float(np.mean(reciprocal_ranks))
    precision_at_k = {k: v / num for k, v in precision_at_k.items()}
    return {"mrr": mrr, "precision_at_k": precision_at_k}


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
        t_emb, _ = model_tasks(tasks)
        tr_emb, _ = model_translators(translators)

    return tr_emb.cpu(), t_emb.cpu()


if __name__ == "__main__":
    wandb.init(
        project="translation-retrieval-sweep",
        config={
            "batch_size": 128,
            "learning_rate": 1e-4,
            "latent_dim": 10,
            "hidden_dim": 36,
            "margin": 0.2,
            "epochs": 3,
        },
    )

    cfg = wandb.config
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    wandb.config.update({"device": str(device)})

    # Log data artifact
    data_art = wandb.Artifact("positives-data", type="dataset")
    data_art.add_file("../data/positives.pkl")
    wandb.log_artifact(data_art)

    positives_df = pd.read_pickle("../data/positives.pkl")
    if not positives_df.index.is_unique:
        positives_df = positives_df.reset_index(drop=True)
    dataset = PositivesDataset(positives_df)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_set,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_set,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=(device.type == "cuda"),
    )

    sample_task, sample_translator, _ = dataset[0]
    TASK_DIM = sample_task.shape[0]
    TRANSLATOR_DIM = sample_translator.shape[0]
    LATENT_DIM = cfg.latent_dim
    HIDDEN_DIM = cfg.hidden_dim

    model_task = Task_AE(
        task_dim=TASK_DIM, latent_dim=LATENT_DIM, hidden_dim=HIDDEN_DIM
    ).to(device)
    model_translator = Translator_AE(
        translator_dim=TRANSLATOR_DIM, latent_dim=LATENT_DIM, hidden_dim=HIDDEN_DIM
    ).to(device)

    wandb.watch(model_task, log="all", log_freq=10)
    wandb.watch(model_translator, log="all", log_freq=10)

    miner = TripletMarginMiner(distance=LpDistance(p=2))
    loss_fn = TripletMarginLoss(distance=LpDistance(p=2), margin=cfg.margin)

    optimizer_task = torch.optim.Adam(
        model_task.parameters(), lr=cfg.learning_rate
    )
    optimizer_translator = torch.optim.Adam(
        model_translator.parameters(), lr=cfg.learning_rate
    )

    for epoch in range(cfg.epochs):
        train_loss = train_step(
            train_loader,
            model_task,
            model_translator,
            loss_fn,
            miner,
            optimizer_task,
            optimizer_translator,
            device,
        )

        val_loss = eval_step(
            val_loader, model_task, model_translator, loss_fn, miner, device
        )

        metrics = calculate_accuracy_metrics(
            val_loader,
            model_task,
            model_translator,
            distance_metric=LpDistance(p=2),
            k_values=[1, 3, 5, 10],
            device=device,
        )

        wandb.log(
            {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_mrr": metrics["mrr"],
                **{f"precision@{k}": p for k, p in metrics["precision_at_k"].items()},
            }
        )

        print(
            f"Epoch {epoch + 1}/{cfg.epochs} - train_loss: {train_loss:.4f} - val_loss: {val_loss:.4f} - mrr: {metrics['mrr']:.4f}"
        )

    torch.save(model_task.state_dict(), "task_ae.pth")
    torch.save(model_translator.state_dict(), "trans_ae.pth")
    model_art = wandb.Artifact("ae-models", type="model")
    model_art.add_file("task_ae.pth")
    model_art.add_file("trans_ae.pth")
    wandb.log_artifact(model_art)

    print("Training finished.")
    wandb.finish()
