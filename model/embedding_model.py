import torch
import pandas as pd
import numpy as np

from pytorch_metric_learning.losses import TripletMarginLoss
from pytorch_metric_learning.miners import TripletMarginMiner, BaseMiner
from pytorch_metric_learning.distances import LpDistance

from torch.utils.data import Dataset, DataLoader
from torch import nn

from tqdm import tqdm

from typing import Optional, Callable


def create_features(features, data):
    return_data = []
    for col in features:
        col_data = data[col]
        if isinstance(col_data, (list, np.ndarray)):
            return_data.extend(col_data)
        elif isinstance(col_data, (int, float, np.number)):
            return_data.extend([col_data])

    return return_data


class PositivesDataset(Dataset):
    def __init__(self, dataframe):
        self.dataframe = dataframe

        self.pair_labels = self.dataframe.index.tolist()

        self.task_samples = []
        self.translator_samples = []

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

        for i in range(len(dataframe)):
            row_data = self.dataframe.iloc[i]

            data_task = create_features(self.task_features, row_data)
            data_translator = create_features(self.translator_features, row_data)

            features_task = np.array(data_task, dtype=np.float32)
            features_translator = np.array(data_translator, dtype=np.float32)

            self.task_samples.append(features_task)
            self.translator_samples.append(features_translator)

        assert len(self.translator_samples) == len(self.task_samples)

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        task = self.task_samples[idx]
        translator = self.translator_samples[idx]
        pair_label = self.pair_labels[idx]

        return torch.tensor(task), torch.tensor(translator), torch.tensor(pair_label)


class Translator_AE(nn.Module):
    def __init__(
        self,
        translator_dim: int,
        latent_dim: int,
        hidden_dim: int = 64,
    ):
        super(Translator_AE, self).__init__()

        self.translator_dim = translator_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim

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

    def forward(self, batch):
        translator_embeddings = self.encoder(batch)
        translator_recontrucctions = self.decoder(translator_embeddings)
        return translator_embeddings, translator_recontrucctions


class Task_AE(nn.Module):
    def __init__(
        self,
        task_dim: int,
        latent_dim: int,
        hidden_dim: int = 64,
    ):
        super(Task_AE, self).__init__()

        self.task_dim = task_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim

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

    def forward(self, batch):
        task_embeddings = self.encoder(batch)
        task_recontrucctions = self.decoder(task_embeddings)
        return task_embeddings, task_recontrucctions


def train_step(
    train_dataloader: torch.utils.data.DataLoader,
    loss_func,
    miner_func: BaseMiner,
    op_func_task: torch.optim.Optimizer,
    op_func_trans: torch.optim.Optimizer,
):
    num_batches = 0
    total_epoch_metric_loss = 0
    avg_epoch_metric_loss = None

    # all of these are in batches
    for translators, tasks, pair_labels in train_dataloader:
        optimizer_translator.zero_grad()
        optimizer_task.zero_grad()

        trans_embed, trans_output = model_translator.train()(translators)
        task_embed, task_output = model_task.train()(tasks)

        concat_embed = torch.cat([task_embed, trans_embed], dim=0)
        concat_labels = torch.cat([pair_labels, pair_labels], dim=0)

        hard_triplets = miner_func(concat_embed, concat_labels)

        metric_loss_value = loss_func(concat_embed, concat_labels, hard_triplets)
        metric_loss_value.backward()

        op_func_task.step()
        op_func_trans.step()

        total_epoch_metric_loss += metric_loss_value.item()
        num_batches += 1

        avg_epoch_metric_loss = (
            total_epoch_metric_loss / num_batches if num_batches > 0 else 0.0
        )

    return avg_epoch_metric_loss


def eval_step(dataloader: torch.utils.data.DataLoader, loss_func: Callable):
    num_batches = 0
    total_epoch_metric_loss = 0
    avg_epoch_metric_loss = None

    for translators, tasks, pair_labels in dataloader:
        trans_embed, trans_output = model_translator.eval()(translators)
        task_embed, task_output = model_task.eval()(tasks)

        concat_embed = torch.cat([task_embed, trans_embed], dim=0)
        concat_labels = torch.cat([pair_labels, pair_labels], dim=0)

        metric_loss_value = loss_func(concat_embed, concat_labels)

        total_epoch_metric_loss += metric_loss_value.item()
        num_batches += 1

        avg_epoch_metric_loss = (
            total_epoch_metric_loss / num_batches if num_batches > 0 else 0.0
        )

    return avg_epoch_metric_loss


def inference(
    model_tasks: nn.Module,
    model_translators: nn.Module,
    batch: tuple,
    device: Optional[torch.device] = None,
):
    assert len(batch) == 3

    if device is None:
        device = torch.device("cpu")  # defaulting to cpu

    translators, tasks, _ = batch

    trans_embed, _ = model_translators.eval()(translators)
    task_embed, _ = model_tasks.eval()(tasks)

    return trans_embed, task_embed


if __name__ == "__main__":
    positives = pd.read_pickle("data/positives.pkl")

    dataset = PositivesDataset(positives)
    train_set, val_set = torch.utils.data.random_split(dataset, [0.8, 0.2])

    BATCH_SIZE = 128

    train_dataloder = DataLoader(
        train_set,
        batch_size=BATCH_SIZE,
        shuffle=True,  # important
    )
    val_dataloder = DataLoader(
        val_set,
        batch_size=BATCH_SIZE,
        shuffle=True,  # important
    )

    model_task = Task_AE(
        task_dim=42,
        latent_dim=10,
        hidden_dim=36,
    )

    model_translator = Translator_AE(
        translator_dim=42,
        latent_dim=10,
        hidden_dim=36,
    )

    miner = TripletMarginMiner()
    loss = TripletMarginLoss()

    optimizer_translator = torch.optim.Adam(model_translator.parameters(), lr=0.0001)
    optimizer_task = torch.optim.Adam(model_task.parameters(), lr=0.0001)

    num_epochs = 10

    for epoch in range(num_epochs):
        train_loss = train_step(
            train_dataloder, loss, miner, optimizer_task, optimizer_translator
        )
        val_loss = eval_step(val_dataloder, loss)

        print(f"loss: {train_loss:.6f}, val_loss: {val_loss:.6f}")

    batch_test = val_set[0]

    translators_embedded, tasks_embedded = inference(
        batch=batch_test, model_tasks=model_task, model_translators=model_translator
    )
