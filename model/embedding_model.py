import torch
import pandas as pd
import numpy as np

from pytorch_metric_learning.losses import TripletMarginLoss
from pytorch_metric_learning.miners import TripletMarginMiner
from pytorch_metric_learning.distances import LpDistance

from torch.utils.data import Dataset, DataLoader
from torch import nn

from tqdm import tqdm

from typing import Optional


class PositivesDataset(Dataset):
    def __init__(self, dataframe):
        self.dataframe = dataframe

        self.pair_labels = self.dataframe.index.tolist()
        self.samples = []

        for i in range(len(dataframe)):
            self.samples.append((i, self.pair_labels[i], 0))
            self.samples.append((i, self.pair_labels[i], 1))

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

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        original_row_idx, label, entity_type = self.samples[idx]
        row_data = self.dataframe.iloc[original_row_idx]

        if entity_type == 0:
            feature_list = []
            for col in self.task_features:
                col_data = row_data[col]
                if isinstance(col_data, (list, np.ndarray)):
                    feature_list.extend(col_data)
                elif isinstance(col_data, (int, float, np.number)):
                    feature_list.append(col_data)

            features = np.array(feature_list, dtype=np.float32)

        elif entity_type == 1:
            feature_list = []
            for col in self.translator_features:
                col_data = row_data[col]
                if isinstance(col_data, (list, np.ndarray)):
                    feature_list.extend(col_data)
                elif isinstance(col_data, (int, float, np.number)):
                    feature_list.append(col_data)
            features = np.array(feature_list, dtype=np.float32)

        else:
            raise ValueError("entity_type must be 0 or 1")

        features_tensor = torch.from_numpy(features)
        label_tensor = torch.tensor(label, dtype=torch.float32)
        entity_type_tensor = torch.tensor(entity_type, dtype=torch.float32)

        return features_tensor, label_tensor, entity_type_tensor


def collate_fn(batch):
    return tuple(zip(*batch))


class Autoencoder(nn.Module):
    def __init__(
        self,
        task_dim: int,
        translator_dim: int,
        latent_dim: int,
        hidden_dim: int = 64,
    ):
        super(Autoencoder, self).__init__()

        self.task_dim = task_dim
        self.translator_dim = translator_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim

        self.encoder_task = nn.Sequential(
            nn.Linear(task_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )

        self.encoder_translator = nn.Sequential(
            nn.Linear(translator_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )

        self.decoder_task = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, task_dim),
        )

        self.decoder_translator = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, translator_dim),
        )

    def forward(self, batch, entity):
        if entity == 0:
            task_embeddings = self.encoder_task(batch)
            task_recontrucctions = self.decoder_task(task_embeddings)
            return task_embeddings, task_recontrucctions

        if entity == 1:
            translator_embeddings = self.encoder_translator(batch)
            translator_recontrucctions = self.decoder_translator(translator_embeddings)
            return translator_embeddings, translator_recontrucctions


def train_step(train_dataloader, loss_func, miner_func):
    num_batches = 0
    total_epoch_metric_loss = 0
    avg_epoch_metric_loss = None
    for features, labels, entity_types in train_dataloader:
        all_embeddings = []
        all_labels = []
        all_reconstruccions = []

        for i in range(len(features)):
            single_features = features[i]
            single_label = labels[i]
            single_entity = entity_types[i]

            if single_entity.item() == 0:  # this is a task
                embeddings, reconstruction = model(single_features, single_entity)

            elif single_entity.item() == 1:  # this is a tranlator
                embeddings, reconstruction = model(single_features, single_entity)

            else:
                raise ValueError

            all_embeddings.append(embeddings)
            all_labels.append(single_label)
            all_reconstruccions.append(reconstruction)

        all_embeddings = torch.stack(all_embeddings)
        all_labels = torch.stack(all_labels)
        all_reconstruccions = torch.stack(all_reconstruccions)

        hard_triplets = miner_func(all_embeddings, all_labels)

        metric_loss_value = torch.tensor(0.0)

        if len(hard_triplets) > 0:
            metric_loss_value = loss_func(all_embeddings, all_labels, hard_triplets)
        else:
            print("ERRORRR")

        metric_loss_value.backward()
        optimizer.step()

        total_epoch_metric_loss += metric_loss_value.item()
        num_batches += 1

        avg_epoch_metric_loss = (
            total_epoch_metric_loss / num_batches if num_batches > 0 else 0.0
        )

    return avg_epoch_metric_loss


def eval_step(dataloader, loss_func):
    num_batches = 0
    total_epoch_metric_loss = 0
    avg_epoch_metric_loss = None
    for features, labels, entity_types in dataloader:
        all_embeddings = []
        all_labels = []
        all_reconstruccions = []

        for i in range(len(features)):
            single_features = features[i]
            single_label = labels[i]
            single_entity = entity_types[i]

            if single_entity.item() == 0:  # this is a task
                embeddings, reconstruction = model(single_features, single_entity)

            elif single_entity.item() == 1:  # this is a tranlator
                embeddings, reconstruction = model(single_features, single_entity)

            else:
                raise ValueError

            all_embeddings.append(embeddings)
            all_labels.append(single_label)
            all_reconstruccions.append(reconstruction)

        all_embeddings = torch.stack(all_embeddings)
        all_labels = torch.stack(all_labels)

        metric_loss_value = loss_func(all_embeddings, all_labels)

        total_epoch_metric_loss += metric_loss_value.item()
        num_batches += 1

        avg_epoch_metric_loss = (
            total_epoch_metric_loss / num_batches if num_batches > 0 else 0.0
        )

    return avg_epoch_metric_loss


def inference(model: nn.Module, batch: tuple, device: Optional[torch.device] = None):
    model.eval()

    if device is None:
        device = torch.device("cpu")  # defaulting to cpu

    features_tuple, labels_tuple, entity_tuples = batch

    embeddings_info = []
    with torch.no_grad():
        for i in range(len(features_tuple)):
            single_features = features_tuple.to(device)
            single_label = labels_tuple.item()
            single_entity = entity_tuples.item()

            print(single_features.shape)

            if single_entity == 0:
                embedding = model.encoder_task(single_features.unsqueeze(0))
            elif single_entity == 1:
                embedding = model.decoder_task(single_features.unsqueeze(0))
            else:
                raise ValueError("invalid entity type")

            single_embedding = embedding.squeeze(0)
            embeddings_info.append((single_embedding, single_label, single_entity))

    return embeddings_info


if __name__ == "__main__":
    positives = pd.read_pickle("data/positives.pkl")

    dataset = PositivesDataset(positives)
    train_set, val_set = torch.utils.data.random_split(dataset, [0.8, 0.2])

    train_dataloder = DataLoader(
        train_set, batch_size=32, shuffle=True, collate_fn=collate_fn
    )
    val_dataloder = DataLoader(
        val_set, batch_size=32, shuffle=True, collate_fn=collate_fn
    )

    model = Autoencoder(
        task_dim=42,
        translator_dim=42,
        latent_dim=10,
        hidden_dim=36,
    )

    miner = TripletMarginMiner()
    loss = TripletMarginLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    num_epochs = 1

    for epoch in range(num_epochs):
        train_loss = train_step(train_dataloder, loss, miner)
        val_loss = eval_step(val_dataloder, loss)

        print(f"loss: {train_loss:.6f}, val_loss: {val_loss:.6f}")

    batch_test = val_set[0]
