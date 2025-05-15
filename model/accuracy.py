import torch
import pandas as pd
import numpy as np

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
            return_data.append(float(col_data)) # Ensure float
        # Add handling for other types if necessary, or raise error
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
        # Using DataFrame index as the unique identifier for each positive pair
        self.pair_labels = self.dataframe.index.tolist()

        self.task_samples = []
        self.translator_samples = []

        # Define feature columns for tasks and translators
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

            # Ensure all features are converted to float32 numpy arrays
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
        # pair_label is the original index, crucial for identifying true positives
        pair_label = self.pair_labels[idx] 

        return torch.tensor(task), torch.tensor(translator), torch.tensor(pair_label, dtype=torch.long)


# --- Model Definitions (Autoencoders) ---
class Translator_AE(nn.Module):
    def __init__(self, translator_dim: int, latent_dim: int, hidden_dim: int = 64):
        super(Translator_AE, self).__init__()
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
        translator_embeddings = self.encoder(batch)
        translator_reconstructions = self.decoder(translator_embeddings)
        return translator_embeddings, translator_reconstructions


class Task_AE(nn.Module):
    def __init__(self, task_dim: int, latent_dim: int, hidden_dim: int = 64):
        super(Task_AE, self).__init__()
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
        task_embeddings = self.encoder(batch)
        task_reconstructions = self.decoder(task_embeddings)
        return task_embeddings, task_reconstructions


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
    """Performs a single training epoch."""
    model_task.train()
    model_translator.train()
    total_epoch_metric_loss = 0
    num_batches = 0

    for tasks, translators, pair_labels in tqdm(train_dataloader, desc="Training"):
        tasks, translators, pair_labels = \
            tasks.to(device), translators.to(device), pair_labels.to(device)

        optimizer_task.zero_grad()
        optimizer_translator.zero_grad()

        task_embed, _ = model_task(tasks)
        trans_embed, _ = model_translator(translators)

        # Concatenate embeddings and labels for triplet mining
        # Both tasks and translators can serve as anchors relative to each other
        concat_embed = torch.cat([task_embed, trans_embed], dim=0)
        # Labels need to be structured so miner knows task_embed[i] is positive to trans_embed[i]
        # For TripletMarginMiner, labels indicate identity.
        # Here, pair_labels are unique IDs for each pair.
        # We duplicate them because we concatenated task and translator embeddings from the same pairs.
        concat_labels = torch.cat([pair_labels, pair_labels], dim=0)


        hard_triplets = miner_func(concat_embed, concat_labels)
        metric_loss_value = loss_func(concat_embed, concat_labels, hard_triplets)
        
        metric_loss_value.backward()
        optimizer_task.step()
        optimizer_translator.step()

        total_epoch_metric_loss += metric_loss_value.item()
        num_batches += 1

    avg_epoch_metric_loss = total_epoch_metric_loss / num_batches if num_batches > 0 else 0.0
    return avg_epoch_metric_loss


def eval_step(
    val_dataloader: DataLoader,
    model_task: nn.Module,
    model_translator: nn.Module,
    loss_func: Callable, # Note: Miner is not used in this eval_loss calculation
    device: torch.device,
) -> float:
    """Performs evaluation for loss calculation on the validation set."""
    model_task.eval()
    model_translator.eval()
    total_epoch_metric_loss = 0
    num_batches = 0

    with torch.no_grad():
        for tasks, translators, pair_labels in tqdm(val_dataloader, desc="Calculating Val Loss"):
            tasks, translators, pair_labels = \
                tasks.to(device), translators.to(device), pair_labels.to(device)

            task_embed, _ = model_task(tasks)
            trans_embed, _ = model_translator(translators)
            
            concat_embed = torch.cat([task_embed, trans_embed], dim=0)
            concat_labels = torch.cat([pair_labels, pair_labels], dim=0)
            
            # For eval loss, we don't mine, just compute loss on all pairs/triplets implicitly formed
            # The loss function itself might handle this (e.g. if it can take embeddings and labels directly)
            # TripletMarginLoss expects triplets. If no miner, it might not work as intended or might try to form all possible triplets.
            # For a simpler validation loss, one might use a different strategy or ensure the loss function can handle this.
            # However, to be consistent with how pytorch-metric-learning examples often do it,
            # we can pass None for triplets if the loss supports it, or compute pairwise distances and a simpler loss.
            # For now, let's assume loss_func can handle it or we'd need a simpler val loss.
            # A common practice is to compute the loss over all possible pairs/triplets if a miner is not used.
            # For TripletMarginLoss, if triplets are not provided, it will raise an error.
            # So, for a simple loss value, we might just report 0 or skip this if we only care about retrieval metrics.
            # OR, we can use a miner here too, but it's less common for simple val loss.
            # For now, let's calculate it similarly to training but without backprop.
            # This requires a miner. If no miner, then this loss is not well-defined for TripletLoss.
            # Let's assume we want to see how the loss behaves on val set with all potential triplets.
            # A simple way: just use the loss function with embeddings and labels, some can compute loss over all pairs.
            # Given the original code used loss_func directly, it implies it might work or it was an oversight.
            # Let's stick to the original structure for eval_loss, assuming loss_func can handle it.
            # If TripletMarginLoss is used, it *needs* triplets.
            # A quick fix is to not calculate this specific loss if it's problematic without a miner.
            # For now, let's keep it for consistency but acknowledge this might need adjustment
            # depending on the exact loss_func behavior without explicit triplets.
            # A safe bet for TripletMarginLoss is to not call it without triplets or use a miner.
            # Let's assume the user wants a loss value, so we'll pass None for triplets and see if the library handles it
            # by forming all possible ones (can be computationally expensive).
            # On second thought, TripletMarginLoss(embeddings, labels) without triplets is not standard.
            # We should probably remove this or use a different metric for simple validation loss.
            # For now, returning 0.0 as a placeholder if a proper val loss without miner isn't set up.
            # A better approach for val_loss would be to use a different loss (e.g. reconstruction loss if AEs are used for that)
            # or to use the miner as well.
            # Given the focus is on retrieval metrics, let's simplify this eval_step's loss.
            # We'll calculate an average loss using the miner, similar to training, but without backprop.
            
            # Re-instating miner for val_loss calculation for consistency:
            temp_miner = TripletMarginMiner() # Or pass the train miner if its state doesn't matter
            hard_triplets = temp_miner(concat_embed, concat_labels)
            if hard_triplets[0].numel() > 0: # Check if any triplets were mined
                 metric_loss_value = loss_func(concat_embed, concat_labels, hard_triplets)
                 total_epoch_metric_loss += metric_loss_value.item()
                 num_batches += 1


    avg_epoch_metric_loss = total_epoch_metric_loss / num_batches if num_batches > 0 else 0.0
    return avg_epoch_metric_loss

# --- Accuracy Metrics Calculation ---
def calculate_accuracy_metrics(
    val_dataloader: DataLoader,
    model_task: nn.Module,
    model_translator: nn.Module,
    distance_metric: Callable = LpDistance(p=2), # Euclidean distance
    k_values: List[int] = [1, 5, 10],
    device: Optional[torch.device] = None
) -> Dict[str, float | Dict[int, float]]:
    """
    Calculates retrieval metrics (MRR, Precision@k, Recall@k) on the validation set.
    Args:
        val_dataloader: DataLoader for the validation set.
        model_task: The trained task embedding model.
        model_translator: The trained translator embedding model.
        distance_metric: The distance function (e.g., LpDistance, CosineSimilarity).
        k_values: A list of integers for k in Precision@k and Recall@k.
        device: The device to run computations on.
    Returns:
        A dictionary containing MRR, Precision@k, and Recall@k.
    """
    model_task.eval()
    model_translator.eval()

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_task.to(device)
    model_translator.to(device)

    all_task_embeddings_list = []
    all_translator_embeddings_list = []
    all_original_pair_labels_list = [] # These are the original DataFrame indices

    print("\nGathering embeddings for accuracy calculation...")
    with torch.no_grad():
        for tasks_batch, translators_batch, pair_labels_batch in tqdm(val_dataloader, desc="Embedding Val Set"):
            tasks_batch = tasks_batch.to(device)
            translators_batch = translators_batch.to(device)

            task_embeds_batch, _ = model_task(tasks_batch)
            translator_embeds_batch, _ = model_translator(translators_batch)

            all_task_embeddings_list.append(task_embeds_batch.cpu())
            all_translator_embeddings_list.append(translator_embeds_batch.cpu())
            all_original_pair_labels_list.append(pair_labels_batch.cpu())

    all_task_embeddings = torch.cat(all_task_embeddings_list, dim=0)
    all_translator_embeddings = torch.cat(all_translator_embeddings_list, dim=0)
    # These labels correspond to the original positive pairs.
    # all_task_embeddings[i] was originally paired with all_translator_embeddings[i],
    # and this pair has the ID all_original_pair_labels[i].
    all_original_pair_labels = torch.cat(all_original_pair_labels_list, dim=0)

    num_samples = all_task_embeddings.size(0)
    if num_samples == 0:
        return {"mrr": 0.0, "precision_at_k": {k: 0.0 for k in k_values}, "recall_at_k": {k: 0.0 for k in k_values}}

    precision_at_k = {k: 0 for k in k_values}
    recall_at_k = {k: 0 for k in k_values} # Will be same as precision if one true positive per task
    reciprocal_ranks = []

    print("Calculating accuracy metrics...")
    for i in tqdm(range(num_samples), desc="Calculating Metrics"):
        task_anchor_embedding = all_task_embeddings[i].unsqueeze(0) # Shape: (1, embed_dim)
        # This is the label of the true positive translator for the current task_anchor_embedding
        true_positive_translator_original_label = all_original_pair_labels[i]

        # Calculate distances from the anchor task to ALL translator embeddings in the validation set.
        # `all_translator_embeddings` contains embeddings of all translators from the val set.
        # Each of these translators was originally paired with some task.
        distances = distance_metric(task_anchor_embedding, all_translator_embeddings) # Shape: (1, num_samples)
        distances = distances.squeeze(0) # Shape: (num_samples)

        # Get the indices that would sort the distances in ascending order (closest first)
        sorted_indices = torch.argsort(distances)
        
        # Get the original_pair_labels of the translators in the ranked order
        ranked_translator_original_labels = all_original_pair_labels[sorted_indices]

        # Find rank of the true positive translator for the current task
        true_positive_rank = -1
        for rank, current_label in enumerate(ranked_translator_original_labels):
            if current_label.item() == true_positive_translator_original_label.item():
                true_positive_rank = rank + 1 # 1-based rank
                break
        
        if true_positive_rank != -1:
            reciprocal_ranks.append(1.0 / true_positive_rank)
            for k_val in k_values:
                if true_positive_rank <= k_val:
                    precision_at_k[k_val] += 1
                    recall_at_k[k_val] += 1 # As discussed, for 1-to-1 positive pairs from input
        else:
            # This should ideally not happen if the true positive is part of all_translator_embeddings
            # and all_original_pair_labels correctly map.
            reciprocal_ranks.append(0.0)


    mrr = np.mean(reciprocal_ranks) if reciprocal_ranks else 0.0
    final_precision_at_k = {k: (count / num_samples) for k, count in precision_at_k.items()}
    final_recall_at_k = {k: (count / num_samples) for k, count in recall_at_k.items()}
        
    return {"mrr": mrr, "precision_at_k": final_precision_at_k, "recall_at_k": final_recall_at_k}


# --- Inference Function ---
def inference(
    model_tasks: nn.Module,
    model_translators: nn.Module,
    batch: tuple, # Expected (tasks_tensor, translators_tensor, pair_labels_tensor)
    device: Optional[torch.device] = None,
):
    """Performs inference to get embeddings for a batch."""
    model_tasks.eval()
    model_translators.eval()

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_tasks.to(device)
    model_translators.to(device)
    
    tasks, translators, _ = batch # Unpack the batch
    tasks = tasks.to(device)
    translators = translators.to(device)

    with torch.no_grad():
        task_embed, _ = model_tasks(tasks)
        trans_embed, _ = model_translators(translators)

    return trans_embed.cpu(), task_embed.cpu()


# --- Main Execution Block ---
if __name__ == "__main__":
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data (ensure 'positives.pkl' is in 'data/' directory or adjust path)
    try:
        positives_df = pd.read_pickle("../data/positives.pkl")
    except FileNotFoundError:
        print("Error: 'data/positives.pkl' not found. Please ensure the path is correct.")
        exit()
    
    # Ensure DataFrame index is unique if not already, as it's used for pair_labels
    if not positives_df.index.is_unique:
        print("Warning: DataFrame index is not unique. Resetting index.")
        positives_df = positives_df.reset_index(drop=True)

    dataset = PositivesDataset(positives_df)
    
    # Split dataset (ensure reproducible splits if needed by setting torch.manual_seed)
    # torch.manual_seed(42) # For reproducible splits
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])

    BATCH_SIZE = 128 # Consider making this configurable

    train_dataloader = DataLoader(
        train_set,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2, # Adjust based on your system
        pin_memory=True if device.type == 'cuda' else False
    )
    # For validation/test, shuffle is often False for consistent evaluation order,
    # but for aggregated metrics over the whole set, it doesn't strictly matter.
    val_dataloader = DataLoader(
        val_set,
        batch_size=BATCH_SIZE,
        shuffle=False, 
        num_workers=2,
        pin_memory=True if device.type == 'cuda' else False
    )

    # Define model dimensions (these should match your feature vector sizes)
    # Infer dimensions from the first sample if possible, or ensure they are correct
    sample_task_features, sample_translator_features, _ = dataset[0]
    TASK_DIM = sample_task_features.shape[0]
    TRANSLATOR_DIM = sample_translator_features.shape[0]
    LATENT_DIM = 10  # Example, tune this
    HIDDEN_DIM = 36 # Example, tune this
    print(f"Task feature dimension: {TASK_DIM}, Translator feature dimension: {TRANSLATOR_DIM}")


    model_task = Task_AE(task_dim=TASK_DIM, latent_dim=LATENT_DIM, hidden_dim=HIDDEN_DIM).to(device)
    model_translator = Translator_AE(translator_dim=TRANSLATOR_DIM, latent_dim=LATENT_DIM, hidden_dim=HIDDEN_DIM).to(device)

    # Miner and Loss function
    # Consider trying other miners and distance functions as well
    miner = TripletMarginMiner(distance=LpDistance(p=2)) 
    loss = TripletMarginLoss(distance=LpDistance(p=2), margin=0.2) # margin is tunable

    # Optimizers
    LEARNING_RATE = 0.0001 # Tunable
    optimizer_translator = torch.optim.Adam(model_translator.parameters(), lr=LEARNING_RATE)
    optimizer_task = torch.optim.Adam(model_task.parameters(), lr=LEARNING_RATE)

    num_epochs = 3 # Tunable

    print("\nStarting training...")
    for epoch in range(num_epochs):
        print(f"\n--- Epoch {epoch+1}/{num_epochs} ---")
        train_loss = train_step(
            train_dataloader, model_task, model_translator, loss, miner,
            optimizer_task, optimizer_translator, device
        )
        
        # Calculate validation loss (optional, can be slow if using miner)
        # val_loss = eval_step(val_dataloader, model_task, model_translator, loss, device)
        # print(f"Epoch {epoch+1} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
        print(f"Epoch {epoch+1} - Train Loss: {train_loss:.6f}")


        # Calculate accuracy metrics on the validation set
        accuracy_results = calculate_accuracy_metrics(
            val_dataloader, 
            model_task, 
            model_translator, 
            distance_metric=LpDistance(p=2), # Ensure this matches miner/loss distance
            k_values=[1, 3, 5, 10],
            device=device
        )
        print(f"Validation MRR: {accuracy_results['mrr']:.4f}")
        for k, p_at_k in accuracy_results['precision_at_k'].items():
            print(f"  Precision@{k}: {p_at_k:.4f}", end="")
        print() # Newline
        # Recall will be the same as precision in this specific setup
        # for k, r_at_k in accuracy_results['recall_at_k'].items():
        #     print(f"Validation Recall@{k}: {r_at_k:.4f}")


    print("\nTraining finished.")

    # Example of running inference on a sample from the validation set
    if len(val_set) > 0:
        # Get a single batch from val_dataloader for inference example
        try:
            sample_val_batch = next(iter(val_dataloader))
            # Ensure the batch is a tuple of 3 tensors
            if isinstance(sample_val_batch, list) and len(sample_val_batch) == 3:
                 # Reconstruct as a tuple for the inference function
                sample_val_batch_tuple = (sample_val_batch[0], sample_val_batch[1], sample_val_batch[2])
                
                print("\nRunning sample inference...")
                # Note: inference function expects a batch, not a single item from val_set directly
                # So we pass the first batch from the val_dataloader
                inferred_translator_embeds, inferred_task_embeds = inference(
                    model_tasks=model_task, 
                    model_translators=model_translator,
                    batch=sample_val_batch_tuple, 
                    device=device
                )
                print(f"Shape of inferred translator embeddings (first batch): {inferred_translator_embeds.shape}")
                print(f"Shape of inferred task embeddings (first batch): {inferred_task_embeds.shape}")
                assert inferred_translator_embeds.shape[1] == LATENT_DIM
                assert inferred_task_embeds.shape[1] == LATENT_DIM
            else:
                print("Could not get a sample batch in the expected format for inference example.")

        except StopIteration:
            print("Validation set is empty, cannot run sample inference.")
    else:
        print("Validation set is empty, cannot run sample inference.")

    print("\nScript finished.")

