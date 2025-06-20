import pickle
import re

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


def industry_tokenizer(in_language: str) -> str:
    result = re.sub(r"\&", "and", in_language)
    return result


class SimpleAutoencoder(nn.Module):
    def __init__(self, input_dim=384, intermediate_dim=64, latent_dim=10):
        super(SimpleAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, intermediate_dim),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(intermediate_dim, latent_dim),
            nn.Dropout(0.2),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, intermediate_dim),
            nn.ReLU(),
            nn.Linear(intermediate_dim, input_dim),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


class EmbeddingLookup:
    """
    Provides a lookup interface for retrieving pre-computed language embeddings.
    """

    def __init__(
        self,
        loaded_data: dict,
    ):
        """
        Initializes the EmbeddingLookup class.

        Args:
            lang_to_index (dict): Dictionary mapping language names to row indices.
            original_embeddings (np.ndarray): The original high-dimensional embeddings.
            latent_embeddings (np.ndarray): The latent low-dimensional embeddings.
        """

        lang_to_index = loaded_data.get("indu_to_index")
        original_embeddings = loaded_data.get("original_embeddings")
        latent_embeddings = loaded_data.get("latent_embeddings")

        if not isinstance(lang_to_index, dict):
            raise TypeError("lang_to_index must be a dictionary.")
        if (
            not isinstance(original_embeddings, np.ndarray)
            or original_embeddings.ndim != 2
        ):
            raise TypeError("original_embeddings must be a 2D NumPy array.")
        if not isinstance(latent_embeddings, np.ndarray) or latent_embeddings.ndim != 2:
            raise TypeError("latent_embeddings must be a 2D NumPy array.")
        if original_embeddings.shape[0] != latent_embeddings.shape[0]:
            raise ValueError(
                "Original and latent embeddings must have the same number of rows (samples)."
            )
        if len(lang_to_index) != original_embeddings.shape[0]:
            raise ValueError(
                "Size of lang_to_index mapping must match the number of embedding rows."
            )

        self.lang_to_index = lang_to_index
        self.original_embeddings = original_embeddings
        self.latent_embeddings = latent_embeddings
        print(f"EmbeddingLookup initialized with {len(lang_to_index)} languages.")
        print(f"  Original embedding dimension: {self.original_embeddings.shape[1]}")
        print(f"  Latent embedding dimension: {self.latent_embeddings.shape[1]}")

    def get_vector(
        self, language_name: str, embedding_type: str = "latent"
    ) -> np.ndarray | None:
        """
        Retrieves the pre-computed embedding vector for a given language name.
        """
        index = self.lang_to_index.get(language_name)

        if index is None:
            print(f"Warning: Language '{language_name}' not found in the lookup table.")
            return None

        if embedding_type == "latent":
            embedding_matrix = self.latent_embeddings
        elif embedding_type == "original":
            embedding_matrix = self.original_embeddings
        else:
            print(
                f"Warning: Unknown embedding_type '{embedding_type}'. Choose 'latent' or 'original'."
            )
            return None

        vector = embedding_matrix[index]
        return vector


def save_embedding_data(
    filename: str,
    industries: list,
    mapping: dict,
    original_emb: np.ndarray,
    latent_emb: np.ndarray,
):
    # Bundle data into a dictionary
    data_to_save = {
        "unique_industries": industries,
        "indu_to_index": mapping,
        "original_embeddings": original_emb,
        "latent_embeddings": latent_emb,
    }

    try:
        with open(filename, "wb") as f:  # 'wb' means write binary
            pickle.dump(data_to_save, f)
        print(f"Successfully saved embedding data to '{filename}'")
    except Exception as e:
        print(f"Error saving data to '{filename}': {e}")


def load_embedding_data(filename: str) -> dict | None:
    try:
        with open(filename, "rb") as f:  # 'rb' means read binary
            loaded_data = pickle.load(f)
        print(f"Successfully loaded embedding data from '{filename}'")
        # Optional: Basic validation
        if all(
            k in loaded_data
            for k in [
                "unique_industries",
                "indu_to_index",
                "original_embeddings",
                "latent_embeddings",
            ]
        ):
            return loaded_data
        else:
            print("Error: Loaded data is missing expected keys.")
            return None
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
        return None
    except Exception as e:
        print(f"Error loading data from '{filename}': {e}")
        return None


if __name__ == "__main__":
    df_tasks = pd.read_csv("../data/data_enhanced.csv", decimal=".")

    print("Loaded the CSV...")

    set_industries = set(df_tasks["MANUFACTURER_INDUSTRY"].unique())
    set_industries = set(industry_tokenizer(indu) for indu in set_industries)

    indu_to_index = {indu: i for i, indu in enumerate(set_industries)}

    print(f"Created the industries set, with a total of {len(set_industries)} items")

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    embed_industry = model.encode(list(set_industries), show_progress_bar=True)

    data_tensor = torch.Tensor(embed_industry)

    input_dim = 384
    latent_dim = 10
    learning_rate = 1e-3
    num_epochs = 200

    model = SimpleAutoencoder(input_dim=input_dim, latent_dim=latent_dim)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    losses = []

    print("Started model training...")
    model.train()
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        reconstructed = model(data_tensor)
        loss = criterion(reconstructed, data_tensor)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    model.eval()
    with torch.no_grad():
        encoded_data = model.encoder(data_tensor)

    print(f"Dim reduction from {input_dim} to {latent_dim}")

    encoded_data_cpu = encoded_data.to("cpu").numpy()

    print()
    print()

    output_filename = "industry_embeddings.pkl"

    save_embedding_data(
        filename=output_filename,
        industries=list(set_industries),
        mapping=indu_to_index,
        latent_emb=encoded_data_cpu,
        original_emb=embed_industry,
    )

    original_embeddings_np = embed_industry
    latent_embeddings_np = encoded_data_cpu

    input_filename = "industry_embeddings.pkl"
    loaded_data = load_embedding_data(input_filename)

    print()
    print()

    if loaded_data is not None:
        lookup = EmbeddingLookup(loaded_data)

        # Example usage
        es_us_vector = lookup.get_vector("Tobacco", embedding_type="latent")

        print("The vector for `Tobacco` is")
        print(es_us_vector)
