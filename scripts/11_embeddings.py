import pickle
import re

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


def language_tokenizer(in_language: str) -> str:
    """
    Normalizes language strings.
    """
    lang = in_language.strip()
    regex = r"(\w+)\s*\(([\w\s]+)\)"

    def replace_func(match):
        word1 = match.group(1)
        word2_raw = match.group(2).strip()
        word2_processed = re.sub(r"\s+", "_", word2_raw)
        return f"{word1}_{word2_processed}"

    result = re.sub(regex, replace_func, lang)
    result = re.sub(r"[\s_]+", "_", result)
    result = result.strip("_")
    return result


def industry_tokenizer(in_industry: str) -> str:
    """
    Normalizes industry strings by replacing '&' with 'and'.
    """
    return re.sub(r"&", "and", in_industry).strip()


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
    Provides a lookup interface for retrieving pre-computed embeddings.
    """

    def __init__(self, loaded_data: dict, index_key: str = None):
        # Determine mapping key automatically if not provided
        if index_key:
            to_index = loaded_data.get(index_key)
        else:
            # try common keys
            to_index = loaded_data.get('lang_to_index') or loaded_data.get('indu_to_index')

        original_embeddings = loaded_data.get('original_embeddings')
        latent_embeddings = loaded_data.get('latent_embeddings')

        if not isinstance(to_index, dict):
            raise TypeError("to_index must be a dictionary mapping items to indices.")
        if not isinstance(original_embeddings, np.ndarray) or original_embeddings.ndim != 2:
            raise TypeError("original_embeddings must be a 2D NumPy array.")
        if not isinstance(latent_embeddings, np.ndarray) or latent_embeddings.ndim != 2:
            raise TypeError("latent_embeddings must be a 2D NumPy array.")
        if original_embeddings.shape[0] != latent_embeddings.shape[0]:
            raise ValueError(
                "Original and latent embeddings must have the same number of rows."
            )
        if len(to_index) != original_embeddings.shape[0]:
            raise ValueError(
                "Size of mapping must match the number of embedding rows."
            )

        self.to_index = to_index
        self.original_embeddings = original_embeddings
        self.latent_embeddings = latent_embeddings
        print(f"EmbeddingLookup initialized with {len(to_index)} items.")
        print(f"  Original dim: {self.original_embeddings.shape[1]}")
        print(f"  Latent dim: {self.latent_embeddings.shape[1]}")

    def get_vector(self, name: str, embedding_type: str = 'latent') -> np.ndarray | None:
        idx = self.to_index.get(name)
        if idx is None:
            print(f"Warning: '{name}' not found in lookup.")
            return None
        matrix = (
            self.latent_embeddings
            if embedding_type == 'latent'
            else self.original_embeddings
            if embedding_type == 'original'
            else None
        )
        if matrix is None:
            print("Warning: Unknown embedding_type. Use 'latent' or 'original'.")
            return None
        return matrix[idx]


def save_embedding_data(
    filename: str,
    items: list,
    mapping: dict,
    original_emb: np.ndarray,
    latent_emb: np.ndarray,
    mapping_key: str = 'to_index',
    unique_key: str = 'unique_items'
):
    data = {
        unique_key: items,
        mapping_key: mapping,
        'original_embeddings': original_emb,
        'latent_embeddings': latent_emb,
    }
    try:
        with open(filename, 'wb') as f:
            pickle.dump(data, f)
        print(f"Saved embedding data to '{filename}'")
    except Exception as e:
        print(f"Error saving data: {e}")


def load_embedding_data(filename: str) -> dict | None:
    try:
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        print(f"Loaded embedding data from '{filename}'")
        if 'original_embeddings' in data and 'latent_embeddings' in data:
            return data
        print("Error: Missing expected keys in loaded data.")
        return None
    except FileNotFoundError:
        print(f"File '{filename}' not found.")
        return None
    except Exception as e:
        print(f"Error loading data: {e}")
        return None


# Example usage in __main__ omitted for brevity. Combine your data loading, tokenizing, encoding, training, and saving steps as needed.
