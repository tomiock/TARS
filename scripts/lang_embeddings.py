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


def get_top_n_neighbors(lang_index, similarity_matrix, language_list, n=3):
    """Finds the top N most similar languages for a given language index."""
    sim_scores = similarity_matrix[lang_index]

    sorted_indices = np.argsort(sim_scores)[::-1]

    neighbors = []
    count = 0
    for i in sorted_indices:
        if i == lang_index:
            continue

        neighbors.append((language_list[i], sim_scores[i]))
        count += 1
        if count >= n:
            break
    return neighbors


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

        lang_to_index = loaded_data.get("lang_to_index")
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
    languages: list,
    mapping: dict,
    original_emb: np.ndarray,
    latent_emb: np.ndarray,
):
    # Bundle data into a dictionary
    data_to_save = {
        "unique_languages": languages,
        "lang_to_index": mapping,
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
                "unique_languages",
                "lang_to_index",
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
    df_translators = pd.read_csv("../data/translators.csv", decimal=".")

    print("Loaded the CSVs...")

    unique_lang_source = df_translators["SOURCE_LANG"].unique()
    unique_lang_target = df_translators["TARGET_LANG"].unique()

    set_languages = set(unique_lang_target) | set(unique_lang_source)

    set_languages.remove("Spanish (SOURCE)")
    set_languages.remove("Portuguese (SOURCE)")

    set_languages = set(language_tokenizer(lang) for lang in set_languages)

    print(f"Created the language set, with a total of {len(set_languages)} items")

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    embed_lang = model.encode(list(set_languages), show_progress_bar=True)

    data_tensor = torch.Tensor(embed_lang)

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

    print("Dim reduction from 384 to 10")

    encoded_data_cpu = encoded_data.to("cpu").numpy()

    lang_to_index = {lang: i for i, lang in enumerate(set_languages)}

    similarity_original = cosine_similarity(embed_lang)
    similarity_latent = cosine_similarity(encoded_data_cpu)

    # --- Iterate and compare neighbors for each language ---

    print("\n--- Comparing Top 3 Neighbors ---")

    unique_languages = list(set_languages)

    for i, lang in enumerate(unique_languages):
        print(f"\n--- Language: {lang} ---")

        # Neighbors in Original Space (384-dim)
        neighbors_original = get_top_n_neighbors(
            i, similarity_original, unique_languages, n=3
        )
        print("  Original (384d) Neighbors:")
        if neighbors_original:
            for neighbor, score in neighbors_original:
                print(f"    - {neighbor} (Score: {score:.4f})")
        else:
            print("    - (No other languages to compare)")

        # Neighbors in Latent Space (10-dim)
        neighbors_latent = get_top_n_neighbors(
            i, similarity_latent, unique_languages, n=3
        )
        print("  Latent (10d) Neighbors:")
        if neighbors_latent:
            for neighbor, score in neighbors_latent:
                print(f"    - {neighbor} (Score: {score:.4f})")
        else:
            print("    - (No other languages to compare)")

    # --- Specific Check for English variants (if they exist in your list) ---
    print("\n--- Specific Checks ---")
    english_variants = [
        lang for lang in unique_languages if lang.startswith("English_")
    ]

    if len(english_variants) > 1:
        for lang in english_variants:
            if lang in lang_to_index:
                lang_idx = lang_to_index[lang]
                print(f"\n--- Neighbors for: {lang} ---")
                neighbors_orig = get_top_n_neighbors(
                    lang_idx, similarity_original, unique_languages, n=3
                )
                neighbors_lat = get_top_n_neighbors(
                    lang_idx, similarity_latent, unique_languages, n=3
                )
                print(f"  Original: {[f'{n} ({s:.3f})' for n, s in neighbors_orig]}")
                print(f"  Latent:   {[f'{n} ({s:.3f})' for n, s in neighbors_lat]}")
    else:
        print(
            "\nCould not perform specific check: Less than two 'English_' variants found in list."
        )

    # --- Specific Check for Chinese variants (if they exist in your list) ---
    chinese_variants = [
        lang for lang in unique_languages if lang.startswith("Chinese_")
    ]
    # Add other checks as needed (e.g., Spanish variants)

    if len(chinese_variants) > 1:
        for lang in chinese_variants:
            if lang in lang_to_index:
                lang_idx = lang_to_index[lang]
                print(f"\n--- Neighbors for: {lang} ---")
                neighbors_orig = get_top_n_neighbors(
                    lang_idx, similarity_original, unique_languages, n=3
                )
                neighbors_lat = get_top_n_neighbors(
                    lang_idx, similarity_latent, unique_languages, n=3
                )
                print(f"  Original: {[f'{n} ({s:.3f})' for n, s in neighbors_orig]}")
                print(f"  Latent:   {[f'{n} ({s:.3f})' for n, s in neighbors_lat]}")
    else:
        print(
            "\nCould not perform specific check: Less than two 'Chinese_' variants found in list."
        )

    spanish_variants = [
        lang for lang in unique_languages if lang.startswith("Spanish_")
    ]
    if len(spanish_variants) > 1:
        for lang in spanish_variants:
            if lang in lang_to_index:
                lang_idx = lang_to_index[lang]
                print(f"\n--- Neighbors for: {lang} ---")
                neighbors_orig = get_top_n_neighbors(
                    lang_idx, similarity_original, unique_languages, n=3
                )
                neighbors_lat = get_top_n_neighbors(
                    lang_idx, similarity_latent, unique_languages, n=3
                )
                print(f"  Original: {[f'{n} ({s:.3f})' for n, s in neighbors_orig]}")
                print(f"  Latent:   {[f'{n} ({s:.3f})' for n, s in neighbors_lat]}")

    print()
    print()

    output_filename = 'language_embeddings.pkl' 

    original_embeddings_np = embed_lang
    latent_embeddings_np = encoded_data_cpu

    save_embedding_data(
        output_filename,
        unique_languages, 
        lang_to_index, 
        original_embeddings_np, 
        latent_embeddings_np 
    )

    input_filename = 'language_embeddings.pkl'
    loaded_data = load_embedding_data(input_filename)

    print()
    print()

    if loaded_data is not None:
        lookup = EmbeddingLookup( lang_to_index=loaded_data['lang_to_index'],
            original_embeddings=loaded_data['original_embeddings'],
            latent_embeddings=loaded_data['latent_embeddings']
        )

        # Example usage
        es_us_vector = lookup.get_vector('Spanish_US', embedding_type='latent')

        print("The vector for `Spanish_US` is")
        print(es_us_vector)
