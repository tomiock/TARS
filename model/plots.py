import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from tqdm import tqdm
# Required for type hinting if your imported classes use it.
from typing import List

try:
    from embedding_model import (
        create_features,
        PositivesDataset
        # Task_AE and Translator_AE are no longer needed for this script
    )
    print("Successfully imported 'create_features' and 'PositivesDataset' from training_utils.py")
except ImportError as e:
    print(f"Error importing from training_utils.py: {e}")
    print("Please ensure 'training_utils.py' exists and contains the required definitions:")
    print("- create_features (function)")
    print("- PositivesDataset (class)")
    exit()
# --- End Imports ---


def plot_tsne_on_features(features_np: np.ndarray, title: str, perplexity=30, n_iter=1000):
    """Generates and shows a t-SNE plot for the given features."""
    if features_np.ndim == 1: # Reshape if it's a single feature vector (e.g. for a single sample)
        features_np = features_np.reshape(1, -1)
        
    if features_np.shape[0] == 0:
        print(f"No features to plot for {title}.")
        return

    current_perplexity = perplexity
    # t-SNE constraint: perplexity must be less than the number of samples.
    if features_np.shape[0] <= current_perplexity:
        print(f"Warning: Number of samples ({features_np.shape[0]}) is less than or equal to perplexity ({current_perplexity}). Adjusting perplexity.")
        current_perplexity = max(1, features_np.shape[0] - 1) # Must be at least 1 if samples > 0

    if current_perplexity == 0 and features_np.shape[0] > 0: # Only 1 sample
        print(f"Only one sample for {title}. Cannot run t-SNE. Plotting first two dimensions if available.")
        plt.figure(figsize=(10, 8))
        x_val = features_np[0, 0] if features_np.shape[1] > 0 else 0
        y_val = features_np[0, 1] if features_np.shape[1] > 1 else 0
        plt.scatter([x_val], [y_val], alpha=0.7) # Plot as a single point
        plt.title(f"{title} (Single Sample - Original Features)")
        plt.xlabel("Feature 1 (or Value)")
        plt.ylabel("Feature 2 (or N/A)")
        plt.grid(True)
        plt.show()
        return
    elif features_np.shape[0] == 0 : # No samples
         print(f"No samples to plot for {title}.")
         return


    print(f"Running t-SNE for {title} with {features_np.shape[0]} samples and perplexity {current_perplexity}...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=current_perplexity, n_iter=n_iter, init='pca', learning_rate='auto')
    features_2d = tsne.fit_transform(features_np)

    plt.figure(figsize=(10, 8))
    plt.scatter(features_2d[:, 0], features_2d[:, 1], alpha=0.7, s=10) # Smaller points for dense plots
    plt.title(title)
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    # Configuration
    TSNE_PERPLEXITY = 30
    TSNE_ITERATIONS = 1000  # Increase for better quality, decrease for speed

    # 1. Load data
    try:
        positives_df = pd.read_pickle("../data/positives.pkl")
    except FileNotFoundError:
        print("Error: ../data/positives.pkl not found. Please check the path.")
        exit()

    if not positives_df.index.is_unique:
        positives_df = positives_df.reset_index(drop=True)

    print(f"Loaded {len(positives_df)} samples from positives.pkl")

    # 2. Define feature names (must match those used by create_features and your data)
    # These might ideally be imported or shared from training_utils.py if they are constants there.
    task_feature_names_list = [
        "SOURCE_LANG_task", "TARGET_LANG_task", "INDUSTRY_task",
        "FORECAST_task", "HOURLY_RATE_task", "task_categorical_vector",
    ]
    translator_feature_names_list = [
        "SOURCE_LANG_EMBED_translator", "TARGET_LANG_EMBED_translator",
        "INDUSTRY_EMBED_translator", "FORECAST_mean", "HOURLY_RATE_mean",
        "HOURLY_RATE_translator", "QUALITY_EVALUATION_mean", "translator_categorical_vector",
    ]

    # 3. Extract raw features using create_features
    all_raw_task_features = []
    all_raw_translator_features = []
    print("Extracting raw features for scaling...")
    for _, row in tqdm(positives_df.iterrows(), total=len(positives_df)):
        all_raw_task_features.append(create_features(task_feature_names_list, row))
        all_raw_translator_features.append(create_features(translator_feature_names_list, row))

    try:
        task_len_check = len(all_raw_task_features[0]) if all_raw_task_features else 0
        if not all(len(tf) == task_len_check for tf in all_raw_task_features):
             raise ValueError("Inconsistent task feature vector lengths. Check 'create_features' and data consistency.")
        np_all_raw_task_features = np.array(all_raw_task_features, dtype=np.float32) if all_raw_task_features else np.array([])

        translator_len_check = len(all_raw_translator_features[0]) if all_raw_translator_features else 0
        if not all(len(tf) == translator_len_check for tf in all_raw_translator_features):
            raise ValueError("Inconsistent translator feature vector lengths. Check 'create_features' and data consistency.")
        np_all_raw_translator_features = np.array(all_raw_translator_features, dtype=np.float32) if all_raw_translator_features else np.array([])
    except (ValueError, IndexError) as e: # Catch potential errors during array conversion
        print(f"Error during raw feature array preparation: {e}")
        print("This often occurs if 'create_features' produces variable length outputs or encounters unexpected data.")
        exit()


    # 4. Initialize and fit scalers
    task_scaler = StandardScaler()
    translator_scaler = StandardScaler()

    scaled_task_features_np = np.array([])
    scaled_translator_features_np = np.array([])

    if np_all_raw_task_features.size > 0:
        scaled_task_features_np = task_scaler.fit_transform(np_all_raw_task_features)
        print(f"Task features scaled. Shape: {scaled_task_features_np.shape}")
    else:
        print("No task features to scale.")

    if np_all_raw_translator_features.size > 0:
        scaled_translator_features_np = translator_scaler.fit_transform(np_all_raw_translator_features)
        print(f"Translator features scaled. Shape: {scaled_translator_features_np.shape}")
    else:
        print("No translator features to scale.")

    # 5. (Optional but good for consistency) Use PositivesDataset to hold scaled features
    # If your PositivesDataset in training_utils.py is designed to take already scaled data,
    # you might skip its instantiation here. But if it expects raw data + scalers to do its own thing,
    # then using it ensures you're plotting what it would have prepared.
    # The PositivesDataset provided in previous examples expects raw df + FITTED scalers.
    
    # For this simplified script, we've already scaled the features above (scaled_task_features_np).
    # We can directly use these for t-SNE.
    # If you want to strictly use PositivesDataset to get the samples:
    #   full_dataset = PositivesDataset(positives_df, task_scaler, translator_scaler) # scalers are already fitted
    #   task_samples_for_tsne = np.array(full_dataset.task_samples) # Assuming PositivesDataset stores them
    #   translator_samples_for_tsne = np.array(full_dataset.translator_samples)
    # For directness and simplicity, we'll use scaled_task_features_np and scaled_translator_features_np.

    # 6. Run t-SNE and plot the scaled input features
    if scaled_task_features_np.size > 0:
        plot_tsne_on_features(scaled_task_features_np, "t-SNE of Scaled Task Input Features",
                              perplexity=TSNE_PERPLEXITY, n_iter=TSNE_ITERATIONS)
    else:
        print("Skipping t-SNE for task features as no data is available.")

    if scaled_translator_features_np.size > 0:
        plot_tsne_on_features(scaled_translator_features_np, "t-SNE of Scaled Translator Input Features",
                                 perplexity=TSNE_PERPLEXITY, n_iter=TSNE_ITERATIONS)
    else:
        print("Skipping t-SNE for translator features as no data is available.")

    if not (scaled_task_features_np.size > 0 or scaled_translator_features_np.size > 0):
        print("No features were available to plot.")

    print("Done.")
