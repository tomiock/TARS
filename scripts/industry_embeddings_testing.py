import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from industry_embeddings import EmbeddingLookup


# Load the data
with open("scripts/industry_embeddings.pkl", "rb") as f:
    loaded_data = pickle.load(f)

# Create the lookup
lookup = EmbeddingLookup(loaded_data)


"""     To see all industries in set:

lookup = EmbeddingLookup(loaded_data)

print("Available industries:")
for industry in sorted(lookup.lang_to_index.keys()):
    print(industry)

"""

def compare_industry(industry, industry_2):

    # Get the vectors
    vec1 = lookup.get_vector(industry, embedding_type="latent")
    vec2 = lookup.get_vector(industry_2, embedding_type="latent")

    # Compute cosine similarity
    if vec1 is not None and vec2 is not None:
        similarity = cosine_similarity(vec1.reshape(1, -1), vec2.reshape(1, -1))[0][0]
        print(f"Cosine similarity between '{industry}' and '{industry_2}': {similarity:.4f}")
    else:
        print("One or both industries not found.")

# Pick industries
industry1 = "Automobile Manufacturers"
industry2 = "Automobiles"
industry3 = "Health Care Facilities"

# Testing
compare_industry(industry1, industry2)
compare_industry(industry1, industry3)
compare_industry(industry2, industry3)



