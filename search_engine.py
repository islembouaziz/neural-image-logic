import numpy as np
import faiss
from extract_features import get_embedding

data = np.load("index.npz", allow_pickle=True)

embeddings = data["embeddings"]
labels = data["labels"]
paths = data["paths"]

index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

def search(image_path, k=5):
    query = get_embedding(image_path).astype("float32")

    distances, indices = index.search(query, k)

    results = []
    for i in indices[0]:
        results.append((paths[i], labels[i]))

    return results