import os
import numpy as np
from extract_features import get_embedding

dataset_path = "dataset"

embeddings = []
labels = []
paths = []

for class_name in os.listdir(dataset_path):
    class_path = os.path.join(dataset_path, class_name)

    for img in os.listdir(class_path):
        img_path = os.path.join(class_path, img)

        emb = get_embedding(img_path)

        embeddings.append(emb[0])
        labels.append(class_name)
        paths.append(img_path)

np.savez("index.npz",
         embeddings=np.array(embeddings).astype("float32"),
         labels=np.array(labels),
         paths=np.array(paths))