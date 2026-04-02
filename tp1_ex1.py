import torch
import open_clip
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"

# =========================
# 1. LOAD CLIP MODEL
# =========================
model, preprocess, tokenizer = open_clip.create_model_and_transforms(
    "ViT-B-32",
    pretrained="openai"
)

model = model.to(device)
model.eval()

# =========================
# 2. IMAGE ENCODING
# =========================
def encode_image(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = preprocess(Image.fromarray(img)).unsqueeze(0).to(device)

    with torch.no_grad():
        feat = model.encode_image(img)

    feat = feat / feat.norm(dim=-1, keepdim=True)
    return feat.cpu().numpy().flatten()

# =========================
# 3. LOAD DATABASE
# =========================
folder = "base_images/"

features_db = []
images_db = []
names_db = []

for f in os.listdir(folder):
    path = os.path.join(folder, f)
    img = cv2.imread(path)

    if img is not None:
        features_db.append(encode_image(img))
        images_db.append(img)
        names_db.append(f)

print("✅ Database loaded:", len(images_db), "images")

# =========================
# 4. COSINE SIMILARITY
# =========================
def cosine(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-7)

# =========================
# 5. SEARCH FUNCTION
# =========================
def search(img_query, k=5):
    q = encode_image(img_query)

    scores = [cosine(q, f) for f in features_db]
    idxs = np.argsort(scores)[::-1][:k]

    plt.figure(figsize=(15, 5))

    plt.subplot(1, k+1, 1)
    plt.title("Query")
    plt.imshow(cv2.cvtColor(img_query, cv2.COLOR_BGR2RGB))
    plt.axis("off")

    for i, idx in enumerate(idxs):
        plt.subplot(1, k+1, i+2)
        plt.title(f"{names_db[idx]}\n{scores[idx]:.2f}")
        plt.imshow(cv2.cvtColor(images_db[idx], cv2.COLOR_BGR2RGB))
        plt.axis("off")

    plt.show()

# =========================
# 6. TEST
# =========================
img = cv2.imread("dog1.jpg")

if img is None:
    print("❌ image not found")
else:
    search(img)