from search_engine import search
import matplotlib.pyplot as plt
import cv2

results = search("image32.png")

plt.figure(figsize=(12,6))

for i, (img_path, label) in enumerate(results):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    plt.subplot(1, len(results), i+1)
    plt.imshow(img)
    plt.title(label)
    plt.axis("off")

plt.show()