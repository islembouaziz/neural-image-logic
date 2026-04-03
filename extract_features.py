from clip_model import model, processor, device
from PIL import Image
import torch

def get_embedding(image_path):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device)

    with torch.no_grad():
        # Using model.get_image_features usually returns the projected tensor
        outputs = model.get_image_features(**inputs)

    # If outputs is still a container, this ensures we get the underlying tensor
    if hasattr(outputs, "last_hidden_state"):
        emb = outputs.last_hidden_state.mean(dim=1)
    else:
        emb = outputs # This is usually the tensor for get_image_features

    return emb.detach().cpu().numpy()