import cv2
import numpy as np
from PIL import Image
import torch
from torchvision import transforms

def preprocess_sketch(pil_img):
    img_np = np.array(pil_img)
    clahe = cv2.createCLAHE(clipLimit=2.0)
    img_np = clahe.apply(img_np)
    img_np = cv2.GaussianBlur(img_np, (3, 3), 0)
    img_resized = cv2.resize(img_np, (256, 256))
    tensor = torch.from_numpy(img_resized).float().div(255).unsqueeze(0)  # (1, H, W)
    return tensor
