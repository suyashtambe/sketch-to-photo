import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

class SketchToImageDataset(Dataset):
    def __init__(self, sketch_dir, photo_dir, transform=None):
        """
        Initializes the dataset by loading paired sketch and photo image paths.

        Args:
            sketch_dir (str): Directory containing sketch images (.png).
            photo_dir (str): Directory containing photo images (.jpg).
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.sketch_dir = sketch_dir
        self.photo_dir = photo_dir
        self.transform = transform

        # List all .png files in sketch_dir and sort them
        sketch_files = sorted([f for f in os.listdir(sketch_dir) if f.endswith('.png')])

        # List all .jpg files in photo_dir and sort them
        photo_files = sorted([f for f in os.listdir(photo_dir) if f.endswith('.jpg')])

        # Extract filenames without extensions
        sketch_names = set(os.path.splitext(f)[0] for f in sketch_files)
        photo_names = set(os.path.splitext(f)[0] for f in photo_files)

        # Find common filenames in both folders (i.e., paired images)
        common_names = list(sketch_names & photo_names)

        # Create list of full path tuples for paired data
        self.data_pairs = [(os.path.join(sketch_dir, f"{name}.png"),
                            os.path.join(photo_dir, f"{name}.jpg")) for name in common_names]

    def __len__(self):
        """
        Returns the total number of paired images.
        """
        return len(self.data_pairs)

    def __getitem__(self, idx):
        """
        Loads and processes the sketch and photo image at the given index.

        Args:
            idx (int): Index of the item to retrieve.

        Returns:
            tuple: (sketch_tensor, photo_tensor)
        """
        sketch_path, photo_path = self.data_pairs[idx]

        # Load sketch as grayscale (1 channel)
        sketch = cv2.imread(sketch_path, cv2.IMREAD_GRAYSCALE)

        # Load photo as RGB (3 channels)
        photo = cv2.imread(photo_path, cv2.IMREAD_COLOR)

        # Resize both images to 128x128
        sketch = cv2.resize(sketch, (128, 128))
        photo = cv2.resize(photo, (128, 128))

        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to enhance edges in sketch
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        sketch = clahe.apply(sketch)

        # Apply Gaussian Blur to denoise the sketch image
        sketch = cv2.GaussianBlur(sketch, (3, 3), 0)

        # Normalize pixel values to [0, 1]
        sketch = sketch / 255.0
        photo = photo / 255.0

        # Convert sketch to 3 channels by duplicating the grayscale values
        sketch = np.expand_dims(sketch, axis=-1)  # shape becomes (128, 128, 1)
        sketch = np.repeat(sketch, 3, axis=-1)     # shape becomes (128, 128, 3)

        # Convert both sketch and photo to PyTorch tensors and rearrange dimensions to (C, H, W)
        sketch = torch.tensor(sketch, dtype=torch.float32).permute(2, 0, 1)
        photo = torch.tensor(photo, dtype=torch.float32).permute(2, 0, 1)

        return sketch, photo
