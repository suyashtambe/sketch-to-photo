import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import matplotlib.pyplot as plt
from multiprocessing import Process

# Define constants
IMG_SIZE = 256  # Resizing all images to 256x256
TRAIN_SPLIT = 0.8  # 80% training, 20% testing
OUTPUT_TRAIN_DIR = "processed_dataset_1/train"
OUTPUT_TEST_DIR = "processed_dataset_1/test"
SKETCH_DIR = r"C:\Users\Suyash Tambe\Desktop\sketch-photo\256x256\sketch\tx_000000000000"  
PHOTO_DIR = r"C:\Users\Suyash Tambe\Desktop\sketch-photo\256x256\photo\tx_000000000000"  

# Ensure output directories exist
os.makedirs(OUTPUT_TRAIN_DIR, exist_ok=True)
os.makedirs(OUTPUT_TEST_DIR, exist_ok=True)

# Set device for PyTorch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SketchToImageDataset(Dataset):
    def __init__(self, sketch_dir, photo_dir, transform=None):
        self.sketch_dir = sketch_dir
        self.photo_dir = photo_dir
        self.transform = transform
        self.categories = os.listdir(photo_dir)  # Ensure categories match
        self.data_pairs = []
     
        for category in self.categories:
            sketch_category_path = os.path.join(sketch_dir, category)
            photo_category_path = os.path.join(photo_dir, category)
            
            if not os.path.exists(sketch_category_path):
                continue  # Skip categories without corresponding sketches

            photos = sorted(os.listdir(photo_category_path))
            for photo in photos:
                sketch_files = sorted(os.listdir(sketch_category_path))
                for sketch in sketch_files:
                    sketch_path = os.path.join(sketch_category_path, sketch)
                    photo_path = os.path.join(photo_category_path, photo)
                    self.data_pairs.append((sketch_path, photo_path))

    def __len__(self):
        return len(self.data_pairs)

    def __getitem__(self, idx):
        sketch_path, photo_path = self.data_pairs[idx]
        sketch = cv2.imread(sketch_path, cv2.IMREAD_GRAYSCALE)  # Load sketch in grayscale
        photo = cv2.imread(photo_path, cv2.IMREAD_COLOR)  # Load photo in RGB
        
        # Resize images to 256x256
        sketch = cv2.resize(sketch, (IMG_SIZE, IMG_SIZE))
        photo = cv2.resize(photo, (IMG_SIZE, IMG_SIZE))
        
        # Enhance sketch using CLAHE (adaptive histogram equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        sketch = clahe.apply(sketch)
        
        # Apply Gaussian blur to remove noise
        sketch = cv2.GaussianBlur(sketch, (3,3), 0)
        
        # Normalize pixel values to [0,1]
        sketch = sketch / 255.0
        photo = photo / 255.0
        
        # Convert sketch to 3-channel
        sketch = np.expand_dims(sketch, axis=-1)
        sketch = np.repeat(sketch, 3, axis=-1)  
        
        # Convert to tensors and move to device
        sketch = torch.tensor(sketch, dtype=torch.float32).permute(2, 0, 1).to(device)
        photo = torch.tensor(photo, dtype=torch.float32).permute(2, 0, 1).to(device)
        
        return sketch, photo

# Initialize dataset
dataset = SketchToImageDataset(SKETCH_DIR, PHOTO_DIR)

# Split dataset into training and testing
train_size = int(TRAIN_SPLIT * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# Save images in parallel
def save_images(dataset, output_dir):
    os.makedirs(os.path.join(output_dir, "sketch"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "photo"), exist_ok=True)
    
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False)
    for i, (sketch, photo) in enumerate(data_loader):
        sketch = sketch.squeeze(0).cpu().numpy()
        photo = photo.squeeze(0).cpu().numpy()
        sketch_path = os.path.join(output_dir, "sketch", f"{i}.png")
        photo_path = os.path.join(output_dir, "photo", f"{i}.jpg")
        
        cv2.imwrite(sketch_path, (sketch * 255).astype(np.uint8).transpose(1, 2, 0))
        cv2.imwrite(photo_path, (photo * 255).astype(np.uint8).transpose(1, 2, 0))

if __name__ == "__main__":
    # Run saving processes in parallel
    train_process = Process(target=save_images, args=(train_dataset, OUTPUT_TRAIN_DIR))
    test_process = Process(target=save_images, args=(test_dataset, OUTPUT_TEST_DIR))
    train_process.start()
    test_process.start()
    train_process.join()
    test_process.join()

    print(f"Processed dataset saved in folders: {OUTPUT_TRAIN_DIR} and {OUTPUT_TEST_DIR}")

    # Display sample images
    sample_sketch, sample_photo = dataset[0]
    plt.figure(figsize=(8,4))
    plt.subplot(1,2,1)
    plt.imshow(sample_sketch.cpu().numpy().transpose(1, 2, 0), cmap='gray')
    plt.title("Sketch")
    plt.subplot(1,2,2)
    plt.imshow(cv2.cvtColor((sample_photo.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8), cv2.COLOR_BGR2RGB))
    plt.title("Photo")
    plt.show()
