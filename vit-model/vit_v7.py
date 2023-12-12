# Program: ViT v7 
# Author: Jack Friedman
# Date: 12/11/2023
# Purpose: Implements Vision Transformer with pretrained weights from 2020 model

## Import libraries
import os
import io
import gc
import time 
import pickle
import gzip
import matplotlib.pyplot as plt
from datetime import timedelta
import numpy as np
import pandas as pd
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping
import torch
from torch import Tensor, nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
from torchmetrics.functional import accuracy, f1_score
from torchvision.models.vision_transformer import EncoderBlock
from sklearn.model_selection import train_test_split
from vision_transformer import VisionTransformer
import warnings
warnings.filterwarnings('ignore')
import sys
sys.path.append('../preprocessing')
from Preprocessing_v6 import *
from DataLoader import load_data, load_data_tubevit

# Setting seed for reproducibility
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)

## Step 0: Define key constants and hyperparameters
# DATA
BATCH_SIZE = 32
INPUT_SHAPE = (10, 54, 120)  # (C, H, W)
TEST_SIZE = 0.2
VAL_SIZE = 0.25  # % of the training data size (0.25 * 0.8 = 0.2 in this case)

# OPTIMIZER
LEARNING_RATE = 1e-5

# TUBELET EMBEDDING
PATCH_SIZE = (6, 6)
NUM_PATCHES = (INPUT_SHAPE[2] // PATCH_SIZE[0]) ** 2

# ViT ARCHITECTURE
LAYER_NORM_EPS = 1e-6
PROJECTION_DIM = 256
NUM_HEADS = 8
NUM_LAYERS = 12
DROP_RATE = 0.2

## Step 1: Load and Process Data
print("LOADING DATA")
### Step 1A: Load 2024 data
# Load data
[games_df, players_df, plays_df, tracking_df, df_2020] = load_data_tubevit()

### Step 1B: Process 2024 data
print("PREPROCESSING DATA")

# Preprocess data
tracking_df_clean = preprocess_all_df(plays_df, games_df, players_df, tracking_df)

### Step 1C: Get tensors
print("GETTING TENSORS")
play_ids, tensor_list, labels = prepare_4d_tensors(tracking_df_clean, min_frames = 12, tensor_type = 'torch')

# Keep only the handoff frame (final frame)
print("GETTING HANDOFF FRAMES")
def get_final_frames(tensor_list):
    final_frames = []
    for tensor in tensor_list:
        frame = tensor[tensor.shape[0] - 1, :, :, :]
        final_frames += [frame]
    return final_frames
tensor_list = get_final_frames(tensor_list)
## Step 2: Prep data for training (model-specific preprocessing)
### Step 2A: Get min and max yard indices

# STEP 0: Round to nearest yard and adjust by 99 (because of the 2020 specifications)
indexed_labels = [round(label) + 99 for label in labels]
min_idx_y = np.min(indexed_labels)
max_idx_y = np.max(indexed_labels)
print('min yardIndex:', min_idx_y)
print('max yardIndex:', max_idx_y)

# STEP 1: CALCULATE NUMBER OF CLASSES (YARDS)
num_classes_y = max_idx_y - min_idx_y + 1
print('num classes:', num_classes_y)


print("GETTING DATA LOADERS")
### Step 2B: Train-test split
X_train, X_test, y_train, y_test = train_test_split(tensor_list, labels, test_size=TEST_SIZE, random_state=SEED)

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=VAL_SIZE, random_state=SEED) # 0.25 x 0.8 = 0.2
### Step 2C: Preprocess and build dataloaders
# Build data loaders


class ImageDataset(Dataset):
    def __init__(self, images, labels, num_classes_y, min_idx_y):
        self.images = images
        self.labels = labels
        self.num_classes_y = num_classes_y
        self.min_idx_y = min_idx_y

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Convert the image to float32
        image = self.images[idx].float()

        # Permute the tensor dimensions from (W, H, C) to (C, H, W)
        image = image.permute(2, 1, 0)

        # Preprocess the label
        label_indexed = int(round(self.labels[idx].item())) + 99
        label_one_hot = torch.zeros(self.num_classes_y, dtype=torch.float32)
        label_one_hot[label_indexed - self.min_idx_y] = 1.0

        return image, label_one_hot
    
def prepare_dataloader(videos, labels, num_classes_y, min_idx_y, loader_type='train', batch_size=32):
    dataset = ImageDataset(videos, labels, num_classes_y, min_idx_y)
    shuffle = loader_type == 'train'
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader


trainloader = prepare_dataloader(X_train, y_train, num_classes_y, min_idx_y, 'train', BATCH_SIZE)
validloader = prepare_dataloader(X_val, y_val, num_classes_y, min_idx_y, 'valid', BATCH_SIZE)
testloader = prepare_dataloader(X_test, y_test, num_classes_y, min_idx_y, 'test', BATCH_SIZE)

### Step 2D: Define model architecture

# Loss function - Continuous Ranked Probability Score
def crps_loss(y_true, y_pred):
    loss = torch.mean(torch.sum((torch.cumsum(y_pred, dim=1) - torch.cumsum(y_true, dim=1)) ** 2, dim=1)) / 199
    return loss

# Model architecture
class ViTLightningModule(pl.LightningModule):
    def __init__(self, learning_rate=LEARNING_RATE):
        super().__init__()
        self.model = VisionTransformer(img_size = (INPUT_SHAPE[1], INPUT_SHAPE[2]),
            patch_size =PATCH_SIZE,
            in_chans = INPUT_SHAPE[0],
            num_classes = num_classes_y,
            global_pool = 'avg',
            embed_dim = PROJECTION_DIM,
            depth = NUM_LAYERS,
            num_heads = NUM_HEADS,
            drop_rate = DROP_RATE,
            # drop_path_rate = 0.3,
            pos_drop_rate = DROP_RATE,
            patch_drop_rate = DROP_RATE,
            proj_drop_rate = DROP_RATE,
            attn_drop_rate = DROP_RATE) 
        self.learning_rate = learning_rate

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        probabilities = self(x)
        loss = crps_loss(y, probabilities)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        probabilities = self(x)
        loss = crps_loss(y, probabilities)
        self.log('val_loss', loss)  # Logging validation loss
        return loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        probabilities = self(x)
        loss = crps_loss(y, probabilities)
        self.log('test_loss', loss)  # Logging test loss
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
    


## Step 3: Train and Evaluate Model
# PRINT HYPERPARAMETERS
print("THIS IS ViT v7")
print("HYPERPARAMETERS:")
params = list(globals().items())
for name, value in params:
    if name.isupper():  # Assuming hyperparameters are in uppercase
        print(f"{name} = {value}")
model = ViTLightningModule(LEARNING_RATE)
model.to('cpu')
print("MODEL ARCHITECTURE:", model)

# Initialize weights
pretrained_dict = torch.load("vit_2020_v0_state_dict.pth")
model_dict = model.state_dict()

# 1. filter out unnecessary keys
pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
# 2. overwrite entries in the existing state dict
model_dict.update(pretrained_dict) 
# 3. load the new state dict
model.load_state_dict(model_dict)

# Set up early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=10)

# Create the trainer with early stopping
trainer = pl.Trainer(callbacks=[early_stopping], enable_progress_bar=False)

# Train the model with the DataLoaders
print("training model...")
trainer.fit(model, trainloader, validloader)
print("finished trainining")

# Get train loss
print("\nGET TRAINING LOSS:")
trainer.test(model, trainloader)
print("^^^^ THIS ABOVE NUMBER IS THE TRAIN SET LOSS\n")

print("\nGET VALIDATION SET LOSS:")
trainer.validate(model, validloader)
print("^^^^ THIS ABOVE NUMBER IS THE VALIDATION SET LOSS\n")

# Evaluate the model on the test set
print("\n GET TEST SET LOSS:")
trainer.test(model, testloader)
print("^^^^ THIS ABOVE NUMBER IS THE TEST SET LOSS\n")
