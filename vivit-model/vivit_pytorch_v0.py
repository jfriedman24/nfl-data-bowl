# Author: Jack Friedman 
# Date 11/21/2024 
# Adapted from: Aritra Roy Gosthipaty and Ayush Thakur (https://github.com/keras-team/keras-io/blob/master/examples/vision/vivit.py) <br>
# Original Paper: ViViT: A Video Vision Transformer (https://arxiv.org/abs/2103.15691) by Arnab et al. <br>

## Import libraries
import os
import io
import pickle
import gzip
import time 
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
import warnings
warnings.filterwarnings('ignore')
import sys
sys.path.append('../preprocessing')
from Preprocessing_v6 import *
from DataLoader import load_data, load_data_tubevit
## Preprocessing
# Setting seed for reproducibility
SEED = 42
## Step 0: Define key hyperparameters and constants
# DATA
BATCH_SIZE = 8
FRAMES_PER_PLAY = 12
INPUT_SHAPE = (FRAMES_PER_PLAY, 120, 54, 10)

# OPTIMIZER
LEARNING_RATE = 1e-5


# TUBELET EMBEDDING
PATCH_SIZE = (6, 6, 6)
NUM_PATCHES = (INPUT_SHAPE[0] // PATCH_SIZE[0]) * (INPUT_SHAPE[1] // PATCH_SIZE[1]) * (INPUT_SHAPE[2] // PATCH_SIZE[2])
# NUM_PATCHES = (INPUT_SHAPE[0] // PATCH_SIZE[0]) ** 2

# ViViT ARCHITECTURE
LAYER_NORM_EPS = 1e-6
EMBED_DIM = 256
NUM_HEADS = 8
NUM_LAYERS = 12
DROP_RATE = 0.2


## Step 1: Load and preprocess data
# Load data
[games_df, players_df, plays_df, tracking_df] = load_data()
# Preprocess data
tracking_df_clean = preprocess_all_df(plays_df, games_df, players_df, tracking_df)

# Get tensors
play_ids, tensor_list, labels = prepare_4d_tensors(tracking_df_clean, min_frames = 12, tensor_type = 'torch')


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
### Step 2B: Train-test split
X_train, X_test, y_train, y_test = train_test_split(tensor_list, labels, test_size=0.2, random_state=SEED)

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=SEED) # 0.25 x 0.8 = 0.2


# Build data loaders
class VideoDataset(Dataset):
    def __init__(self, videos, labels, num_classes_y, min_idx_y):
        self.videos = videos
        self.labels = labels
        self.num_classes_y = num_classes_y
        self.min_idx_y = min_idx_y

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):
        # Convert the frames to float32
        frames = self.videos[idx].float()

        # Permute the tensor dimensions from (T, W, H, C) to (C, T, H, W)
        frames = frames.permute(3, 0, 2, 1)

        # frames = frames.unsqueeze(-1)  # Adding new axis

        # Preprocess the label
        label_indexed = int(round(self.labels[idx].item())) + 99
        label_one_hot = torch.zeros(self.num_classes_y, dtype=torch.float32)
        label_one_hot[label_indexed - self.min_idx_y] = 1.0

        return frames, label_one_hot
    
def prepare_dataloader(videos, labels, num_classes_y, min_idx_y, loader_type='train', batch_size=32):
    dataset = VideoDataset(videos, labels, num_classes_y, min_idx_y)
    shuffle = (loader_type == 'train')
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader


trainloader = prepare_dataloader(X_train, y_train, num_classes_y, min_idx_y, 'train', BATCH_SIZE)
validloader = prepare_dataloader(X_val, y_val, num_classes_y, min_idx_y, 'valid', BATCH_SIZE)
testloader = prepare_dataloader(X_test, y_test, num_classes_y, min_idx_y, 'test', BATCH_SIZE)

class TubeletEmbedding(nn.Module):
    def __init__(self, embed_dim, patch_size):
        super().__init__()
        self.projection = nn.Conv3d(in_channels=10, out_channels=embed_dim, 
                                    kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.projection(x)  # Output shape: [batch_size, embed_dim, D, H, W]
        
        # Flatten D, H, W dimensions into a single dimension
        batch_size, embed_dim, D, H, W = x.shape
        x = x.view(batch_size, embed_dim, -1)  # New shape: [batch_size, embed_dim, D*H*W]

        # Transpose to get [batch_size, D*H*W, embed_dim] which is [batch_size, seq_length, embed_dim]
        x = x.transpose(1, 2)

        return x
    
class PositionalEncoder(nn.Module):
    def __init__(self, embed_dim, num_tokens):
        super(PositionalEncoder, self).__init__()
        self.embed_dim = embed_dim
        self.position_embedding = nn.Embedding(num_tokens, embed_dim)
        self.positions = torch.arange(0, num_tokens).unsqueeze(1)

    def forward(self, encoded_tokens):
        # encoded_tokens is expected to have shape [batch_size, num_tokens, embed_dim]
        batch_size, seq_length, _ = encoded_tokens.shape
        
        # Expand position indices to match batch size and apply embeddings
        device = encoded_tokens.device  # Get the device of the input
        positions = self.positions.expand(seq_length, batch_size).transpose(0, 1).to(device)
        encoded_positions = self.position_embedding(positions)

        # Add positional encodings
        encoded_tokens = encoded_tokens + encoded_positions
        return encoded_tokens


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout_prob):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim, eps = 1e-6)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout_prob)
        self.norm2 = nn.LayerNorm(embed_dim, eps = 1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.GELU(),
            nn.Linear(4 * embed_dim, embed_dim), 
        )

    def forward(self, x):
        x2 = self.norm1(x)
        x = x + self.attn(x2, x2, x2)[0]
        x2 = self.norm2(x)
        x = x + self.mlp(x2)
        return x

# Loss function - Continuous Ranked Probability Score
def crps_loss(y_true, y_pred):
    loss = torch.mean(torch.sum((torch.cumsum(y_pred, dim=1) - torch.cumsum(y_true, dim=1)) ** 2, dim=1)) / 199
    return loss

class ViViTClassifier(pl.LightningModule):
    def __init__(self, tubelet_embedder, positional_encoder, num_layers, num_heads, embed_dim, num_classes, learning_rate, pdrop):
        super().__init__()
        self.tubelet_embedder = tubelet_embedder
        self.positional_encoder = positional_encoder

        # Transformer layers
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, pdrop) for _ in range(num_layers)
        ])

        self.layer_norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(embed_dim, num_classes)

        self.learning_rate = learning_rate

        # Store hyperparameters
        self.save_hyperparameters()

    def forward(self, x):
        x = self.tubelet_embedder(x)
        x = self.positional_encoder(x)

        for block in self.transformer_blocks:
            x = block(x)

        x = self.layer_norm(x)
        x = x.mean(dim=1)
        x = self.classifier(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        x, y = x.to(self.device), y.to(self.device)  # Move data to the same device as the model
        y_pred = self.forward(x)
        y_pred = self.forward(x)
        loss = crps_loss(y, y_pred)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x, y = x.to(self.device), y.to(self.device)  # Move data to the same device as the model
        y_pred = self.forward(x)
        loss = crps_loss(y, y_pred)
        self.log('val_loss', loss)

    def test_step(self, batch, batch_idx):
        x, y = batch
        x, y = x.to(self.device), y.to(self.device)  # Move data to the same device as the model
        y_pred = self.forward(x)
        y_pred = self.forward(x)
        loss = crps_loss(y, y_pred)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

# Create the model
model = ViViTClassifier(tubelet_embedder = TubeletEmbedding(EMBED_DIM, PATCH_SIZE), 
                        positional_encoder = PositionalEncoder(EMBED_DIM, NUM_PATCHES), 
                        num_layers = NUM_LAYERS,
                        num_heads = NUM_HEADS, 
                        embed_dim = EMBED_DIM, 
                        num_classes = num_classes_y, 
                        learning_rate = LEARNING_RATE, 
                        pdrop = DROP_RATE)

model = model.to('cpu')

# PRINT HYPERPARAMETERS
print("THIS IS ViViT_v0")
print("HYPERPARAMETERS:")
params = list(globals().items())
for name, value in params:
    if name.isupper():  # Assuming hyperparameters are in uppercase
        print(f"{name} = {value}")

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

# Save the model
torch.save(model.state_dict(), 'vivit_v0_torch_state_dict.pth')

