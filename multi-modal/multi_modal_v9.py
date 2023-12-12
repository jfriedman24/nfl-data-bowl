# Program: Multi Modal Model v9
# Author: Jack Friedman
# Date: 12/11/2023
# Purpose: Implements multi modal transformer to predict tackle location for NFL Big Data Bowl 2024 


# Import libraries
import os
import io
import pickle
import gzip
import time 
import itertools
import matplotlib.pyplot as plt
from datetime import timedelta
import numpy as np
import pandas as pd
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping
import torch
from torch import Tensor, nn, optim
torch.set_float32_matmul_precision('medium')
from torch.nn import functional as F
from torchvision import datasets, transforms
from torch.nn import TransformerEncoder, TransformerEncoderLayer, TransformerDecoder, TransformerDecoderLayer
from torch.utils.data import Dataset, DataLoader
from torchmetrics.functional import accuracy, f1_score
from torchvision.models.vision_transformer import EncoderBlock
from sklearn.model_selection import train_test_split
from vision_transformer_mm import VisionTransformer
from tabnet import *
from functools import partial
from bidirectional_cross_attention import BidirectionalCrossAttention
import warnings
warnings.filterwarnings('ignore')
import sys
sys.path.append('../preprocessing')
from Preprocessing_v6 import *
from DataLoader import load_data


## Step 0: Define hyperparameters
# Data splitting params
SEED = 42
TEST_SIZE = 0.2
VAL_SIZE = 0.25  # % of the training data size (0.25 * 0.8 = 0.2 in this case)

# Optimizer
BATCH_SIZE = 16
LEARNING_RATE = 1e-5

# VIDEO PARAMS
FRAMES_PER_PLAY = 12
VIDEO_INPUT_SHAPE = (FRAMES_PER_PLAY, 120, 54, 10)

# embedding
VIDEO_PATCH_SIZE = (6, 6, 6)
VIDEO_NUM_PATCHES = (VIDEO_INPUT_SHAPE[0] // VIDEO_PATCH_SIZE[0]) * (VIDEO_INPUT_SHAPE[1] // VIDEO_PATCH_SIZE[1]) * (VIDEO_INPUT_SHAPE[2] // VIDEO_PATCH_SIZE[2])

# ViViT architecture
VIDEO_EMBED_DIM = 256
VIDEO_NUM_HEADS = 8
VIDEO_NUM_LAYERS = 12
VIDEO_DROP_RATE = 0.2
VIDEO_PRETRAIN = True

# IMAGE PARAMS
IMAGE_INPUT_SHAPE = (10, 54, 120)  # (C, H, W)

# embedding
IMAGE_PATCH_SIZE = (6, 6)
IMAGE_NUM_PATCHES = (IMAGE_INPUT_SHAPE[1] // IMAGE_PATCH_SIZE[0]) * (IMAGE_INPUT_SHAPE[2] // IMAGE_PATCH_SIZE[1])

# ViT ARCHITECTURE
IMAGE_EMBED_DIM = 256
IMAGE_NUM_HEADS = 8
IMAGE_NUM_LAYERS = 12
IMAGE_DROP_RATE = 0.2
IMAGE_GLOBAL_POOL = 'token'
IMAGE_2024_PRETRAIN = True 

# TabNet Params
TABNET_PARAMS = {'inp_dim': 351, 
                    'out_dim': 28,  # number of classes
                    'n_d': 32, 
                    'n_a': 32, 
                    'n_shared': 2, 
                    'n_ind': 2, 
                    'n_steps': 5,
                    'relax': 1.2, 
                    'vbs': BATCH_SIZE} 
OPTIMIZER_PARAMS = {'lr': 0.01}
TABNET_PRETRAIN = True

# Multimodal params
FUSION_OUTPUT_SIZE = 128
CROSS_ATTENTION = True
CROSS_ATTENTION_HEADS = 12
CROSS_ATTENTION_DIM = 128
MULTI_DROP_RATE = 0.1

## Step 1: Load and Preprocess Data
# Load data
[games_df, players_df, plays_df, tracking_df] = load_data()

### Step 1A: Video Data
# Preprocess data
tracking_df_clean = preprocess_all_df(plays_df, games_df, players_df, tracking_df)

# Get 4D tensors
play_ids, tensor_list, labels = prepare_4d_tensors(tracking_df_clean, min_frames = 12, tensor_type = 'torch')

### Step 1B: Image Data
# Get handoff trames
def get_final_frames(tensor_list):
    final_frames = []
    for tensor in tensor_list:
        frame = tensor[tensor.shape[0] - 1, :, :, :]
        final_frames += [frame]
    return final_frames

image_tensor_list = get_final_frames(tensor_list)

### Step 1C: Play Data
# Get clean plays data 
plays_df_clean = preprocess_plays_df_naive_models(plays_df, games_df, include_nfl_features = True)

# Get list of unique IDs from gtracking data with more than min frames
tracking_ids_df = pd.DataFrame(play_ids, columns = ['gameId', 'playId'])
# Reorder plays df so same order as video/image tensors
plays_df_clean = pd.merge(tracking_ids_df, plays_df_clean, on=['gameId', 'playId'], how='inner')

# Drop game and play ID
X_play = plays_df_clean.drop(['gameId', 'playId', 'TARGET'], axis = 1)
y_play = plays_df_clean['TARGET']

### Step 1D: Train-test Split
# Get indices for each set
indices = range(len(plays_df_clean))
train_idx, test_idx = train_test_split(indices, test_size=TEST_SIZE, random_state=SEED)
train_idx, val_idx = train_test_split(train_idx, test_size=VAL_SIZE, random_state=SEED)  # 0.25 x 0.8 = 0.2

# Video data
X_train_video, y_train_video = [tensor_list[i] for i in train_idx], [int(labels[i]) for i in train_idx]
X_val_video, y_val_video = [tensor_list[i] for i in val_idx], [int(labels[i]) for i in val_idx]
X_test_video, y_test_video = [tensor_list[i] for i in test_idx], [int(labels[i]) for i in test_idx]

# Image data
X_train_image, y_train_image = [image_tensor_list[i] for i in train_idx], [int(labels[i]) for i in train_idx]
X_val_image, y_val_image = [image_tensor_list[i] for i in val_idx], [int(labels[i]) for i in val_idx]
X_test_image, y_test_image = [image_tensor_list[i] for i in test_idx], [int(labels[i]) for i in test_idx]

# Play data
X_train_play, y_train_play = [X_play.iloc[i] for i in train_idx], [y_play.iloc[i] for i in train_idx]
X_val_play, y_val_play = [X_play.iloc[i] for i in val_idx], [y_play.iloc[i] for i in val_idx]
X_test_play, y_test_play = [X_play.iloc[i] for i in test_idx], [y_play.iloc[i] for i in test_idx]

### Step 1E: Create dataloaders
# Get key parameters for indexing labels
indexed_labels = [round(label) + 99 for label in labels]
min_idx_y = np.min(indexed_labels)
max_idx_y = np.max(indexed_labels)
print('min yardIndex:', min_idx_y)
print('max yardIndex:', max_idx_y)

num_classes_y = max_idx_y - min_idx_y + 1
print('num classes:', num_classes_y)
class MultimodalDataset(Dataset):
    def __init__(self, videos, images, plays, labels, num_classes_y, min_idx_y):
        self.videos = videos
        self.images = images
        self.plays = plays
        self.labels = labels
        self.num_classes_y = num_classes_y
        self.min_idx_y = min_idx_y

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # Process video
        video = self.videos[idx].float()
        video = video.permute(3, 0, 1, 2)  # From (T, W, H, C) to (C, T, H, W)

        # Process image
        image = self.images[idx].float()
        image = image.permute(2, 1, 0)  # from (W, H, C) to (C, H, W)

        # Process play data
        play = torch.tensor(self.plays[idx], dtype=torch.float32)

        # Process label
        label_indexed = int(round(self.labels[idx])) + 99
        label_one_hot = torch.zeros(self.num_classes_y, dtype=torch.float32)
        label_one_hot[label_indexed - self.min_idx_y] = 1.0

        return video, image, play, label_one_hot

def prepare_dataloader(videos, images, plays, labels, num_classes_y, min_idx_y, loader_type='train', batch_size=32):
    dataset = MultimodalDataset(videos, images, plays, labels, num_classes_y, min_idx_y)
    shuffle = loader_type == 'train'
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader


trainloader = prepare_dataloader(X_train_video, X_train_image, X_train_play, y_train_video, num_classes_y, min_idx_y, 'train', BATCH_SIZE)
validloader = prepare_dataloader(X_val_video, X_val_image, X_val_play, y_val_video, num_classes_y, min_idx_y, 'valid', BATCH_SIZE)
testloader = prepare_dataloader(X_test_video, X_test_image, X_test_play, y_test_video, num_classes_y, min_idx_y, 'test', BATCH_SIZE)

## Step 2: Build Model Archtecture

### A) Import models separately
#### (i) ViVIT

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
            # Regularization can be added here if needed
        )

    def forward(self, x):
        x2 = self.norm1(x)
        x = x + self.attn(x2, x2, x2)[0]
        x2 = self.norm2(x)
        x = x + self.mlp(x2)
        return x

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
            # Regularization can be added here if needed
        )

    def forward(self, x):
        x2 = self.norm1(x)
        x = x + self.attn(x2, x2, x2)[0]
        x2 = self.norm2(x)
        x = x + self.mlp(x2)
        return x


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

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

#### (ii) ViT
# Model architecture
class ViTLightningModule(pl.LightningModule):
    def __init__(self, learning_rate=1e-7):
        super().__init__()
        self.model = VisionTransformer(img_size = (IMAGE_INPUT_SHAPE[1], IMAGE_INPUT_SHAPE[2]),
            patch_size =IMAGE_PATCH_SIZE,
            in_chans = IMAGE_INPUT_SHAPE[0],
            num_classes = num_classes_y,
            global_pool = 'avg',
            embed_dim =IMAGE_EMBED_DIM,
            depth = IMAGE_NUM_LAYERS,
            num_heads = IMAGE_NUM_HEADS,
            drop_rate = IMAGE_DROP_RATE,
            # drop_path_rate = 0.3,
            pos_drop_rate = IMAGE_DROP_RATE,
            patch_drop_rate = IMAGE_DROP_RATE,
            proj_drop_rate = IMAGE_DROP_RATE,
            attn_drop_rate = IMAGE_DROP_RATE) 
        self.learning_rate = LEARNING_RATE

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
#### (iii) Play TabNet
# FROM https://github.com/microsoft/qlib/blob/main/qlib/contrib/model/pytorch_tabnet.py 

class GBN(nn.Module):
    """
    Ghost Batch Normalization
    an efficient way of doing batch normalization

    Args:
        vbs: virtual batch size
    """

    def __init__(self, inp, vbs=1024, momentum=0.01):
        super().__init__()
        self.bn = nn.BatchNorm1d(inp, momentum=momentum)
        self.vbs = vbs

    def forward(self, x):
        if x.size(0) <= self.vbs:  # can not be chunked
            return self.bn(x)
        else:
            chunk = torch.chunk(x, x.size(0) // self.vbs, 0)
            res = [self.bn(y) for y in chunk]
            return torch.cat(res, 0)


class GLU(nn.Module):
    """
    GLU block that extracts only the most essential information

    Args:
        vbs: virtual batch size
    """

    def __init__(self, inp_dim, out_dim, fc=None, vbs=1024):
        super().__init__()
        if fc:
            self.fc = fc
        else:
            self.fc = nn.Linear(inp_dim, out_dim * 2)
        self.bn = GBN(out_dim * 2, vbs=vbs)
        self.od = out_dim

    def forward(self, x):
        x = self.bn(self.fc(x))
        return torch.mul(x[:, : self.od], torch.sigmoid(x[:, self.od :]))


class AttentionTransformer(nn.Module):
    """
    Args:
        relax: relax coefficient. The greater it is, we can
        use the same features more. When it is set to 1
        we can use every feature only once
    """

    def __init__(self, d_a, inp_dim, relax, vbs=1024):
        super().__init__()
        self.fc = nn.Linear(d_a, inp_dim)
        self.bn = GBN(inp_dim, vbs=vbs)
        self.r = relax

    # a:feature from previous decision step
    def forward(self, a, priors):
        a = self.bn(self.fc(a))
        mask = SparsemaxFunction.apply(a * priors)
        priors = priors * (self.r - mask)  # updating the prior
        return mask


class FeatureTransformer(nn.Module):
    def __init__(self, inp_dim, out_dim, shared, n_ind, vbs):
        super().__init__()
        first = True
        self.shared = nn.ModuleList()
        if shared:
            self.shared.append(GLU(inp_dim, out_dim, shared[0], vbs=vbs))
            first = False
            for fc in shared[1:]:
                self.shared.append(GLU(out_dim, out_dim, fc, vbs=vbs))
        else:
            self.shared = None
        self.independ = nn.ModuleList()
        if first:
            self.independ.append(GLU(inp_dim, out_dim, vbs=vbs))
        for x in range(first, n_ind):
            self.independ.append(GLU(out_dim, out_dim, vbs=vbs))
        self.scale = float(np.sqrt(0.5))

    def forward(self, x):
        if self.shared:
            x = self.shared[0](x)
            for glu in self.shared[1:]:
                x = torch.add(x, glu(x))
                x = x * self.scale
        for glu in self.independ:
            x = torch.add(x, glu(x))
            x = x * self.scale
        return x


class DecisionStep(nn.Module):
    """
    One step for the TabNet
    """

    def __init__(self, inp_dim, n_d, n_a, shared, n_ind, relax, vbs):
        super().__init__()
        self.atten_tran = AttentionTransformer(n_a, inp_dim, relax, vbs)
        self.fea_tran = FeatureTransformer(inp_dim, n_d + n_a, shared, n_ind, vbs)

    def forward(self, x, a, priors):
        mask = self.atten_tran(a, priors)
        sparse_loss = ((-1) * mask * torch.log(mask + 1e-10)).mean()
        x = self.fea_tran(x * mask)
        return x, sparse_loss


def make_ix_like(input, dim=0):
    d = input.size(dim)
    rho = torch.arange(1, d + 1, device=input.device, dtype=input.dtype)
    view = [1] * input.dim()
    view[0] = -1
    return rho.view(view).transpose(0, dim)


class SparsemaxFunction(Function):
    """
    SparseMax function for replacing reLU
    """

    @staticmethod
    def forward(ctx, input, dim=-1):
        ctx.dim = dim
        max_val, _ = input.max(dim=dim, keepdim=True)
        input -= max_val  # same numerical stability trick as for softmax
        tau, supp_size = SparsemaxFunction.threshold_and_support(input, dim=dim)
        output = torch.clamp(input - tau, min=0)
        ctx.save_for_backward(supp_size, output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        supp_size, output = ctx.saved_tensors
        dim = ctx.dim
        grad_input = grad_output.clone()
        grad_input[output == 0] = 0

        v_hat = grad_input.sum(dim=dim) / supp_size.to(output.dtype).squeeze()
        v_hat = v_hat.unsqueeze(dim)
        grad_input = torch.where(output != 0, grad_input - v_hat, grad_input)
        return grad_input, None

    @staticmethod
    def threshold_and_support(input, dim=-1):
        input_srt, _ = torch.sort(input, descending=True, dim=dim)
        input_cumsum = input_srt.cumsum(dim) - 1
        rhos = make_ix_like(input, dim)
        support = rhos * input_srt > input_cumsum

        support_size = support.sum(dim=dim).unsqueeze(dim)
        tau = input_cumsum.gather(dim, support_size - 1)
        tau /= support_size.to(input.dtype)
        return tau, support_size
    
class TabNet(nn.Module):
    def __init__(self, inp_dim=15, out_dim=6, n_d=64, n_a=64, n_shared=2, n_ind=2, n_steps=5, relax=1.2, vbs=1024):
        """
        Args:
            n_d: dimension of the features used to calculate the final results
            n_a: dimension of the features input to the attention transformer of the next step
            n_shared: numbr of shared steps in feature transformer(optional)
            n_ind: number of independent steps in feature transformer
            n_steps: number of steps of pass through tabbet
            relax coefficient:
            virtual batch size:
        """
        super().__init__()

        # set the number of shared step in feature transformer
        if n_shared > 0:
            self.shared = nn.ModuleList()
            self.shared.append(nn.Linear(inp_dim, 2 * (n_d + n_a)))
            for x in range(n_shared - 1):
                self.shared.append(nn.Linear(n_d + n_a, 2 * (n_d + n_a)))  # preset the linear function we will use
        else:
            self.shared = None

        self.first_step = FeatureTransformer(inp_dim, n_d + n_a, self.shared, n_ind, vbs)
        self.steps = nn.ModuleList()
        for x in range(n_steps - 1):
            self.steps.append(DecisionStep(inp_dim, n_d, n_a, self.shared, n_ind, relax, vbs))
        self.fc = nn.Linear(n_d, out_dim)
        self.bn = nn.BatchNorm1d(inp_dim, momentum=0.01)
        self.n_d = n_d

    def forward(self, x, priors):
        assert not torch.isnan(x).any()
        x = self.bn(x)
        x_a = self.first_step(x)[:, self.n_d :]
        sparse_loss = []
        out = torch.zeros(x.size(0), self.n_d).to(x.device)
        for step in self.steps:
            x_te, loss = step(x, x_a, priors)
            out += F.relu(x_te[:, : self.n_d])  # split the feature from feat_transformer
            x_a = x_te[:, self.n_d :]
            sparse_loss.append(loss)
        return self.fc(out), sum(sparse_loss)


class LightningTabNet(pl.LightningModule):
    def __init__(self, tabnet_params, optimizer_params):
        super().__init__()
        self.save_hyperparameters()
        
        self.tabnet_model = TabNet(**tabnet_params)

    def forward(self, x, priors = None):
        if priors is None:
            # If priors are not provided, we initialize them as ones
            priors = torch.ones(x.shape[0], self.hparams.tabnet_params['inp_dim']).to(self.device)
        return self.tabnet_model(x, priors)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), **self.hparams.optimizer_params)
        return optimizer

    def loss_fn(self, pred, label):
        loss = torch.mean(torch.sum((torch.cumsum(pred, dim=1) - torch.cumsum(label, dim=1)) ** 2, dim=1)) / 199
        return loss
    
### B) Define Multi-Modal Transformer

# Loss function - Continuous Ranked Probability Score
def crps_loss(y_true, y_pred):
    loss = torch.mean(torch.sum((torch.cumsum(y_pred, dim=1) - torch.cumsum(y_true, dim=1)) ** 2, dim=1)) / 199
    return loss

class MultimodalTransformerModule(pl.LightningModule): 
    def __init__(self, video_model, vision_model, tabular_model, fusion_output_size, task_output_size, learning_rate, cross_attention = False):
        super().__init__()
        self.video_model = video_model
        self.vision_model = vision_model
        self.tabular_model = tabular_model

        self.cross_attention = cross_attention
        
        # Assuming each model outputs a vector of size fusion_output_size
        transformer_input_size = 28 * 3  # Concatenated features from 3 models

        # Transformer Encoder Layer
        self.transformer_encoder = TransformerEncoder(TransformerEncoderLayer(
            d_model=transformer_input_size, nhead=4, dim_feedforward=256), num_layers=22)

        # Transformer Decoder Layer
        self.transformer_decoder = TransformerDecoder(TransformerDecoderLayer(
            d_model=transformer_input_size, nhead=4, dim_feedforward=256), num_layers=22)

        # Final task layer
        self.task_layer = nn.Sequential(
            nn.Linear(transformer_input_size, task_output_size),
            nn.Softmax(dim = 1)
        )

        self.learning_rate = learning_rate

    def forward(self, video_data, image_data, tabular_data):
        device = self.device
        video_data = video_data.to(device)
        image_data = image_data.to(device)
        tabular_data = tabular_data.to(device)

        video_features = self.video_model(video_data)
        image_features = self.vision_model(image_data)
        tabular_features, _ = self.tabular_model(tabular_data)

        # Concatenate features
        x = torch.cat((video_features, image_features, tabular_features), dim=1)

        # Transformer Encoding and Decoding
        x = x.unsqueeze(0)  # Add sequence length dimension for transformer
        x = self.transformer_encoder(x)
        x = self.transformer_decoder(x, x)
        x = x.squeeze(0)  # Remove sequence length dimension

        # Task specific layer
        x = self.task_layer(x)
        return x

    def training_step(self, batch, batch_idx):
        video_data, image_data, tabular_data, labels = batch
        outputs = self(video_data, image_data, tabular_data)
        loss = crps_loss(outputs, labels)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        video_data, image_data, tabular_data, labels = batch
        outputs = self(video_data, image_data, tabular_data)
        loss = crps_loss(outputs, labels)
        self.log('val_loss', loss)

    def test_step(self, batch, batch_idx):
        video_data, image_data, tabular_data, labels = batch
        outputs = self(video_data, image_data, tabular_data)
        loss = crps_loss(outputs, labels)
        self.log('test_loss', loss)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
    

# Function that assembles the multimodal modal
def build_model():
    video_model = ViViTClassifier(tubelet_embedder = TubeletEmbedding(VIDEO_EMBED_DIM, VIDEO_PATCH_SIZE), 
                            positional_encoder = PositionalEncoder(VIDEO_EMBED_DIM, VIDEO_NUM_PATCHES), 
                            num_layers = VIDEO_NUM_LAYERS,
                            num_heads = VIDEO_NUM_HEADS, 
                            num_classes = num_classes_y,
                            embed_dim = VIDEO_EMBED_DIM, 
                            learning_rate = LEARNING_RATE, 
                            pdrop = VIDEO_DROP_RATE)
    if VIDEO_PRETRAIN:
        pretrained_dict = torch.load("../vivit_model/vivit_v0_torch_state_dict.pth")
        model_dict = video_model.state_dict()

        # 1. filter out unnecessary keys
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict) 
        # 3. load the new state dict
        video_model.load_state_dict(model_dict)

    # # Freeze the weights of video_model
    for param in video_model.parameters():
        param.requires_grad = False

    vision_model = ViTLightningModule(learning_rate=LEARNING_RATE)
    if IMAGE_2024_PRETRAIN:
        pretrained_dict = torch.load("../vit_model/vit_v7_state_dict.pth")
        model_dict = vision_model.state_dict()

        # 1. filter out unnecessary keys
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict) 
        # 3. load the new state dict
        vision_model.load_state_dict(model_dict)
    
    # # Freeze the weights of video_model
    for param in vision_model.parameters():
        param.requires_grad = False

    tabular_model = LightningTabNet(TABNET_PARAMS, OPTIMIZER_PARAMS)
    if TABNET_PRETRAIN:
        pretrained_dict = torch.load("../naive_model/tabnet_v0_state_dict.pth")
        model_dict = tabular_model.state_dict()

        # 1. filter out unnecessary keys
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict) 
        # 3. load the new state dict
        tabular_model.load_state_dict(model_dict)

    # # Freeze the weights of video_model
    for param in tabular_model.parameters():
        param.requires_grad = False

    multimodal_model = MultimodalTransformerModule(video_model, vision_model, tabular_model, 
                                            fusion_output_size = FUSION_OUTPUT_SIZE, 
                                            task_output_size = num_classes_y, 
                                            learning_rate = LEARNING_RATE,
                                            cross_attention = CROSS_ATTENTION)
    multimodal_model.to('cpu')
    
    return multimodal_model

## Step 3: Train and Evaluate Model
# PRINT HYPERPARAMETERS
print("THIS IS v9")
print("HYPERPARAMETERS:")
params = list(globals().items())
for name, value in params:
    if name.isupper():  # Assuming hyperparameters are in uppercase
        print(f"{name} = {value}")
multimodal_model = build_model()
print("MODEL ARCHITECTURE:", multimodal_model)

# Set up early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=10)

# Create the trainer with early stopping
trainer = pl.Trainer(callbacks=[early_stopping], enable_progress_bar=False)



# Train the model with the DataLoaders
print("training model...")
trainer.fit(multimodal_model, trainloader, validloader)
print("finished trainining")

# Get train loss
print("\nGET TRAINING LOSS:")
trainer.test(multimodal_model, trainloader)
print("^^^^ THIS ABOVE NUMBER IS THE TRAIN SET LOSS\n")

print("\nGET VALIDATION SET LOSS:")
trainer.validate(multimodal_model, validloader)
print("^^^^ THIS ABOVE NUMBER IS THE VALIDATION SET LOSS\n")

# Evaluate the model on the test set
print("\n GET TEST SET LOSS:")
trainer.test(multimodal_model, testloader)
print("^^^^ THIS ABOVE NUMBER IS THE TEST SET LOSS\n")

# Save the model
torch.save(multimodal_model.state_dict(), 'mm_v9_state_dict.pth')
