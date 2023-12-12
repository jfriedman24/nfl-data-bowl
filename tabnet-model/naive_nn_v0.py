# Naive Model (Play-Level Data) Neural Network v0

# Author: Jack Friedman 
# Date: 11/27/2023 
# Purpose: Program that uses the play-level data to predict expected yards gained using TabNet model 

# Import libraries
import os
import io
import pickle
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
from torch.autograd import Function
from torchmetrics.functional import accuracy, f1_score
from torchvision.models.vision_transformer import EncoderBlock
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')
import sys
sys.path.append('../preprocessing')
from Preprocessing_v6 import *
from DataLoader import load_data, load_data_tubevit
## Step 0: Hyperparams
TEST_SIZE = 0.2
VAL_SIZE = 0.25 
SEED = 42

BATCH_SIZE = 32
LEARNING_RATE = 0.01

## Step 1: Load Data
# Load data
[games_df, players_df, plays_df, tracking_df] = load_data()
# Get clean plays data 
plays_df_clean = preprocess_plays_df_naive_models(plays_df, games_df, include_nfl_features = True)

# Drop game and play ID
X_play = plays_df_clean.drop(['gameId', 'playId', 'TARGET'], axis = 1)
y_play = plays_df_clean['TARGET']
plays_df_clean.isna().sum().sum()
X_train, X_test, y_train, y_test = train_test_split(X_play, y_play, test_size=TEST_SIZE, random_state=SEED)

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=VAL_SIZE, random_state=SEED) # 0.25 x 0.8 = 0.2
# Get key parameters for indexing labels
indexed_labels = [round(label) + 99 for label in y_play]
min_idx_y = np.min(indexed_labels)
max_idx_y = np.max(indexed_labels)
print('min yardIndex:', min_idx_y)
print('max yardIndex:', max_idx_y)

num_classes_y = max_idx_y - min_idx_y + 1
print('num classes:', num_classes_y)
### Dataloaders
# Build data loaders
class TabularDataset(Dataset):
    def __init__(self, data, labels, num_classes_y, min_idx_y):
        self.data = data
        self.labels = labels
        self.num_classes_y = num_classes_y
        self.min_idx_y = min_idx_y

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Explicitly convert data
        data_row = self.data.iloc[idx]
        data_array = np.array(data_row, dtype=np.float32)
        data_tensor = torch.from_numpy(data_array)

        # Preprocess the label
        label_indexed = int(round(self.labels.iloc[idx].item())) + 99
        label_one_hot = torch.zeros(self.num_classes_y, dtype=torch.float32)
        label_one_hot[label_indexed - self.min_idx_y] = 1.0

        return data_tensor, label_one_hot
    
def prepare_dataloader(data, labels, num_classes_y, min_idx_y, loader_type='train', batch_size=32):
    dataset = TabularDataset(data, labels, num_classes_y, min_idx_y)
    shuffle = loader_type == 'train'
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader


trainloader = prepare_dataloader(X_train, y_train, num_classes_y, min_idx_y, 'train', BATCH_SIZE)
validloader = prepare_dataloader(X_val, y_val, num_classes_y, min_idx_y, 'valid', BATCH_SIZE)
testloader = prepare_dataloader(X_test, y_test, num_classes_y, min_idx_y, 'test', BATCH_SIZE)


## Step 2; Model archtecture
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

    def forward(self, x, priors):
        return self.tabnet_model(x, priors)

    def training_step(self, batch, batch_idx):
        x, y = batch
        priors = torch.ones(x.shape[0], self.hparams.tabnet_params['inp_dim']).to(self.device)
        y_hat, _ = self(x, priors)
        loss = self.loss_fn(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        priors = torch.ones(x.shape[0], self.hparams.tabnet_params['inp_dim']).to(self.device)
        y_hat, _ = self(x, priors)
        loss = self.loss_fn(y_hat, y)
        self.log('val_loss', loss)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        priors = torch.ones(x.shape[0], self.hparams.tabnet_params['inp_dim']).to(self.device)
        y_hat, _ = self(x, priors)
        loss = self.loss_fn(y_hat, y)
        self.log('test_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), **self.hparams.optimizer_params)
        return optimizer

    def loss_fn(self, pred, label):
        loss = torch.mean(torch.sum((torch.cumsum(pred, dim=1) - torch.cumsum(label, dim=1)) ** 2, dim=1)) / 199
        return loss

## Step 3: Train

# Define your TabNet parameters and optimizer parameters
tabnet_params = {'inp_dim':X_play.shape[1], 
                 'out_dim':num_classes_y, 
                 'n_d': 32, 
                 'n_a': 32, 
                 'n_shared':2, 
                 'n_ind':2, 
                 'n_steps':5, 
                 'relax':1.2, 
                 'vbs':BATCH_SIZE 
}
optimizer_params = {
    'lr': LEARNING_RATE,
}


# Initialize the Lightning module
lightning_model = LightningTabNet(tabnet_params, optimizer_params)

# Train the model

def print_dataloader_device(dataloader, dataloader_name):
    for batch in dataloader:
        x, labels = batch
        print(f"{dataloader_name} - Batch features device: {x.device}, Batch labels device: {labels.device}")
        break  # Only checking the first batch

print_dataloader_device(trainloader, "TrainLoader")
print_dataloader_device(validloader, "ValidLoader")
print_dataloader_device(testloader, "TestLoader")
print("model device:",lightning_model.device)


# Set up early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=10)

# Create the trainer with early stopping
trainer = pl.Trainer(callbacks=[early_stopping], enable_progress_bar=False)


# Train the model with the DataLoaders
print("training model...")
trainer.fit(lightning_model, trainloader, validloader)
print("finished trainining")

# Get train loss
print("\nGET TRAINING LOSS:")
trainer.test(lightning_model, trainloader)
print("^^^^ THIS ABOVE NUMBER IS THE TRAIN SET LOSS\n")

print("\nGET VALIDATION SET LOSS:")
trainer.validate(lightning_model, validloader)
print("^^^^ THIS ABOVE NUMBER IS THE VALIDATION SET LOSS\n")

# Evaluate the model on the test set
print("\n GET TEST SET LOSS:")
trainer.test(lightning_model, testloader)
print("^^^^ THIS ABOVE NUMBER IS THE TEST SET LOSS\n")


print("tabent v0 params")
print(tabnet_params)

# Save the model
torch.save(lightning_model.state_dict(), 'tabnet_v0_state_dict.pth')
