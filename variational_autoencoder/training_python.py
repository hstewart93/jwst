import numpy as np
import os
import pandas as pd
import torch
import torchvision

from matplotlib import pyplot as plt
# from plotly import express as px
# from sklearn.manifold import TSNE
# from skimage.transform import rotate
from torch.utils.data import DataLoader, random_split
from torch import from_numpy, nn
from torch.nn import functional as F
from torchvision.utils import make_grid
from torchvision import transforms
from tqdm import tqdm
# from time import perf_counter

from variational_autoencoder.model import VariationalAutoencoder

TRAIN_DATA_PATH = f"{os.environ['HOME']}/Code/jwst/data/train/"
TEST_DATA_PATH = f"{os.environ['HOME']}/Code/jwst/data/test/"
VALIDATION_DATA_PATH = f"{os.environ['HOME']}/Code/jwst/data/validation/"
BATCH_SIZE = 32

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
#     transforms.RandomHorizontalFlip(),
#     transforms.RandomVerticalFlip(),
])

# Train data
train_dataset_data = torchvision.datasets.ImageFolder(root=TRAIN_DATA_PATH, transform=transform)
train_loader_data = torch.utils.data.DataLoader(train_dataset_data, batch_size=BATCH_SIZE, shuffle=True)

train_length_data = len(train_dataset_data)

# Test data
test_dataset_data = torchvision.datasets.ImageFolder(root=TEST_DATA_PATH, transform=transform)
test_loader_data = torch.utils.data.DataLoader(test_dataset_data, batch_size=BATCH_SIZE, shuffle=True)

test_length_data = len(test_dataset_data)

# Validation data
validation_dataset_data = torchvision.datasets.ImageFolder(root=VALIDATION_DATA_PATH, transform=transform)
validation_loader_data = torch.utils.data.DataLoader(validation_dataset_data, batch_size=BATCH_SIZE, shuffle=True)

validation_length_data = len(validation_dataset_data)

# Set the random seed for reproducible results
torch.manual_seed(0)

# Check if the GPU is available
train_device = torch.device("cpu") if torch.cuda.is_available() else torch.device("cpu")
print(f"Selected device: {train_device}")

# Use Autoencoder class to instantiate a model
# input_dims is 1 as the input image has 1 channel
variational_autoencoder = VariationalAutoencoder(input_dims=1, latent_dims=3).to(train_device)

# Define the optimizer
# train_optimizer = torch.optim.SGD(variational_autoencoder.parameters(), lr=1e-5, weight_decay=1e-05)
LEARNING_RATE = 0.001
train_optimizer = torch.optim.Adam(variational_autoencoder.parameters(), lr=LEARNING_RATE)

# Define the loss function
def loss_function(image, reconstruction, mean, log_variance):
    """This function will add the reconstruction loss and the KL-Divergence.
    KL-Divergence = 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)"""
    
    # Binary cross entropy loss
    criterion = nn.BCELoss(reduction="sum")
    try:
        bce_loss = criterion(reconstruction, image)
    except RuntimeError:
        import ipdb; ipdb.set_trace(context=25)
    # KL Divergence term
    kl_divergence = -0.5 * torch.sum(1 + log_variance - mean.pow(2) - log_variance.exp())

    return bce_loss + kl_divergence

def train(model, device, dataset, dataloader, loss_method, optimizer):
    """Function to calculate the training loss for a single epoch."""

    # Set train mode for the model
    model.train()
    train_loss = 0

    for i, data in tqdm(enumerate(dataloader), total=int(len(dataset) / dataloader.batch_size)):

        # Move tensor to the proper device
        data = data[0].to(device)
        optimizer.zero_grad()
        
        # Compute reconstructions
        output, mu, log_var = model(data)
        
        # Evaluate loss
        loss = loss_method(data, output, mu, log_var)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    return train_loss / len(dataloader)

def validate(model, device, dataset, dataloader, loss_method):
    model.eval()
    validation_loss = 0
    with torch.no_grad():
        for i, data in tqdm(enumerate(dataloader), total=int(len(dataset) / dataloader.batch_size)):
            # Move tensor to the proper device
            data = data[0].to(device)
            # Compute reconstructions
            output, mu, log_var = model(data)

            # Evaluate loss
            loss = loss_method(data, output, mu, log_var)
            validation_loss += loss.item()

            # save the last batch input and output of every epoch
            if i == int(len(dataset)/dataloader.batch_size) - 1:
                reconstructed_images = output

    return validation_loss / len(dataloader), reconstructed_images

training_loss = []
validation_loss = []
grid_images = []
EPOCHS = 40

for epoch in range(EPOCHS):
    print(f"Epoch {epoch+1} of {EPOCHS}")
    train_epoch_loss = train(
        variational_autoencoder, train_device, train_dataset_data, train_loader_data, loss_function, train_optimizer
    )
    validation_epoch_loss, reconstructed_images = validate(
        variational_autoencoder, train_device, test_dataset_data, validation_loader_data, loss_function
    )
    training_loss.append(train_epoch_loss)
    validation_loss.append(validation_epoch_loss)

    #convert the reconstructed images to PyTorch image grid format
    # image_grid = make_grid(recon_images.detach().cpu())
    # grid_images.append(image_grid)
    print(f"Train Loss: {train_epoch_loss:.4f}")
    print(f"Validation Loss: {validation_epoch_loss:.4f}")