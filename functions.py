import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
# from pptx import Presentation
# from pptx.util import Inches
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F

class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(VAE, self).__init__()

        # dimensions for the intermediate layers
        dims = [input_dim, int(input_dim*0.75), int(input_dim*0.5), int(input_dim*0.25), latent_dim]

        # Encoder layers with BatchNorm
        encoder_layers = []
        for i in range(len(dims)-1):
            encoder_layers.append(nn.Linear(dims[i], dims[i+1]))
            encoder_layers.append(nn.BatchNorm1d(dims[i+1]))
            encoder_layers.append(nn.ReLU())
        self.encoder = nn.Sequential(*encoder_layers)

        # Latent space layers
        self.fc_mu = nn.Linear(dims[-1], latent_dim)
        self.fc_logvar = nn.Linear(dims[-1], latent_dim)

        # Decoder layers with BatchNorm
        decoder_layers = []
        for i in range(len(dims)-2, -1, -1):
            decoder_layers.append(nn.Linear(dims[i+1], dims[i]))
            decoder_layers.append(nn.BatchNorm1d(dims[i]))
            decoder_layers.append(nn.ReLU())
        decoder_layers.append(nn.Sigmoid()) 
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x):
        hidden = self.encoder(x)
        mu = self.fc_mu(hidden)
        logvar = self.fc_logvar(hidden)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def decode(self, z):
        output = self.decoder(z)
        return output

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        output = self.decode(z)
        return output, mu, logvar




def create_dataloader(train_tensor, batch_size):
    dataset = TensorDataset(train_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return dataloader




def kl_divergence(mu, logvar):
    # -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())







class FullConnectedBlock(nn.Module):

    def __init__(self, in_channels, out_channels, dropout_rate, use_residual=True, norm_layer=None):
        super(FullConnectedBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dropout_rate = dropout_rate
        self.use_residual = use_residual
        if norm_layer is None:
            norm_layer = nn.BatchNorm1d

        self.fc1 = nn.Linear(in_channels, out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.norm2 = norm_layer(out_channels)

    def forward(self, x):
        residual = x
        out = self.fc1(x)
        out = self.relu(out)
        out = F.dropout(out, p=self.dropout_rate, training=self.training)
        out = self.norm2(out)
        if self.use_residual and self.in_channels == self.out_channels:
            out += residual
        return out
#TRY dropout before relu
#try to remove relu

class NeuralNetwork(nn.Module):

    def __init__(self, input_dim, dropout_rate=0.05):
        super(NeuralNetwork, self).__init__()
        self.block1 = FullConnectedBlock(input_dim, 512, dropout_rate)
        self.block2 = FullConnectedBlock(512, 256, dropout_rate)
        self.block3 = FullConnectedBlock(256, 128, dropout_rate)
        self.fc_final = nn.Linear(128, 1)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        logits = self.fc_final(x)
        return logits


class NeuralNetwork2(nn.Module):

    def __init__(self, input_dim, dropout_rate=0.05):
        super(NeuralNetwork2, self).__init__()
        self.block1 = FullConnectedBlock(input_dim, 512, dropout_rate)
        self.block2 = FullConnectedBlock(512, 2048, dropout_rate)
        self.block3 = FullConnectedBlock(2048, 32768, dropout_rate)
        self.block4 = FullConnectedBlock(32768, 2048, dropout_rate)
        self.block5 = FullConnectedBlock(2048, 512, dropout_rate)
        self.fc_final = nn.Linear(512, 1)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        logits = self.fc_final(x)
        return logits













