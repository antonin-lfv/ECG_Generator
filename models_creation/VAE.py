from data_managing import *
import torch
from torch import nn
import torch.nn.functional as F

class Encoder(nn.Module):
    """
    Encode les données et renvoie le vecteur latent
    """

    def __init__(self):
        super(Encoder, self).__init__()
        # Couches convolutives
        self.conv1 = nn.Conv2d(1, 8, stride=1, padding=2, kernel_size=5)
        self.convbatchNorm1 = nn.BatchNorm2d(8)
        self.maxPool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(8, 16, stride=1, padding=2, kernel_size=5)
        self.convbatchNorm2 = nn.BatchNorm2d(16)
        self.maxPool2 = nn.MaxPool2d(kernel_size=7, stride=5, padding=3)
        self.conv3 = nn.Conv2d(16, 1, kernel_size=5, stride=1, padding=2)
        # Couches denses
        self.dense1 = nn.Linear(250, 125)  # 250 - 125
        self.dense2 = nn.Linear(125, 25)  # 125 - 25
        # Couches concatenation
        self.concatDense1 = nn.Linear(50, 25)  # 50 - 25
        self.concatDense2 = nn.Linear(50, 25)  # 50 - 25
        # tools
        self.N = torch.distributions.Normal(0, 1)  # standard normal distribution
        self.kl = 0  # KL divergence

    def forward(self, x):
        # Convolution
        x_conv = x.unsqueeze(0).unsqueeze(0)  # pour avoir shape (1, 1, 1, 250)
        x_conv = F.relu(self.convbatchNorm1(self.conv1(x_conv)))
        x_conv = self.maxPool1(x_conv)
        x_conv = F.relu(self.convbatchNorm2(self.conv2(x_conv)))
        x_conv = self.maxPool2(x_conv)
        x_conv = self.conv3(x_conv)
        x_conv = x_conv.view(25)
        # Dense
        x = x.flatten()  # pour avoir shape (250)
        x_dense = F.relu(self.dense1(x))
        x_dense = self.dense2(x_dense)
        # Concatenation
        x_concat = torch.cat((x_conv, x_dense), 0)
        mu = self.concatDense1(x_concat)  # moyenne
        sigma = torch.exp(self.concatDense2(x_concat))  # écart type
        # Vecteur latent
        z = mu + sigma * self.N.sample(mu.shape)
        # self.kl = (sigma ** 2 + mu ** 2 - torch.log(sigma) - 1 / 2).sum()  # last KL divergence
        self.kl = 0.5 * torch.sum(sigma - torch.log(sigma) - 1 + mu ** 2)  # KL divergence
        return z


class Decoder(nn.Module):
    """
    Décode le vecteur latent et renvoie les données reconstruites
    """

    def __init__(self):
        super(Decoder, self).__init__()
        # Couches convolutives
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2)  # (1, 16, 1, 25)
        self.convbatchNorm1 = nn.BatchNorm2d(16)
        self.upsampling1 = nn.Upsample(size=(1, 250))  # (1, 16, 1, 250)
        self.ConvUp1 = nn.ConvTranspose2d(16, 8, kernel_size=1, stride=1)  # (1, 8, 1, 250)
        self.ConvUp2 = nn.ConvTranspose2d(8, 1, kernel_size=1, stride=1)  # (1, 1, 1, 250)
        # Couches denses
        self.dense1 = nn.Linear(25, 25)
        self.dense2 = nn.Linear(25, 125)
        self.dense3 = nn.Linear(125, 250)
        # concaténation
        self.concatDense1 = nn.Linear(500, 250)

    def forward(self, z):
        # Couches convolutives
        z_conv = z.reshape((1, 25)).unsqueeze(0).unsqueeze(0)
        z_conv = F.relu(self.convbatchNorm1(self.conv1(z_conv)))
        z_conv = self.upsampling1(z_conv)
        z_conv = self.ConvUp1(z_conv)
        z_conv = self.ConvUp2(z_conv)
        z_conv = z_conv.flatten()
        # Couches denses
        z_dense = self.dense1(z)
        z_dense = F.relu(self.dense2(z_dense))
        z_dense = F.relu(self.dense3(z_dense))
        z_dense = z_dense.flatten()
        # Concaténation
        z_dense = torch.cat((z_conv, z_dense), 0)
        # Dernières couches denses
        z_final = self.concatDense1(z_dense)
        z_final = z_final.reshape(1, 250)
        return z_final


class VariationalAutoencoder(nn.Module):
    def __init__(self):
        super(VariationalAutoencoder, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)
