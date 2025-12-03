from data_managing import *
import torch
from tqdm import tqdm
import random
from data_managing import *
from VAE import VariationalAutoencoder
from torch.utils.data import DataLoader

seed = random.seed

def train(vae, data_dataloader, epochs, class_name):
    """
    Entraîne le VAE
    :param vae: le modèle
    :param data_dataloader: les données
    :param epochs: nombre d'epochs
    :param class_name: nom de la classe de l'ECG
    :return: le modèle entraîné et les erreurs à chaque epoch
    """
    loss_evolution = []
    opt = torch.optim.Adam(autoencoder.parameters())
    print(f"[INFO] Training started for class {class_name}")
    for _ in tqdm(range(epochs)):
        for i, (x, _) in enumerate(data_dataloader):
            opt.zero_grad()
            x_hat = vae(x)
            # erreur de reconstruction avec MSE avec ajout de la divergence KL
            loss = ((x - x_hat) ** 2).sum() + autoencoder.encoder.kl
            if i == (len(x) - 1):
                # on ne veut que le dernier loss de chaque epoch
                loss_evolution.append(loss.detach().numpy())
            loss.backward()
            opt.step()
    return vae, loss_evolution


# Chargement unique du dataset complet
ECG = ECGDataset()
train_data, _ = ECG.get_data()

# Boucle sur les différentes classes
classes_name = ['NOR', 'LBBB', 'RBBB', 'AE', 'NE', 'AP', 'aAP', 'SP', 'NP', 'PVC', 'VE', 'fVN']
for class_name in classes_name:
    ECG_filtered_dataset = ECGDatasetFiltered(train_data, ECG_class=class_name)
    ECG_dataloader = DataLoader(ECG_filtered_dataset, batch_size=1, shuffle=True)
    autoencoder = VariationalAutoencoder()
    vae_trained, loss_evolution = train(autoencoder, ECG_dataloader, 200, class_name)
    torch.save(vae_trained.decoder.state_dict(), f"models/{class_name}_decoder_generation.pt")
