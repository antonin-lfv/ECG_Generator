from fileinput import filename
from data_managing import *
import torch
from tqdm import tqdm
import random
from data_managing import *
from VAE import VariationalAutoencoder, Decoder
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


if __name__ == "__main__":
    import plotly.graph_objects as go
    from plotly.offline import plot
    
    print("Test de création de plusieurs ECG à partir d'un decodeur entraîné")
    number_of_ecgs = 5
    category = "aAP"
    
    model = Decoder()
    model.load_state_dict(torch.load(f'models/{category}_decoder_generation.pt'))
    model.eval()
    # generate the ECGs
    generated_ECG = []
    for i in range(number_of_ecgs):
        with torch.no_grad():
            latent_vector_shape = (1, 25)
            # on échantillonne les distributions du vecteur latent pour en générer un nouveau
            new_latent_ECG = torch.distributions.Normal(0, 1).sample(latent_vector_shape)
            # on décode le vecteur latent pour obtenir un nouvel ECG
            generated_ECG.append(model(new_latent_ECG).detach().numpy().flatten())
    
    categories_to_full_name = {'NOR': 'Normal beat (NOR)',
                           'LBBB': 'Left bundle branch block beat (LBBB)',
                           'RBBB': 'Right bundle branch block beat (RBBB)',
                           'NE': 'Nodal (junctional) escape beat (NE)',
                           'AP': 'Atrial premature beat (AP)',
                           'aAP': 'Aberrated atrial premature beat (aAP)',
                           'NP': 'Nodal (junctional) premature beat (NP)',
                           'PVC': 'Premature ventricular contraction (PVC)',
                           'VE': 'Ventricular escape beat (VE)',
                           'fVN': 'Fusion of ventricular and normal beat (VF)',
                           }

    # plot with plotly all the generated ECGs on the same graph
    fig = go.Figure()
    for i in range(number_of_ecgs):
        fig.add_trace(go.Scatter(y=generated_ECG[i]))
    fig.update_layout(showlegend=False)
    # transparent background
    fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
    fig.update_layout(title_text=f'{number_of_ecgs} {categories_to_full_name[category]}')
    plot(fig, filename='test/ECG_test_generation.html')
