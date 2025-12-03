import plotly.graph_objects as go
from plotly.offline import plot
import torch
from VAE import Decoder

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