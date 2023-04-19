import streamlit as st
import torch
import plotly.graph_objects as go
from const import Decoder, categories_to_full_name
import pandas as pd

st.set_page_config(layout="wide",
                   page_title="ECG generator",
                   menu_items={'About': "ECG generator - made by Antonin"},
                   )
# create css class to center the title "ECG generator"
st.markdown(
    """
    <style>
    .title {
        font-size:75px !important;
        font-weight: bold;
        box-sizing: border-box;
        text-align: left;
        width: 100%;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

categories = list(categories_to_full_name.values())

st.markdown('<div class="title">ECG generator</div>', unsafe_allow_html=True)
st.markdown('###')

# Selector and slider
col1, _, col2, _ = st.columns([1, 0.2, 1, 0.2])
# create a selector to select a disease
category = col1.selectbox('Select a type of ECG', categories, index=0, help='Select a type of ECG to generate')
# get the keys from the value in dictionary categories_to_full_name
category = list(categories_to_full_name.keys())[list(categories_to_full_name.values()).index(category)]
# create a slider to select the number of ECGs to generate
number_of_ecgs = col2.slider('Number of ECGs to generate', min_value=1, max_value=500, value=10, step=1)
st.markdown('###')

# Button
col1, _, col2, _ = st.columns([1, 0.2, 1, 0.2])
# create a button to generate the ECGs, centered
button = col1.button('Generate ECGs', key='button', help=f'Click to generate {number_of_ecgs} ECGs')
st.markdown('###')

# if the button is clicked
if button:
    # load the pytorch model at models/category_decoder_generation.pt
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
    _, col1, _ = st.columns([0.1, 1, 0.1])
    # plot with plotly all the generated ECGs on the same graph
    fig = go.Figure()
    for i in range(number_of_ecgs):
        fig.add_trace(go.Scatter(y=generated_ECG[i]))
    fig.update_layout(showlegend=False)
    # transparent background
    fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
    fig.update_layout(title_text=f'{number_of_ecgs} {categories_to_full_name[category]}')
    col1.plotly_chart(fig, use_container_width=True)

    # transform the generated ECGs into a pandas dataframe
    generated_ECG = pd.DataFrame(generated_ECG).T
    generated_ECG.columns = [f'ECG_{i}' for i in range(1, number_of_ecgs + 1)]
    # download the generated ECGs as a csv file by transforming the dataframe into bytes
    csv = generated_ECG.to_csv(index=False).encode()
    # create a button to download the generated ECGs
    col1.markdown('###')
    col1.download_button(label='Download ECGs as CSV', data=csv, file_name='ECGs.csv', mime='text/csv')
