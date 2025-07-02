import streamlit as st
import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
import matplotlib.pyplot as plt

# ===== モデル定義 =====

class Encoder(nn.Module):
    def __init__(self, latent_dim=3, num_classes=10):
        super().__init__()
        self.label_embedding = nn.Linear(num_classes, 16)
        self.fc_hidden = nn.Linear(28 * 28 + 16, 128)
        self.fc_mu = nn.Linear(128, latent_dim)
        self.fc_logvar = nn.Linear(128, latent_dim)

    def forward(self, image, label):
        flattened_image = image.view(image.size(0), -1)
        label_one_hot = F.one_hot(label, num_classes=10).float()
        label_embedding = F.relu(self.label_embedding(label_one_hot))
        concatenated_input = torch.cat([flattened_image, label_embedding], dim=1)
        hidden = F.relu(self.fc_hidden(concatenated_input))
        mu = self.fc_mu(hidden)
        logvar = self.fc_logvar(hidden)
        return mu, logvar

class Decoder(nn.Module):
    def __init__(self, latent_dim=3, num_classes=10):
        super().__init__()
        self.label_embedding = nn.Linear(num_classes, 16)
        self.fc_hidden = nn.Linear(latent_dim + 16, 128)
        self.fc_out = nn.Linear(128, 28 * 28)

    def forward(self, latent_vector, label):
        label_one_hot = F.one_hot(label, num_classes=10).float()
        label_embedding = F.relu(self.label_embedding(label_one_hot))
        concatenated = torch.cat([latent_vector, label_embedding], dim=1)
        hidden = F.relu(self.fc_hidden(concatenated))
        output = torch.sigmoid(self.fc_out(hidden))
        return output.view(-1, 1, 28, 28)

class CVAE(nn.Module):
    def __init__(self, latent_dim=3, num_classes=10):
        super().__init__()
        self.encoder = Encoder(latent_dim, num_classes)
        self.decoder = Decoder(latent_dim, num_classes)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        epsilon = torch.randn_like(std)
        return mu + std * epsilon

    def forward(self, image, label):
        mu, logvar = self.encoder(image, label)
        z = self.reparameterize(mu, logvar)
        reconstructed = self.decoder(z, label)
        return reconstructed, mu, logvar

# ===== Streamlit UI =====

st.title("CVAE による数字画像生成 (matplotlib表示)")

# デバイス設定
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@st.cache_resource
def load_model(path="cvae_model.pth"):
    model = CVAE()
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()
    return model

model = load_model()

label = st.selectbox("生成する数字ラベル (0〜9)", list(range(10)))

if st.button("画像を生成"):
    label_tensor = torch.tensor([label], dtype=torch.long).to(device)
    z = torch.randn(1, 3).to(device)

    with torch.no_grad():
        generated = model.decoder(z, label_tensor).cpu().numpy()

    image_array = generated[0][0]  # (1, 28, 28)

    # Matplotlibで表示
    fig, ax = plt.subplots()
    ax.imshow(image_array, cmap="gray")
    ax.axis("off")
    st.pyplot(fig)
