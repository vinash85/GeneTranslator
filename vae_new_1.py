import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr
from sklearn.manifold import TSNE



class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(VAE, self).__init__()

        # dimensions for the intermediate layers
        dims = [input_dim,12000 ,10000, 8000, 4000, 2000,latent_dim, latent_dim]

        # Encoder layers with BatchNorm
        encoder_layers = []
        for i in range(len(dims)-1):
            encoder_layers.append(nn.Linear(dims[i], dims[i+1]))
            encoder_layers.append(nn.BatchNorm1d(dims[i+1]))
            #encoder_layers.append(nn.LeakyReLU(0.5))
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
            #decoder_layers.append(nn.LeakyReLU(0.5))
            decoder_layers.append(nn.ReLU())
            #print(len(decoder_layers))
        #decoder_layers.append(nn.Sigmoid())
        decoder_layers.append(nn.Linear(input_dim,input_dim))
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
    return 0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())


def pearson_correlation(x, y):
    vx = x - np.mean(x)
    vy = y - np.mean(y)
    return np.sum(vx * vy) / (np.sqrt(np.sum(vx ** 2)) * np.sqrt(np.sum(vy ** 2)))






filename = '/mnt/data/macaulay/datas/processed_OmicsExpression.csv'
df = pd.read_csv(filename)
cell_line_names = df['Cell_line'].copy()  # Save the 'Cell_line' column
df = df.drop(columns=['Cell_line'])

numeric_data = df.values
scaler = MinMaxScaler()
#scaler = StandardScaler()
scaled_data = scaler.fit_transform(numeric_data)
scaled_data = scaled_data*5
train_tensor = torch.tensor(scaled_data, dtype=torch.float32)
print("df shape:", df.shape)
print('training tensor data shape:', train_tensor.shape)
device = ("cuda" if torch.cuda.is_available() else "cpu")
# device = ( "cpu")
print('Using: ', device)
# val_tensor = train_tensor[900:,:]
# train_tensor = train_tensor[0:900,:]

#train_tensor = train_tensor.to(device)

#dummy=train_tensor[0:2,:].numpy()
#sample_coor=pearson_correlation(dummy,dummy)

#print(sample_coor)
#exit()
input_dim = train_tensor.shape[1]
learning_rate = 0.00001
batch_size = 3
num_epochs = 20

latent_dim = 1900


print("loading model")
vae = VAE(input_dim, latent_dim)
vae = vae.to(device)
print("loading completed")
reconstruction_loss = nn.MSELoss()
optimizer = optim.Adam(vae.parameters(), lr=learning_rate)
dataloader = create_dataloader(train_tensor, batch_size)
prev_correlation = -1.0
print("starting training")
for epoch in range(num_epochs):
    embeddings_list = []
    vae.train()
    running_recon_loss = 0.0
    running_kl_loss = 0.0
    reconstructed_data = []

    for batch_data in dataloader:
        optimizer.zero_grad()
        batch_data = batch_data[0].to(device)

        # Forward pass
        output, mu, logvar = vae(batch_data)
        reconstructed_data.append(output.detach())


        recon_loss = reconstruction_loss(output, batch_data)
        kl_loss = kl_divergence(mu, logvar)

        # Combined loss
        combined_loss = recon_loss

        # Backward pass and optimization
        combined_loss.backward()
        optimizer.step()
        #del batch_data
        #torch.empty_cache()

        # Accumulate running losses
        running_recon_loss += recon_loss.item()
        running_kl_loss += kl_loss.item()
        embeddings = mu.detach().cpu().numpy()
        embeddings_list.append(embeddings)


    # After all batches are processed, compute correlation over the entire dataset
    reconstructed_data = torch.cat(reconstructed_data, dim=0)
    reconstructed_data =reconstructed_data.cpu()
    reconstructed_data = reconstructed_data.numpy()
    epoch_correlation = pearson_correlation(reconstructed_data, train_tensor.numpy())
    # Print epoch statistics
    epoch_recon_loss = running_recon_loss / len(dataloader)
    epoch_kl_loss = running_kl_loss / len(dataloader)
    epoch_combined_loss = epoch_recon_loss + epoch_kl_loss

    if epoch%75==0 and epoch!=0 and learning_rate > 0.99e-7:
        learning_rate= learning_rate *0.1

    if epoch_correlation > prev_correlation:
        torch.save(vae.state_dict(), '/mnt/data/macaulay/datas/vae_expression.pth')

        embeddings_array = np.concatenate(embeddings_list)
        df_embeddings = pd.DataFrame(embeddings_array)

        df_embeddings = pd.concat([cell_line_names, df_embeddings], axis=1) #join the cell line names to the embeddings
        df_embeddings.to_csv(f'/mnt/data/macaulay/datas/OmicExpression_embeddings/OmicExpression_embeddings_{epoch}.csv', index=False)
        prev_coorelation = epoch_correlation
    # val_data = val_tensor.to(device)
    # output, mu, logvar = vae(val_data)
    # val_corr = pearson_correlation(output.detach().numpy(),val_tensor.numpy())
    # del val_data
    #torch.empty_cache()

    print("Epoch [{}/{}],learning_rate{}, Combined Training Loss: {:.4f}, Reconstruction Loss: {:.4f}, KL Loss: {:.4f}, Train Correlation: {:.4f}".format(epoch + 1,num_epochs,learning_rate,epoch_combined_loss,epoch_recon_loss, epoch_kl_loss,epoch_correlation))
    with open('/mnt/data/macaulay/datas/vae_metrics_without_five.txt', 'a') as f:
        if epoch == 0:
            f.write('Epoch,Combined Training Loss,Reconstruction Loss,KL Loss,Train Correlation\n')
        f.write(f'{epoch + 1},{epoch_combined_loss},{epoch_recon_loss},{epoch_kl_loss},{epoch_correlation}\n')


plt.plot(num_epochs + 1, epoch_recon_loss, label='Combined Training Loss')
plt.plot(num_epochs + 1, epoch_kl_loss, label='KL Divergence')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Curves')
plt.legend()
plt.savefig('/mnt/data/macaulay/plot_images/vae_loss_curve.png')