import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Independent
import torch.nn.init as init

class TaskIrrelevantDecoder(nn.Module):
    def __init__(self, input_dim, output_dim, y_dim, num_classes):
        super(TaskIrrelevantDecoder, self).__init__()
        # Initialize a learnable dictionary of class embeddings
        self.class_embeddings = nn.Parameter(torch.randn(num_classes, y_dim))
        
        # Fully connected block
        self.fc = nn.Sequential(
            nn.Linear(input_dim + y_dim, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, output_dim),
            nn.Sigmoid()
        )
        
    def forward(self, z, y=None):
        combined = torch.cat((z, y), dim=1)
        return self.fc(combined)

def get_spectral_norm(layer):
    return nn.utils.spectral_norm(layer)
def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, a=0.2)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
class Discriminator(nn.Module):
    def __init__(self, z_s_dim, z_t_dim, hidden_dim=256):
        super(Discriminator, self).__init__()
        self.fc_z_s = get_spectral_norm(nn.Linear(in_features=z_s_dim, out_features=128, bias=True))
        self.fc_z_t = get_spectral_norm(nn.Linear(in_features=z_t_dim, out_features=128, bias=True))
        
        self.fc_blocks = nn.Sequential(
            # Layer 1
            get_spectral_norm(nn.Linear(in_features=256, out_features=hidden_dim, bias=True)),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            # Layer 2
            get_spectral_norm(nn.Linear(in_features=hidden_dim, out_features=hidden_dim, bias=True)),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            # Layer 3
            get_spectral_norm(nn.Linear(in_features=hidden_dim, out_features=1, bias=True)),
            nn.Sigmoid()  # Output a probability
        )
        
        # 2. Initialize weights
        self.apply(init_weights)

    def _call_method(self, z_s, z_t):
        z_s_emb = self.fc_z_s(z_s)
        z_t_emb = self.fc_z_t(z_t)
        combined = torch.cat([z_s_emb, z_t_emb], dim=1)
        return self.fc_blocks(combined)

    def forward(self, z_s, z_t, mode='orig'):
        assert mode in ['orig', 'perm']
        
        if mode == 'orig':
            return self._call_method(z_s, z_t)
        else:  # mode == 'perm'
            # Permuting z_s and z_t embeddings across the batch
            z_s_permed = z_s[torch.randperm(z_s.size(0)).to(z_s.device)]
            z_t_permed = z_t[torch.randperm(z_t.size(0)).to(z_t.device)]
            return self._call_method(z_s, z_t_permed)

class TaskIrrelevantEncoder(nn.Module):
    def __init__(self, input_channels, latent_dim):
        super(TaskIrrelevantEncoder, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1)  # Conv layer with 32 filters
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)  # Conv layer with 64 filters
        self.fc_mu = nn.Linear(64 * 14 * 14, latent_dim)  # Fully connected layer for mean
        self.fc_sigma = nn.Linear(64 * 14 * 14, latent_dim)  # Fully connected layer for std dev

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)  # Flatten the output from conv layers
        z2_mu = self.fc_mu(x)
        z2_sigma = F.softplus(self.fc_sigma(x)) + 1e-7  # Ensure non-zero variance
        z = self.reparameterize(z2_mu, z2_sigma)
        return z

    def reparameterize(self, mu, sigma):
        dis = Independent(Normal(loc=mu, scale=sigma), 1)
        return dis.rsample()

def _weights_init(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)

def add_awgn_noise(z, snr_db):
    # Calculate the signal power
    signal_power = torch.mean(torch.abs(z**2))
    
    # Calculate noise power based on the SNR
    sigma2 = signal_power * 10**(-snr_db/10)
    
    # Generate real noise using torch.randn
    noise = torch.sqrt(sigma2) * torch.randn(z.size(), dtype=z.dtype, device=z.device)
    
    # Return the signal plus the noise (real-valued)
    return z + noise

class TaskRelevantContrastiveEncoder(nn.Module):
    def __init__(self, num_classes=10, contrastive_dimension=64):
        super(TaskRelevantContrastiveEncoder, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)  # 28x28 -> 28x28
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)  # 28x28 -> 28x28
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)  # Downsample by 2x (14x14 -> 7x7 after 2 pools)
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(64 * 7 * 7, 64) 
        self.contrastive_hidden_layer = nn.Linear(64, contrastive_dimension)
        self.contrastive_output_layer = nn.Linear(contrastive_dimension, contrastive_dimension)


        self.dec1 = nn.Linear(contrastive_dimension, 512)
        self.dec2 = nn.Linear(512, 256)
        self.dec3 = nn.Linear(256, num_classes)

        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def freeze_projection(self):
        self.conv1.requires_grad_(False)
        self.conv2.requires_grad_(False)
        self.fc.requires_grad_(False)

    def _forward_impl_encoder(self, x, snr_db, train):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)  # Flatten feature maps
        x = self.fc(x)
        x = F.normalize(x, dim=1)
        x = add_awgn_noise(x, snr_db) #add noise after normalization important for low snr

        return x

    def forward_constrative(self, x, snr_db):
        # Implement from the encoder E to the projection network P
        z = self._forward_impl_encoder(x, snr_db, True)

        x = self.contrastive_hidden_layer(z)
        x = F.relu(x)
        x = self.contrastive_output_layer(x)

        # Normalize to unit hypersphere
        x = F.normalize(x, dim=1)

        return x, z

    def forward(self, x, snr_db ):
        # Implement from the encoder to the decoder network
        x = self._forward_impl_encoder(x, snr_db, False)
        x = torch.relu(self.dec1(x))
        x = torch.relu(self.dec2(x))
        return self.dec3(x)