import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch

def get_cnn_contrastive():
    return CNNContrastive()

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

class CNNContrastive(nn.Module):
    def __init__(self, num_classes=10, contrastive_dimension=64):
        super(CNNContrastive, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)  # 28x28 -> 28x28
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)  # 28x28 -> 28x28
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)  # Downsample by 2x (14x14 -> 7x7 after 2 pools)
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(64 * 7 * 7, 64) 
        self.contrastive_hidden_layer = nn.Linear(64, contrastive_dimension)
        self.contrastive_output_layer = nn.Linear(contrastive_dimension, contrastive_dimension)


        self.dec1 = nn.Linear(64, 512)
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