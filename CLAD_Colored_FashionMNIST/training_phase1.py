import torch
from torch import optim
import os
from torch import optim
import torch.nn.functional as F
from torch import nn

def add_awgn_noise(z, snr_db):
    signal_power = torch.mean(torch.abs(z**2))
    sigma2 = signal_power * 10**(-snr_db / 10)
    noise = torch.sqrt(sigma2) * torch.randn(z.size(), dtype=z.dtype, device=z.device)
    return z + noise

def train_phase1(encoder2, reconstructor, projection_layer, train_loader, args):
    optimizer_E2 = optim.Adam(encoder2.parameters(), lr=args.lr)
    optimizer_Recon = optim.Adam(reconstructor.parameters(), lr=args.lr)
    criterion_reconstruction = nn.MSELoss()

    for epoch in range(args.epochs_phase1):
        encoder2.train()
        reconstructor.train()
        total_loss = 0

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(args.device), targets.to(args.device)
            inputs_flattened = inputs.view(inputs.size(0), -1)

            optimizer_E2.zero_grad()
            optimizer_Recon.zero_grad()

            z2 = encoder2(inputs)
            z2 = add_awgn_noise(z2, args.snr_db)

            targets_one_hot = F.one_hot(targets, num_classes=10).float().to(args.device)
            targets_projected = F.relu(projection_layer(targets_one_hot))
            reconstructed = reconstructor(z2, targets_projected)

            loss = criterion_reconstruction(reconstructed, inputs_flattened)
            loss.backward()
            optimizer_E2.step()
            optimizer_Recon.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch + 1}/{args.epochs_phase1}], Loss: {avg_loss:.4f}")