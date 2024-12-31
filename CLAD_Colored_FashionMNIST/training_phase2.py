import os
import torch
from torch import optim
import torch.nn.functional as F
from utils import adjust_learning_rate, progress_bar
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn

def train_contrastive(model, encoder2, discriminator, train_loader, criterion, optimizer, writer, args):
    model.train()
    best_loss = float("inf")
    optimizer_phi = optim.Adam(discriminator.parameters(), lr=1e-4)
    for epoch in range(args.n_epochs_contrastive):
        print("Epoch [%d/%d]" % (epoch + 1, args.n_epochs_contrastive))

        train_loss = 0
        total_phi_loss = 0
        total_discriminator_correct = 0
        total_discriminator_samples = 0
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs = torch.cat(inputs)
            targets = targets.repeat(2)

            inputs, targets = inputs.to(args.device), targets.to(args.device)
            optimizer.zero_grad()

            projections, z1 = model.forward_constrative(inputs, args.snr_db)
            loss = criterion(projections, targets)

            with torch.no_grad():
                z2 = encoder2(inputs)

            adversarial_loss_fn = nn.BCELoss()
            validity_real = discriminator(z2, z1, 'orig')
            adversarial_loss_real = adversarial_loss_fn(validity_real, torch.ones_like(validity_real))
            loss -=1*adversarial_loss_real
            loss.backward()
            optimizer.step()


            optimizer_phi.zero_grad()

            # Discriminator loss (log(1 - W(s_i, t_i)) + log W(t_pi(i), s_pi(i)))
            #if epoch%epoch==0:
            validity_fake = discriminator(z2.detach(), z1.detach(), 'perm')
            adversarial_loss_fake = adversarial_loss_fn(validity_fake, torch.zeros_like(validity_fake))

            validity_real = discriminator(z2.detach(), z1.detach(), 'orig')  
            adversarial_loss_real = adversarial_loss_fn(validity_real, torch.ones_like(validity_real))

            loss_phi = (adversarial_loss_real + adversarial_loss_fake)

            loss_phi.backward()  
            optimizer_phi.step()
            total_phi_loss += loss_phi.item()

            # Calculate Discriminator Accuracy
            discriminator_predictions = torch.cat([validity_real, validity_fake], dim=0)
            discriminator_true_labels = torch.cat([torch.ones_like(validity_real), torch.zeros_like(validity_fake)], dim=0)
            discriminator_correct = ((discriminator_predictions > 0.5) == discriminator_true_labels).float().sum()
            total_discriminator_correct += discriminator_correct.item()
            total_discriminator_samples += discriminator_true_labels.size(0)

            train_loss += loss.item()
            writer.add_scalar(
                "Loss train | Supervised Contrastive",
                loss.item(),
                epoch * len(train_loader) + batch_idx,
            )
            progress_bar(
                batch_idx,
                len(train_loader),
                "Loss: %.3f " % (train_loss / (batch_idx + 1)),
            )

        avg_loss = train_loss / (batch_idx + 1)
        if epoch % 10 == 0:
            if (train_loss / (batch_idx + 1)) < best_loss:
                print("Saving..")
                state = {
                    "net": model.state_dict(),
                    "avg_loss": avg_loss,
                    "epoch": epoch,
                }
                if not os.path.isdir("checkpoint"):
                    os.mkdir("checkpoint")
                torch.save(state, f"./checkpoint/ckpt_contrastive_{args.snr_db}.pth")
                best_loss = avg_loss
                # Calculate average discriminator accuracy
        discriminator_accuracy = total_discriminator_correct / total_discriminator_samples
        print(discriminator_accuracy)
        adjust_learning_rate(optimizer, epoch, mode="contrastive", args=args)
        adjust_learning_rate(optimizer_phi, epoch, mode="contrastive", args=args)
