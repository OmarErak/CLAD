import argparse
from data_loader import load_dataset, load_contrastive_dataset
from models import TaskIrrelevantEncoder, TaskIrrelevantDecoder, Discriminator, TaskRelevantContrastiveEncoder
from training_phase1 import train_phase1
from training_phase2 import train_contrastive
from training_phase3 import train_cross_entropy
from utils import save_model, load_phase1_model
import torch
import torch.nn as nn
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from loss.spc import SupervisedContrastiveLoss
from torch.backends import cudnn
import os

def main():
    parser = argparse.ArgumentParser(description="Run training phases.")
    parser.add_argument("--phase", type=int, choices=[1, 2, 3], required=True, help="Training phase to execute (1, 2, or 3).")
    parser.add_argument("--epochs_phase1", type=int, default=30, help="Number of epochs for Phase 1 training.")
    parser.add_argument("--n_epochs_contrastive", type=int, default=500, help="Number of epochs for Phase 2 training.")
    parser.add_argument("--n_epochs_cross_entropy", type=int, default=50, help="Number of epochs for Phase 3 training.")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size.")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate.")
    parser.add_argument("--snr_db", type=float, default=12, help="Signal-to-noise ratio for AWGN.")
    parser.add_argument("--cosine", action="store_true", help="Use cosine annealing for learning rate.")
    parser.add_argument("--lr_decay_rate", type=float, default=0.1, help="Learning rate decay rate.")
    parser.add_argument("--lr_decay_epochs", nargs="+", type=int, default=[150, 300, 500], help="Epochs to decay learning rate.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to run on.")
    parser.add_argument("--latent_dim", type=int, default=64, help="Dimension of the latent vector.")
    parser.add_argument("--input_size", type=int, default=28, help="Input image size.")
    parser.add_argument("--dataset", type=str, choices=["coloredmnist", "coloredfmnist", "cifar10"], required=True, help="Dataset to use.")
    parser.add_argument("--auto_augment", type=bool, default=False, help="Whether to apply autoaugment")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for data loading.")
    parser.add_argument("--temperature", type=float, default=0.07, help="Temperature constant for contrastive loss.")
    parser.add_argument("--lr_contrastive", type=float, default=1e-1, help="Learning rate for contrastive step.")
    parser.add_argument("--lr_cross_entropy", type=float, default=5e-2, help="Learning rate for cross-entropy step.")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay for optimizer.")
    parser.add_argument("--momentum", type=float, default=0.9, help="Momentum for SGD optimizer.")
    args = parser.parse_args()

    # Initialize shared components
    num_classes, train_loader, test_loader, transform_train, bg_colors_train = load_dataset(args)
    writer = SummaryWriter("logs")

    if args.phase == 1:
        # Phase 1: Pretraining the encoder and decoder
        encoder = TaskIrrelevantEncoder(3, args.latent_dim).to(args.device)
        decoder = TaskIrrelevantDecoder(args.latent_dim, 3 * args.input_size * args.input_size, args.latent_dim, num_classes).to(args.device)
        projection_layer = nn.Linear(num_classes, args.latent_dim).to(args.device)

        train_phase1(encoder, decoder, projection_layer, train_loader, args)

        # Save the models after Phase 1
        save_model(encoder, "encoder_phase1.pth")
        save_model(decoder, "decoder_phase1.pth")

    elif args.phase == 2:
        model = TaskRelevantContrastiveEncoder()
        model = model.to(args.device)
        discriminator = Discriminator(args.latent_dim, args.latent_dim).to(args.device)
        encoder_path = "encoder_phase1.pth"

        encoder_irrelevant = load_phase1_model(
            encoder_path, TaskIrrelevantEncoder, args
        )

        cudnn.benchmark = True

        if not os.path.isdir("logs"):
            os.makedirs("logs")

        train_loader_contrastive = load_contrastive_dataset(args, transform_train, bg_colors_train)
        optimizer = optim.SGD(
            model.parameters(),
            lr=args.lr_contrastive,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )

        criterion = SupervisedContrastiveLoss(temperature=args.temperature)
        criterion.to(args.device)

        train_contrastive(model, encoder_irrelevant, discriminator, train_loader_contrastive, criterion, optimizer, writer, args)

    elif args.phase == 3:
        # Phase 3: Classification training
        model = TaskRelevantContrastiveEncoder()
        model = model.to(args.device)
        checkpoint = torch.load("./checkpoint/ckpt_contrastive_12.pth")
        model.load_state_dict(checkpoint["net"])

        model.freeze_projection()
        optimizer = optim.SGD(
            model.parameters(),
            lr=args.lr_cross_entropy,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )

        criterion = nn.CrossEntropyLoss()
        criterion.to(args.device)

        args.best_acc = 0.0
        train_cross_entropy(model, train_loader, test_loader, criterion, optimizer, writer, args)

        print("Training phase complete.")

if __name__ == "__main__":
    main()
