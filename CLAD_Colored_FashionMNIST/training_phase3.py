import os
import torch
from utils import adjust_learning_rate

def train_cross_entropy(model, train_loader, test_loader, criterion, optimizer, writer, args):
    for epoch in range(args.n_epochs_cross_entropy):
        print(f"Epoch [{epoch + 1}/{args.n_epochs_cross_entropy}]")

        model.train()
        train_loss = 0
        correct = 0
        total = 0

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(args.device), targets.to(args.device)
            optimizer.zero_grad()
            outputs = model(inputs, args.snr_db)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()
            total += targets.size(0)

            writer.add_scalar("Loss train | Cross Entropy", loss.item(), epoch * len(train_loader) + batch_idx)

        print(f"Train Accuracy: {100.0 * correct / total:.2f}%")
        validation(epoch, model, test_loader, criterion, writer, args, args.snr_db)
        adjust_learning_rate(optimizer, epoch, mode="cross_entropy", args=args)

def validation(epoch, model, test_loader, criterion, writer, args, snr_db):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(args.device), targets.to(args.device)
            outputs = model(inputs, snr_db)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()
            total += targets.size(0)

    acc = 100.0 * correct / total
    print(f"Validation Accuracy: {acc:.2f}%")
    writer.add_scalar("Accuracy validation | Cross Entropy", acc, epoch)
