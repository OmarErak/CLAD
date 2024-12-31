# Modular Python Project Structure

# 1. Data Loading (data_loader.py)
import numpy as np
import torch
import torchvision
from torch.utils.data import Dataset
import torchvision.transforms as transforms

predefined_bg_colors = {
    "red": np.array([1, 0, 0]),
    "green": np.array([0, 1, 0]),
    "blue": np.array([0, 0, 1]),
    "yellow": np.array([1, 1, 0]),
    "purple": np.array([1.0, 0, 1.0]),
    "light blue": np.array([0, 1, 1]),
    "black": np.array([0, 0, 0]),
}

color_names = list(predefined_bg_colors.keys())
color_values = np.array(list(predefined_bg_colors.values()))

class ColoredFashionMNIST(Dataset):
    def __init__(self, fashion_mnist_data, bg_colors, contrastive=False):
        self.fashion_mnist_data = fashion_mnist_data
        self.bg_colors = bg_colors
        self.contrastive = contrastive

    def __len__(self):
        return len(self.fashion_mnist_data)

    def colorize_image(self, img, idx):
        img = img.squeeze(0)
        bg_color = self.bg_colors[idx]
        img_colored = torch.zeros(3, 28, 28)
        for i in range(3):
            img_colored[i] = img * 1 + (1 - img) * bg_color[i]
        return img_colored

    def __getitem__(self, idx):
        if self.contrastive:
            (img1, img2), label = self.fashion_mnist_data[idx]
            img1_colored = self.colorize_image(img1, idx)
            img2_colored = self.colorize_image(img2, idx)
            return (img1_colored, img2_colored), label
        else:
            img, label = self.fashion_mnist_data[idx]
            img_colored = self.colorize_image(img, idx)
            return img_colored, label

def load_dataset(args):
    transform_train = [
        transforms.RandomCrop(28, padding=4),
        transforms.RandomHorizontalFlip(),
    ]
    if args.auto_augment:
        from data_augmentation.auto_augment import AutoAugment
        transform_train.append(AutoAugment())
    transform_train.extend([
        transforms.ToTensor(),
    ])
    transform_train = transforms.Compose(transform_train)

    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_set = torchvision.datasets.FashionMNIST(root="../data", train=True, download=True, transform=transform_train)
    num_samples = len(train_set)
    bg_color_indices_train = np.random.randint(0, len(predefined_bg_colors), size=num_samples)
    bg_colors_train = color_values[bg_color_indices_train]
    train_dataset = ColoredFashionMNIST(train_set, bg_colors=bg_colors_train, contrastive=False)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers
    )

    test_set = torchvision.datasets.FashionMNIST(root="../data", train=False, download=True, transform=transform_test)
    bg_color_indices_test = np.random.randint(0, len(predefined_bg_colors), size=len(test_set))
    bg_colors_test = color_values[bg_color_indices_test]
    test_dataset = ColoredFashionMNIST(test_set, bg_colors=bg_colors_test, contrastive=False)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )

    num_classes = 10
    return num_classes, train_loader, test_loader, transform_train, bg_colors_train

from data_augmentation.duplicate_sample_transform import DuplicateSampleTransform
from torchvision import datasets
from torch.utils.data import DataLoader

def load_contrastive_dataset(args, transform_train, bg_colors_train):
    train_contrastive_transform = DuplicateSampleTransform(transform_train)
    train_set = datasets.FashionMNIST(root='../data', train=True, download=True, transform=train_contrastive_transform)
    train_set_contrastive = ColoredFashionMNIST(train_set, bg_colors=bg_colors_train, contrastive=True)

    train_loader_contrastive = DataLoader(
        train_set_contrastive,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
    )
    return train_loader_contrastive