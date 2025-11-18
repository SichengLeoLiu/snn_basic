from __future__ import annotations

from pathlib import Path
from typing import Tuple

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_mnist_dataloaders(
	data_dir: str | Path = "./data",
	batch_size: int = 128,
	workers: int = 4,
) -> Tuple[DataLoader, DataLoader]:
	transform = transforms.Compose([
		transforms.ToTensor(),
		# MNIST 为 0-1，ToTensor 已归一化到 [0,1]
	])
	train_set = datasets.MNIST(root=str(data_dir), train=True, download=True, transform=transform)
	test_set = datasets.MNIST(root=str(data_dir), train=False, download=True, transform=transform)

	train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)
	test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True)
	return train_loader, test_loader


def get_cifar10_dataloaders(
	data_dir: str | Path = "./data",
	batch_size: int = 128,
	workers: int = 4,
) -> Tuple[DataLoader, DataLoader]:
	transform = transforms.Compose([
		transforms.ToTensor(),
		# CIFAR-10 为 0-1，ToTensor 已归一化到 [0,1]
	])
	train_set = datasets.CIFAR10(root=str(data_dir), train=True, download=True, transform=transform)
	test_set = datasets.CIFAR10(root=str(data_dir), train=False, download=True, transform=transform)

	train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)
	test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True)
	return train_loader, test_loader


