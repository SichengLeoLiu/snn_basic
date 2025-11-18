from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import torch
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from data import get_mnist_dataloaders, get_cifar10_dataloaders
from snn import SNNMLP, SNNResNet18, SNNVGG9, poisson_encode, static_encode


def accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
	pred = logits.argmax(dim=1)
	return (pred == targets).float().mean().item()


def run_epoch(
	model: nn.Module,
	loader,
	device: torch.device,
	optimizer: torch.optim.Optimizer | None,
	steps: int,
	train: bool,
	progress_desc: str,
) -> tuple[float, float]:
	model.train(mode=train)
	criterion = nn.CrossEntropyLoss()
	total_loss = 0.0
	total_acc = 0.0
	count = 0
	for images, labels in tqdm(loader, desc=progress_desc, leave=False):
		images = images.to(device)
		labels = labels.to(device)
		if train:
			optimizer.zero_grad(set_to_none=True)
		# 泊松编码时间序列 → 模型 → 平均 logits
		spike_seq = static_encode(images, steps)
		logits = model.forward_sequence(spike_seq)
		loss = criterion(logits, labels)
		if train:
			loss.backward()
			optimizer.step()
		acc = accuracy(logits.detach(), labels)
		batch = images.size(0)
		total_loss += loss.detach().item() * batch
		total_acc += acc * batch
		count += batch
	return total_loss / count, total_acc / count


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--dataset", type=str, default="mnist", choices=["mnist", "cifar10"], help="数据集选择：mnist 或 cifar10")
	parser.add_argument("--data_dir", type=str, default="./data")
	parser.add_argument("--batch_size", type=int, default=128)
	parser.add_argument("--epochs", type=int, default=10)
	parser.add_argument("--steps", type=int, default=4, help="时间步数（脉冲展开步数）")
	parser.add_argument("--lr", type=float, default=1e-3)
	parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
	parser.add_argument("--save_dir", type=str, default="./checkpoints")
	parser.add_argument("--model", type=str, default="mlp", choices=["mlp", "resnet18", "vgg9"], help="模型架构选择：mlp、resnet18 或 vgg9")
	args = parser.parse_args()

	device = torch.device(args.device)
	Path(args.save_dir).mkdir(parents=True, exist_ok=True)
	torch.manual_seed(42)
	# 根据数据集选择加载器
	if args.dataset == "mnist":
		train_loader, test_loader = get_mnist_dataloaders(args.data_dir, args.batch_size)
		input_dim = 28 * 28
		ckpt_name = f"best_snn_{args.model}_mnist.pt"
		if args.model in ["resnet18", "vgg9"]:
			print(f"警告：{args.model.upper()}通常用于CIFAR-10，MNIST建议使用MLP。将使用1通道输入。")
	elif args.dataset == "cifar10":
		train_loader, test_loader = get_cifar10_dataloaders(args.data_dir, args.batch_size)
		input_dim = 32 * 32 * 3
		ckpt_name = f"best_snn_{args.model}_cifar10.pt"
	else:
		raise ValueError(f"不支持的数据集: {args.dataset}")

	# 根据模型类型创建模型
	if args.model == "mlp":
		model = SNNMLP(input_dim=input_dim)
	elif args.model == "resnet18":
		if args.dataset == "mnist":
			# MNIST是1通道，需要调整输入通道数
			model = SNNResNet18(num_classes=10, in_channels=1)
		else:
			# CIFAR-10是3通道
			model = SNNResNet18(num_classes=10, in_channels=3)
	elif args.model == "vgg9":
		if args.dataset == "mnist":
			# MNIST是1通道，需要调整输入通道数
			model = SNNVGG9(num_classes=10, in_channels=1)
		else:
			# CIFAR-10是3通道
			model = SNNVGG9(num_classes=10, in_channels=3)
	else:
		raise ValueError(f"不支持的模型: {args.model}")
	model.to(device)
	optimizer = Adam(model.parameters(), lr=args.lr)
	scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

	best_acc = 0.0
	for epoch in range(1, args.epochs + 1):
		train_loss, train_acc = run_epoch(model, train_loader, device, optimizer, args.steps, True, f"Train {epoch}")
		val_loss, val_acc = run_epoch(model, test_loader, device, None, args.steps, False, f"Eval  {epoch}")
		scheduler.step()
		print(f"Epoch {epoch}: train_loss={train_loss:.4f} acc={train_acc:.4f} | val_loss={val_loss:.4f} acc={val_acc:.4f}")
		if val_acc > best_acc:
			best_acc = val_acc
			ckpt = Path(args.save_dir) / ckpt_name
			torch.save({
				"epoch": epoch,
				"model_state": model.state_dict(),
				"val_acc": val_acc,
				"config": vars(args),
			}, ckpt)
			print(f"Saved best checkpoint to {ckpt} (acc={best_acc:.4f})")

	print(f"Best val acc: {best_acc:.4f}")


if __name__ == "__main__":
	main()


