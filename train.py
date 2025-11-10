from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import torch
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from data import get_mnist_dataloaders
from snn import SNNMLP, poisson_encode, static_encode


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
	parser.add_argument("--data_dir", type=str, default="./data")
	parser.add_argument("--batch_size", type=int, default=128)
	parser.add_argument("--epochs", type=int, default=10)
	parser.add_argument("--steps", type=int, default=4, help="时间步数（脉冲展开步数）")
	parser.add_argument("--lr", type=float, default=1e-3)
	parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
	parser.add_argument("--save_dir", type=str, default="./checkpoints")
	args = parser.parse_args()

	device = torch.device(args.device)
	Path(args.save_dir).mkdir(parents=True, exist_ok=True)
	train_loader, test_loader = get_mnist_dataloaders(args.data_dir, args.batch_size)

	model = SNNMLP()
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
			ckpt = Path(args.save_dir) / "best_snn_mnist.pt"
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


