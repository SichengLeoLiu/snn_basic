from __future__ import annotations

from typing import Iterable

import torch
from torch import nn

from .neurons import LIFNeuron, SpikeLinear, SpikeOutputLayer


class SNNMLP(nn.Module):
	"""完整的脉冲神经网络：SpikeLinear + LIF x N 层，纯脉冲处理。

	输入期望为 [B, C, H, W] 的归一化图像（0-1），外部进行时域编码后在时间维上逐步送入。
	所有层都处理脉冲信号，实现真正的SNN。
	"""

	def __init__(self, input_dim: int = 28 * 28, hidden_dims: Iterable[int] = (512, 256), num_classes: int = 10):
		super().__init__()
		layers: list[nn.Module] = []
		prev = input_dim
		for h in hidden_dims:
			layers.append(SpikeLinear(prev, h))
			layers.append(LIFNeuron())
			prev = h
		layers.append(SpikeOutputLayer(prev, num_classes))
		self.layers = nn.ModuleList(layers)

	def reset_state(self) -> None:
		for m in self.layers:
			if isinstance(m, (LIFNeuron, SpikeOutputLayer)):
				m.reset_state()

	def forward_step(self, x_t: torch.Tensor) -> torch.Tensor:
		# x_t: [B, C, H, W] 或 [B, input_dim] - 脉冲输入
		# 对于MNIST: [B, 1, 28, 28] 或 [B, 784]
		# 对于CIFAR-10: [B, 3, 32, 32] 或 [B, 3072]
		if x_t.dim() > 2:
			x_t = x_t.flatten(1)
		z = x_t
		for m in self.layers:
			z = m(z)
		return z

	def forward_sequence(self, spike_sequence: Iterable[torch.Tensor]) -> torch.Tensor:
		self.reset_state()
		steps = 0

		# T = spike_sequence.shape[0]

		# # 生成时间维度的随机排列索引
		# perm = torch.randperm(T)

		# # 根据该随机排列重新索引时间维度
		# spike_sequence = spike_sequence[perm, :, :]

  
		for x_t in spike_sequence:
			logits_t = self.forward_step(x_t)
			
			steps += 1
		# SpikeOutputLayer 已经累积了所有时间步的输出
		return logits_t


