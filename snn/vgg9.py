from __future__ import annotations

from typing import Iterable

import torch
from torch import nn

from .neurons import LIFNeuron, SpikeConv2d, SpikeOutputLayer


class SNNVGG9(nn.Module):
	"""VGG9的SNN版本
	
	VGG9是一个简化的VGG网络，包含9个卷积层。
	适用于CIFAR-10等小尺寸图像数据集。
	对于MNIST等1通道图像，需要调整输入通道数。
	
	结构：
	- Conv Block 1: 64 channels, 2 layers
	- Conv Block 2: 128 channels, 2 layers  
	- Conv Block 3: 256 channels, 2 layers
	- Global Average Pooling
	- Output Layer
	"""
	
	def __init__(
		self,
		num_classes: int = 10,
		in_channels: int = 3,
		norm_layer: type[nn.Module] | None = None,
	):
		super().__init__()
		if norm_layer is None:
			norm_layer = nn.BatchNorm2d
		
		# Conv Block 1: 64 channels
		self.conv_block1 = nn.Sequential(
			SpikeConv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False),
			norm_layer(64),
			LIFNeuron(),
			SpikeConv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
			norm_layer(64),
			LIFNeuron(),
		)
		self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
		
		# Conv Block 2: 128 channels
		self.conv_block2 = nn.Sequential(
			SpikeConv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False),
			norm_layer(128),
			LIFNeuron(),
			SpikeConv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
			norm_layer(128),
			LIFNeuron(),
		)
		self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
		
		# Conv Block 3: 256 channels
		self.conv_block3 = nn.Sequential(
			SpikeConv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=False),
			norm_layer(256),
			LIFNeuron(),
			SpikeConv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
			norm_layer(256),
			LIFNeuron(),
		)
		self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
		
		# 全局平均池化和输出层
		self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
		self.fc = SpikeOutputLayer(256, num_classes)
	
	def reset_state(self) -> None:
		"""重置所有LIF神经元和输出层的状态"""
		for module in self.modules():
			if isinstance(module, (LIFNeuron, SpikeOutputLayer)):
				module.reset_state()
	
	def forward_step(self, x_t: torch.Tensor) -> torch.Tensor:
		"""单时间步前向传播
		
		Args:
			x_t: [B, C, H, W] - 脉冲输入
			     对于CIFAR-10: [B, 3, 32, 32]
			     对于MNIST: [B, 1, 28, 28]
		
		Returns:
			logits: [B, num_classes]
		"""
		# Conv Block 1
		x = self.conv_block1(x_t)
		x = self.pool1(x)
		
		# Conv Block 2
		x = self.conv_block2(x)
		x = self.pool2(x)
		
		# Conv Block 3
		x = self.conv_block3(x)
		x = self.pool3(x)
		
		# 全局平均池化
		x = self.avgpool(x)
		x = x.flatten(1)  # [B, 256]
		
		# 输出层
		logits = self.fc(x)
		return logits
	
	def forward_sequence(self, spike_sequence: Iterable[torch.Tensor]) -> torch.Tensor:
		"""处理时间序列的脉冲输入
		
		Args:
			spike_sequence: 时间序列的脉冲输入（生成器或可迭代对象）
		
		Returns:
			logits: [B, num_classes]
		"""
		self.reset_state()
		T = len(spike_sequence)

		# 生成时间维度的随机排列索引
		perm = torch.randperm(T)

		# 根据该随机排列重新索引时间维度
		spike_sequence = spike_sequence[perm, :, :]
		for x_t in spike_sequence:
			logits_t = self.forward_step(x_t)
		# SpikeOutputLayer 已经累积了所有时间步的输出
		return logits_t

