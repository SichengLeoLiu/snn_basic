from __future__ import annotations

from typing import Iterable

import torch
from torch import nn

from .neurons import LIFNeuron, SpikeConv2d, SpikeOutputLayer
# 不再导入 SpikeBatchNorm2d，直接使用 nn.BatchNorm2d


class BasicBlock(nn.Module):
	"""ResNet基础残差块（SNN版本）"""
	expansion = 1
	
	def __init__(
		self,
		in_channels: int,
		out_channels: int,
		stride: int = 1,
		norm_layer: type[nn.Module] | None = None,
	):
		super().__init__()
		if norm_layer is None:
			norm_layer = nn.BatchNorm2d  # 直接使用 nn.BatchNorm2d
		
		# 主路径
		self.conv1 = SpikeConv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
		self.bn1 = norm_layer(out_channels)
		self.lif1 = LIFNeuron()
		
		self.conv2 = SpikeConv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
		self.bn2 = norm_layer(out_channels)
		
		# 残差连接
		self.shortcut = nn.Sequential()
		if stride != 1 or in_channels != out_channels:
			self.shortcut = nn.Sequential(
				SpikeConv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
				norm_layer(out_channels),
			)
		
		# 残差连接后的LIF神经元
		self.lif2 = LIFNeuron()
	
	def reset_state(self) -> None:
		self.lif1.reset_state()
		self.lif2.reset_state()
	
	def forward(self, x: torch.Tensor) -> torch.Tensor:
		# 主路径
		out = self.conv1(x)
		out = self.bn1(out)
		out = self.lif1(out)
		
		out = self.conv2(out)
		out = self.bn2(out)
		
		# 残差连接
		identity = self.shortcut(x)
		out = out + identity
		
		# 残差连接后通过LIF神经元
		out = self.lif2(out)
		
		return out


class SNNResNet18(nn.Module):
	"""ResNet18的SNN版本
	
	适用于CIFAR-10等3通道图像数据集。
	对于MNIST等1通道图像，需要调整输入通道数。
	"""
	
	def __init__(
		self,
		num_classes: int = 10,
		in_channels: int = 3,
		norm_layer: type[nn.Module] | None = None,
	):
		super().__init__()
		if norm_layer is None:
			norm_layer = nn.BatchNorm2d  # 直接使用 nn.BatchNorm2d
		
		# 初始卷积层（CIFAR-10使用较小的kernel和stride）
		self.conv1 = SpikeConv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
		self.bn1 = norm_layer(64)
		self.lif1 = LIFNeuron()
		
		# ResNet层
		self.layer1 = self._make_layer(BasicBlock, 64, 64, 2, stride=1, norm_layer=norm_layer)
		self.layer2 = self._make_layer(BasicBlock, 64, 128, 2, stride=2, norm_layer=norm_layer)
		self.layer3 = self._make_layer(BasicBlock, 128, 256, 2, stride=2, norm_layer=norm_layer)
		self.layer4 = self._make_layer(BasicBlock, 256, 512, 2, stride=2, norm_layer=norm_layer)
		
		# 全局平均池化和输出层
		self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
		self.fc = SpikeOutputLayer(512, num_classes)
	
	def _make_layer(
		self,
		block: type[BasicBlock],
		in_channels: int,
		out_channels: int,
		blocks: int,
		stride: int,
		norm_layer: type[nn.Module] | None,
	) -> nn.Sequential:
		layers = []
		layers.append(block(in_channels, out_channels, stride, norm_layer))
		for _ in range(1, blocks):
			layers.append(block(out_channels, out_channels, 1, norm_layer))
		return nn.Sequential(*layers)
	
	def reset_state(self) -> None:
		self.lif1.reset_state()
		for layer in [self.layer1, self.layer2, self.layer3, self.layer4]:
			for block in layer:
				if isinstance(block, BasicBlock):
					block.reset_state()
		self.fc.reset_state()
	
	def forward_step(self, x_t: torch.Tensor) -> torch.Tensor:
		# x_t: [B, C, H, W] - 脉冲输入
		# 对于CIFAR-10: [B, 3, 32, 32]
		
		# 初始层
		x = self.conv1(x_t)
		x = self.bn1(x)
		x = self.lif1(x)
		
		# ResNet层
		x = self.layer1(x)
		x = self.layer2(x)
		x = self.layer3(x)
		x = self.layer4(x)
		
		# 全局平均池化
		x = self.avgpool(x)
		x = x.flatten(1)  # [B, 512]
		
		# 输出层
		logits = self.fc(x)
		return logits
	
	def forward_sequence(self, spike_sequence: Iterable[torch.Tensor]) -> torch.Tensor:
		"""处理时间序列的脉冲输入"""
		self.reset_state()
		for x_t in spike_sequence:
			logits_t = self.forward_step(x_t)
		# SpikeOutputLayer 已经累积了所有时间步的输出
		return logits_t

