from __future__ import annotations

import torch
from torch import nn


class _SurrogateHeaviside(torch.autograd.Function):
	"""Heaviside 近似的替代梯度：前向为阶跃，反向为三角形近似。"""

	@staticmethod
	def forward(ctx, membrane_minus_threshold: torch.Tensor, alpha: float) -> torch.Tensor:
		ctx.save_for_backward(membrane_minus_threshold)
		ctx.alpha = alpha
		return (membrane_minus_threshold >= 0).to(membrane_minus_threshold.dtype)

	@staticmethod
	def backward(ctx, grad_output: torch.Tensor):
		(m_minus_th,) = ctx.saved_tensors
		alpha = ctx.alpha
		# 三角形窗口：|x| < 1 时导数为 alpha*(1-|x|)，否则为 0
		slope_region = (1.0 - m_minus_th.abs()).clamp(min=0.0)
		grad_input = grad_output * (alpha * slope_region)
		# 第二个输入 alpha 无梯度
		return grad_input, None


def surrogate_heaviside(x: torch.Tensor, alpha: float = 2.0) -> torch.Tensor:
	return _SurrogateHeaviside.apply(x, alpha)


class LIFNeuron(nn.Module):
	"""简化的 LIF 神经元（无泄露电导），带替代梯度。状态在时间维度上累积。

	公式（离散时间，步长 dt）：
	  v_t = v_{t-1} + (-(v_{t-1}-v_reset) + I_t) * (dt / tau)
	  s_t = H(v_t - v_th)
	  v_t = v_t - s_t * v_th   (soft reset)
	"""

	def __init__(
		self,
		tau: float = 2.0,
		v_threshold: float = 1.0,
		v_reset: float = 0.0,
		dt: float = 1.0,
		surrogate_alpha: float = 2.0,
	):
		super().__init__()
		self.tau = float(tau)
		self.v_threshold = float(v_threshold)
		self.v_reset = float(v_reset)
		self.dt = float(dt)
		self.surrogate_alpha = float(surrogate_alpha)

		self.register_buffer("_v", None, persistent=False)

	@property
	def v(self) -> torch.Tensor | None:
		return self._v

	def reset_state(self) -> None:
		self._v = None

	def forward(self, input_current: torch.Tensor) -> torch.Tensor:
		if self._v is None:
			self._v = torch.zeros_like(input_current)
		elif self._v.shape != input_current.shape:
			self._v = torch.zeros_like(input_current)
		# 设备检查可以去掉，因为通常不会变化

		alpha = self.dt / self.tau
		self._v = self._v + (-(self._v - self.v_reset) + input_current) * alpha
		m_minus_th = self._v - self.v_threshold
		spike = surrogate_heaviside(m_minus_th, alpha=self.surrogate_alpha)
		# soft reset
		self._v = self._v - spike * self.v_threshold
		return spike


class SpikeLinear(nn.Module):
	"""脉冲线性层：处理脉冲输入，输出加权电流。
	
	这是纯SNN中的基本连接层，将脉冲输入转换为加权电流输出。
	"""
	
	def __init__(self, in_features: int, out_features: int, bias: bool = True):
		super().__init__()
		self.in_features = in_features
		self.out_features = out_features
		self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.1)
		if bias:
			self.bias = nn.Parameter(torch.zeros(out_features))
		else:
			self.register_parameter('bias', None)
	
	def forward(self, spikes: torch.Tensor) -> torch.Tensor:
		# 脉冲输入 [B, in_features] -> 加权电流 [B, out_features]
		# 使用矩阵乘法：current = spikes @ weight.T + bias
		current = torch.matmul(spikes, self.weight.t())
		if self.bias is not None:
			current = current + self.bias
		return current


# class SpikeOutputLayer(nn.Module):
# 	"""脉冲输出层：将脉冲转换为分类logits。
	
# 	通过时间积分将脉冲序列转换为分类概率。
# 	"""
	
# 	def __init__(self, in_features: int, num_classes: int):
# 		super().__init__()
# 		self.in_features = in_features
# 		self.num_classes = num_classes
# 		self.weight = nn.Parameter(torch.randn(num_classes, in_features) * 0.1)
# 		self.bias = nn.Parameter(torch.zeros(num_classes))
		
# 		# 用于累积输出
# 		self.register_buffer("_accumulated_output", None, persistent=False)
	
# 	def reset_state(self) -> None:
# 		self._accumulated_output = None
	
# 	def forward(self, spikes: torch.Tensor) -> torch.Tensor:
# 		# 脉冲输入 [B, in_features] -> 当前时间步的logits [B, num_classes]
# 		current_logits = torch.matmul(spikes, self.weight.t()) + self.bias
		
# 		# 累积输出（用于时间积分）
# 		if self._accumulated_output is None:
# 			self._accumulated_output = current_logits
# 		else:
# 			self._accumulated_output = self._accumulated_output + current_logits
		
# 		return self._accumulated_output


class SpikeOutputLayer(nn.Module):
	"""脉冲输出层：将脉冲转换为分类logits。
	
	通过时间积分将脉冲序列转换为分类概率。
	"""
	
	def __init__(self, in_features: int, num_classes: int):
		super().__init__()
		self.in_features = in_features
		self.num_classes = num_classes
		self.weight = nn.Parameter(torch.randn(num_classes, in_features) * 0.1)
		self.bias = nn.Parameter(torch.zeros(num_classes))
		
		# 用于累积输出
		self.register_buffer("_accumulated_output", None, persistent=False)
	
	def reset_state(self) -> None:
		self._accumulated_output = None
	
	def forward(self, spikes: torch.Tensor) -> torch.Tensor:
		# 脉冲输入 [B, in_features] -> 当前时间步的logits [B, num_classes]
		current_logits = torch.matmul(spikes, self.weight.t()) + self.bias
		
		# 累积输出（用于时间积分）
		if self._accumulated_output is None:
			self._accumulated_output = current_logits
		else:
			self._accumulated_output = self._accumulated_output + current_logits
			# pass
		
		return self._accumulated_output


class SpikeConv2d(nn.Module):
	"""脉冲卷积层：处理脉冲输入，输出加权电流。
	
	这是纯SNN中的卷积层，将脉冲输入转换为加权电流输出。
	"""
	
	def __init__(
		self,
		in_channels: int,
		out_channels: int,
		kernel_size: int | tuple[int, int],
		stride: int | tuple[int, int] = 1,
		padding: int | tuple[int, int] | str = 0,
		bias: bool = True,
	):
		super().__init__()
		self.in_channels = in_channels
		self.out_channels = out_channels
		self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
		self.stride = stride if isinstance(stride, tuple) else (stride, stride)
		self.padding = padding
		
		self.weight = nn.Parameter(
			torch.randn(out_channels, in_channels, *self.kernel_size) * 0.1
		)
		if bias:
			self.bias = nn.Parameter(torch.zeros(out_channels))
		else:
			self.register_parameter('bias', None)
	
	def forward(self, spikes: torch.Tensor) -> torch.Tensor:
		# 脉冲输入 [B, C, H, W] -> 加权电流 [B, C_out, H_out, W_out]
		current = nn.functional.conv2d(
			spikes, self.weight, self.bias, self.stride, self.padding
		)
		return current


class SpikeBatchNorm2d(nn.Module):
	"""脉冲批归一化层：对卷积输出进行归一化。
	
	在SNN中，归一化通常在LIF神经元之前完成。
	这里实现一个简化版本，使用running统计量。
	"""
	
	def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1):
		super().__init__()
		self.num_features = num_features
		self.eps = eps
		self.momentum = momentum
		
		self.register_buffer('running_mean', torch.zeros(num_features))
		self.register_buffer('running_var', torch.ones(num_features))
		self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
	
	def forward(self, x: torch.Tensor) -> torch.Tensor:
		if self.training:
			# 计算当前batch的统计量
			mean = x.mean(dim=[0, 2, 3], keepdim=True)
			var = x.var(dim=[0, 2, 3], keepdim=True, unbiased=False)
			# 更新running统计量
			self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean.squeeze()
			self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var.squeeze()
			self.num_batches_tracked += 1
			# 归一化
			x = (x - mean) / torch.sqrt(var + self.eps)
		else:
			# 推理时使用running统计量
			mean = self.running_mean.view(1, -1, 1, 1)
			var = self.running_var.view(1, -1, 1, 1)
			x = (x - mean) / torch.sqrt(var + self.eps)
		return x