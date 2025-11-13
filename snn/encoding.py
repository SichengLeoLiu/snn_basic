from __future__ import annotations

import torch


def poisson_encode(images: torch.Tensor, num_steps: int, generator: torch.Generator | None = None):
	"""泊松编码：按像素强度（0-1）在每个时间步采样伯努利脉冲。

	返回一个长度为 num_steps 的生成器，每次产生与 images 同形状的 {0,1} 脉冲张量。
	"""
	device = images.device
	for _ in range(num_steps):
		if generator is not None:
			r = torch.rand(images.shape, generator=generator, device=device, dtype=images.dtype)
		else:
			r = torch.rand_like(images)
		yield (r <= images).to(images.dtype)

def static_encode(images: torch.Tensor, num_steps: int, generator: torch.Generator | None = None):
	"""静态编码：所有时间步都返回相同的脉冲（即 images > 0 的位置为 1，其余为 0）"""

	return images.unsqueeze(0).expand(num_steps, *images.shape)
	# for _ in range(num_steps):
	# 	yield images
