# 脉冲神经网络（SNN）识别 MNIST 手写数字

本项目实现了使用替代梯度训练的 LIF 脉冲神经网络，在 MNIST 数据集上进行分类。

## 环境与依赖

- Python 3.10+
- PyTorch、Torchvision、tqdm、numpy（见 `requirements.txt`）

安装依赖：

```bash
pip install -r requirements.txt
```

若在 Windows 上使用 CUDA，请提前根据显卡与 CUDA 版本安装对应的 PyTorch 轮子。

## 目录结构

- `snn/neurons.py`: LIF 神经元，替代梯度实现
- `snn/model.py`: 简单前馈 SNN（MLP）
- `snn/encoding.py`: 泊松编码，将像素强度转为时序脉冲
- `data.py`: MNIST 数据加载
- `train.py`: 训练与评估、保存最佳模型

## 训练

默认使用 20 个时间步：

```bash
python train.py --epochs 10 --steps 20 --batch_size 128 --device cuda
```

CPU 也可运行：

```bash
python train.py --device cpu
```

训练过程中会在 `./checkpoints/best_snn_mnist.pt` 保存最佳验证集精度的模型。

## 说明

- 前向：每个时间步把泊松采样得到的二值脉冲送入网络；读出层对所有时间步的 logits 求平均后计算交叉熵。
- 反向：发放函数使用三角形窗口的替代梯度，稳定训练。
- LIF 使用 soft reset，状态在每个 batch 的序列开始时自动重置。

## 可调整的超参数

- `--steps`: 时间步（越大越稳定，但耗时更久）
- `--lr`: 学习率（默认 1e-3）
- `--epochs`: 训练轮数
- `--batch_size`: 批大小

## 可能的精度

该实现（MLP）在 10~20 个 epoch、20~30 时间步下，通常在 MNIST 上可达到 97%+ 的验证精度（取决于随机种子与设备）。

## 许可证

MIT
"# snn_basic" 
