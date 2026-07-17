"""
=============================================================================
五子棋神经网络模型 v2.0 — 硬件自适应 + SE 注意力增强
=============================================================================

基于 AlphaZero 风格的残差网络：策略头 + 价值头
输入: (N, 4, 15, 15) → 输出: 策略概率 (N, 225), 价值 (N, 1)

v2.0 优化：
  1. 【SE 注意力模块】Squeeze-and-Excitation 模块，提升特征表达能力
  2. 【硬件自适应构造】根据硬件自动选择网络大小
  3. 【参数初始化优化】更好的权重初始化策略

网络规模选项（根据显存自动选择）：
  - 极小（16通道, 5残差块）:  ~0.03M 参数 → 显存 < 2GB
  - 小  （64通道, 7残差块）:  ~0.4M 参数 → 显存 2-4GB
  - 中  （128通道, 7残差块）:  ~1.6M 参数 → 显存 4-8GB  ← RTX3060 6GB
  - 大  （192通道, 10残差块）: ~5.0M 参数 → 显存 8-12GB
  - 超大（256通道, 15残差块）: ~12M 参数 → 显存 > 12GB
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import math


class SELayer(nn.Module):
    """
    Squeeze-and-Excitation 注意力模块。

    通过全局平均池化 + 两个全连接层学习通道间注意力权重，
    自动增强重要特征通道、抑制不重要通道。
    引入极少的额外参数（约 2*C*C/r），对五子棋的棋型检测有明显提升。
    """

    def __init__(self, channels: int, reduction: int = 8):
        """
        Args:
            channels: 输入通道数
            reduction: 压缩比例（默认 8，参数约 2*C*C/8 = C^2/4）
        """
        super().__init__()
        self.fc1 = nn.Linear(channels, max(4, channels // reduction))
        self.fc2 = nn.Linear(max(4, channels // reduction), channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (N, C, H, W)
        Returns:
            (N, C, H, W) 通道注意力加权后的特征
        """
        batch_size, channels, _, _ = x.size()
        # Squeeze: 全局平均池化
        y = F.adaptive_avg_pool2d(x, 1).view(batch_size, channels)
        # Excitation: 两层全连接 + Sigmoid
        y = F.relu(self.fc1(y))
        y = torch.sigmoid(self.fc2(y))
        # Scale: 通道加权
        return x * y.view(batch_size, channels, 1, 1)


class ResidualBlock(nn.Module):
    """
    残差块 v2.0 — 可选 SE 注意力模块。

    结构：Conv3×3 → BN → ReLU → Conv3×3 → BN → [SE] → + → ReLU
    """

    def __init__(self, channels: int, use_se: bool = True):
        """
        Args:
            channels: 输入/输出通道数
            use_se: 是否添加 SE 注意力模块
        """
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

        # SE 注意力模块（可选）
        self.se = SELayer(channels) if use_se else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.se is not None:
            out = self.se(out)
        out += residual
        out = F.relu(out)
        return out


class GomokuNet(nn.Module):
    """
    五子棋神经网络 v2.0 — 硬件自适应 + SE 注意力。

    使用方法：
        # 自动根据硬件选择网络大小
        from hardware_auto_config import get_auto_config
        config = get_auto_config()
        net = GomokuNet.from_hardware_config(config)

        # 或手动指定
        net = GomokuNet(num_filters=128, num_res_blocks=7, use_se=True)
    """

    def __init__(self,
                 board_size: int = 15,
                 in_channels: int = 4,
                 num_filters: int = 128,
                 num_res_blocks: int = 7,
                 policy_channels: int = 32,
                 value_channels: int = 32,
                 value_hidden: int = 128,
                 use_se: bool = True):
        """
        Args:
            board_size: 棋盘大小
            in_channels: 输入通道数（固定为4）
            num_filters: 卷积通道数（硬件自适应）
            num_res_blocks: 残差块数（硬件自适应）
            policy_channels: 策略头通道数
            value_channels: 价值头通道数
            value_hidden: 价值头全连接隐层大小
            use_se: 是否使用 SE 注意力模块
        """
        super().__init__()
        self.board_size = board_size
        self.num_filters = num_filters
        self.num_res_blocks = num_res_blocks
        self.use_se = use_se

        # 初始卷积层
        self.conv_initial = nn.Conv2d(in_channels, num_filters, kernel_size=3, padding=1, bias=False)
        self.bn_initial = nn.BatchNorm2d(num_filters)

        # 残差塔（v2.0: 支持 SE 注意力）
        self.res_blocks = nn.ModuleList([
            ResidualBlock(num_filters, use_se=use_se) for _ in range(num_res_blocks)
        ])

        # 策略头
        self.policy_conv = nn.Conv2d(num_filters, policy_channels, kernel_size=1, bias=False)
        self.policy_bn = nn.BatchNorm2d(policy_channels)
        self.policy_fc = nn.Linear(policy_channels * board_size * board_size, board_size * board_size)

        # 价值头
        self.value_conv = nn.Conv2d(num_filters, value_channels, kernel_size=1, bias=False)
        self.value_bn = nn.BatchNorm2d(value_channels)
        self.value_fc1 = nn.Linear(value_channels * board_size * board_size, value_hidden)
        self.value_fc2 = nn.Linear(value_hidden, 1)

        self._init_weights()

    def _init_weights(self):
        """v2.0 权重初始化（更细致的策略）"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        # 策略输出初始化为小值（方便早期探索）
        nn.init.normal_(self.policy_fc.weight, mean=0, std=0.001)
        nn.init.constant_(self.policy_fc.bias, 0)
        # 价值输出初始化为0（tanh(0)=0）
        nn.init.constant_(self.value_fc2.weight, 0)
        nn.init.constant_(self.value_fc2.bias, 0)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (N, 4, 15, 15) 输入状态
        Returns:
            policy_logits: (N, 225) 策略logits
            value: (N, 1) 价值 [-1, 1]
        """
        # 初始卷积
        x = F.relu(self.bn_initial(self.conv_initial(x)))

        # 残差塔
        for res_block in self.res_blocks:
            x = res_block(x)

        # 策略头
        p = F.relu(self.policy_bn(self.policy_conv(x)))
        p = p.view(p.size(0), -1)
        policy_logits = self.policy_fc(p)

        # 价值头
        v = F.relu(self.value_bn(self.value_conv(x)))
        v = v.view(v.size(0), -1)
        v = F.relu(self.value_fc1(v))
        value = torch.tanh(self.value_fc2(v))

        return policy_logits, value

    def predict(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        推理模式：返回 softmax 策略概率和价值。
        """
        self.eval()
        with torch.no_grad():
            policy_logits, value = self.forward(x)
            policy_probs = F.softmax(policy_logits, dim=1)
        return policy_probs, value

    def get_device(self) -> torch.device:
        """获取模型所在设备"""
        return next(self.parameters()).device

    def count_parameters(self) -> int:
        """统计参数量"""
        return sum(p.numel() for p in self.parameters())

    def save_checkpoint(self, path: str, optimizer=None, step: int = 0, extra: dict = None):
        """保存检查点（v2.0: 保存网络结构参数）"""
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'board_size': self.board_size,
            'num_filters': self.num_filters,
            'num_res_blocks': self.num_res_blocks,
            'policy_channels': self.policy_conv.out_channels,
            'value_channels': self.value_conv.out_channels,
            'value_hidden': self.value_fc1.out_features,
            'use_se': self.use_se,
            'step': step,
        }
        if optimizer is not None:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        if extra is not None:
            checkpoint.update(extra)
        torch.save(checkpoint, path)

    @classmethod
    def load_checkpoint(cls, path: str, device: torch.device = None) -> Tuple['GomokuNet', dict]:
        """加载检查点（v2.0: 自动适配保存时的网络结构）"""
        checkpoint = torch.load(path, map_location=device or torch.device('cpu'))

        # 尝试加载 v2.0 结构参数，兼容 v1.0 旧格式
        model = cls(
            board_size=checkpoint.get('board_size', 15),
            num_filters=checkpoint.get('num_filters', 128),
            num_res_blocks=checkpoint.get('num_res_blocks', 7),
            policy_channels=checkpoint.get('policy_channels', 32),
            value_channels=checkpoint.get('value_channels', 32),
            value_hidden=checkpoint.get('value_hidden', 128),
            use_se=checkpoint.get('use_se', False),  # 旧模型没有 SE
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        if device:
            model = model.to(device)

        info = {
            'step': checkpoint.get('step', 0),
            'optimizer_state_dict': checkpoint.get('optimizer_state_dict'),
        }
        for key in checkpoint:
            if key not in ['model_state_dict', 'optimizer_state_dict', 'board_size',
                           'num_filters', 'num_res_blocks', 'policy_channels',
                           'value_channels', 'value_hidden', 'use_se', 'step']:
                info[key] = checkpoint[key]

        return model, info

    @classmethod
    def from_hardware_config(cls, config: dict) -> 'GomokuNet':
        """
        根据硬件配置自动创建网络实例。

        Args:
            config: hardware_auto_config.get_auto_config() 的返回值

        Returns:
            配置好的 GomokuNet 实例
        """
        net_cfg = config.get('network', {})
        return cls(
            num_filters=net_cfg.get('num_filters', 128),
            num_res_blocks=net_cfg.get('num_res_blocks', 7),
            policy_channels=net_cfg.get('policy_channels', 32),
            value_channels=net_cfg.get('value_channels', 32),
            value_hidden=net_cfg.get('value_hidden', 128),
            use_se=True,
        )
