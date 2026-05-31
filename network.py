"""
五子棋神经网络模型
基于AlphaZero风格的残差网络：策略头 + 价值头
输入: (N, 4, 15, 15) → 输出: 策略概率 (N, 225), 价值 (N, 1)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
import math


class ResidualBlock(nn.Module):
    """残差块：两个3×3卷积 + BN + ReLU + 跳跃连接"""

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = F.relu(out)
        return out


class GomokuNet(nn.Module):
    """
    五子棋神经网络
    参数约160万（128通道，7残差块）
    """

    def __init__(self,
                 board_size: int = 15,
                 in_channels: int = 4,
                 num_filters: int = 128,
                 num_res_blocks: int = 7,
                 policy_channels: int = 32,
                 value_channels: int = 32,
                 value_hidden: int = 128):
        super().__init__()
        self.board_size = board_size
        self.num_filters = num_filters
        self.num_res_blocks = num_res_blocks

        # 初始卷积层
        self.conv_initial = nn.Conv2d(in_channels, num_filters, kernel_size=3, padding=1, bias=False)
        self.bn_initial = nn.BatchNorm2d(num_filters)

        # 残差塔
        self.res_blocks = nn.ModuleList([
            ResidualBlock(num_filters) for _ in range(num_res_blocks)
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
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        # 策略输出初始化为小值
        nn.init.constant_(self.policy_fc.weight, 0)
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
        p = p.view(p.size(0), -1)  # 展平
        policy_logits = self.policy_fc(p)

        # 价值头
        v = F.relu(self.value_bn(self.value_conv(x)))
        v = v.view(v.size(0), -1)
        v = F.relu(self.value_fc1(v))
        value = torch.tanh(self.value_fc2(v))

        return policy_logits, value

    def predict(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        推理模式：返回softmax策略概率和价值
        Args:
            x: (N, 4, 15, 15)
        Returns:
            policy_probs: (N, 225) softmax概率
            value: (N, 1)
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
        """保存检查点"""
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'board_size': self.board_size,
            'num_filters': self.num_filters,
            'num_res_blocks': self.num_res_blocks,
            'step': step,
        }
        if optimizer is not None:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        if extra is not None:
            checkpoint.update(extra)
        torch.save(checkpoint, path)

    @classmethod
    def load_checkpoint(cls, path: str, device: torch.device = None) -> Tuple['GomokuNet', dict]:
        """加载检查点"""
        checkpoint = torch.load(path, map_location=device or torch.device('cpu'))
        model = cls(
            board_size=checkpoint.get('board_size', 15),
            num_filters=checkpoint.get('num_filters', 128),
            num_res_blocks=checkpoint.get('num_res_blocks', 7),
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        if device:
            model = model.to(device)
        info = {
            'step': checkpoint.get('step', 0),
            'optimizer_state_dict': checkpoint.get('optimizer_state_dict'),
        }
        # 合并额外信息
        for key in checkpoint:
            if key not in ['model_state_dict', 'optimizer_state_dict', 'board_size',
                           'num_filters', 'num_res_blocks', 'step']:
                info[key] = checkpoint[key]
        return model, info
