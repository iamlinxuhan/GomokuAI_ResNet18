"""
=============================================================================
硬件自动检测与智能配置模块
=============================================================================

功能：
1. 自动检测 GPU 型号、显存大小、计算能力
2. 自动检测 CPU 核心数、架构
3. 自动检测系统内存
4. 根据检测结果智能推荐最优训练参数
5. 跨平台兼容（Windows/Linux/macOS）
6. 提供硬件诊断信息和性能预估

适配您的戴尔 G16 (i7-12700H + RTX 3060 6GB) 及其他常见配置。
"""

import os
import sys
import math
import json
import platform
import logging
from typing import Dict, Any, Tuple, Optional

import torch
import numpy as np

logger = logging.getLogger(__name__)

# ============================================================================
# 硬件信息收集
# ============================================================================

def get_gpu_info() -> Dict[str, Any]:
    """
    全面检测 GPU 硬件信息。

    Returns 示例（RTX 3060 笔记本版）:
        {
            'available': True,
            'name': 'NVIDIA GeForce RTX 3060 Laptop GPU',
            'total_memory_mb': 6144,
            'compute_capability': (8, 6),
            'cuda_version': '11.8',
            'multi_processor_count': 30,
            'tensor_cores': True,      # RTX 3060 有 Tensor Cores（第3代）
            'max_threads_per_block': 1024,
            'warp_size': 32,
        }
    """
    info = {
        'available': torch.cuda.is_available(),
        'name': 'CPU (无CUDA GPU)',
        'total_memory_mb': 0,
        'compute_capability': (0, 0),
        'cuda_version': '',
        'multi_processor_count': 0,
        'tensor_cores': False,
        'max_threads_per_block': 1024,
        'warp_size': 32,
    }

    if not info['available']:
        return info

    try:
        device_count = torch.cuda.device_count()
        if device_count == 0:
            info['available'] = False
            return info

        # 使用第一个 GPU（通常是主 GPU）
        info['name'] = torch.cuda.get_device_name(0)

        # 显存信息
        try:
            total_memory = torch.cuda.get_device_properties(0).total_memory
            info['total_memory_mb'] = total_memory // (1024 * 1024)
        except Exception:
            # 备用方案：通过 nvidia-smi 获取
            import subprocess
            try:
                result = subprocess.run(
                    ['nvidia-smi', '--query-gpu=memory.total', '--format=csv,noheader,nounits'],
                    capture_output=True, text=True, timeout=5
                )
                if result.returncode == 0 and result.stdout.strip():
                    info['total_memory_mb'] = int(result.stdout.strip().split('\n')[0])
            except Exception:
                pass

        # 计算能力
        try:
            props = torch.cuda.get_device_properties(0)
            info['compute_capability'] = (props.major, props.minor)
            info['multi_processor_count'] = props.multi_processor_count
            info['max_threads_per_block'] = props.max_threads_per_block
        except Exception:
            pass

        # CUDA 版本
        try:
            info['cuda_version'] = torch.version.cuda or ''
        except Exception:
            pass

        # Tensor Cores 检测：计算能力 >= 7.0 即有 Tensor Cores
        cc = info['compute_capability']
        info['tensor_cores'] = (cc[0] >= 7)

        # 当前显存使用情况
        try:
            info['memory_allocated_mb'] = torch.cuda.memory_allocated(0) // (1024 * 1024)
            info['memory_reserved_mb'] = torch.cuda.memory_reserved(0) // (1024 * 1024)
        except Exception:
            info['memory_allocated_mb'] = 0
            info['memory_reserved_mb'] = 0

    except Exception as e:
        logger.warning(f"GPU 信息检测异常: {e}")
        info['available'] = False

    return info


def get_cpu_info() -> Dict[str, Any]:
    """
    检测 CPU 硬件信息。

    Returns 示例 (i7-12700H):
        {
            'cores_physical': 14,        # 6P + 8E
            'cores_logical': 20,         # 超线程
            'architecture': 'x86_64',
            'processor_brand': 'Intel',
            'platform': 'Windows-11',
        }
    """
    info = {
        'cores_physical': os.cpu_count() or 1,
        'cores_logical': os.cpu_count() or 1,
        'architecture': platform.machine(),
        'processor_brand': 'Unknown',
        'platform': platform.platform(),
    }

    # 尝试获取物理核心数（比逻辑核心更准确）
    try:
        if sys.platform == 'linux':
            # Linux 下通过 /proc/cpuinfo 获取
            result = os.popen('cat /proc/cpuinfo | grep "cpu cores" | uniq').read()
            if result:
                info['cores_physical'] = int(result.split(':')[1].strip())
        elif sys.platform == 'win32':
            # Windows 下通过环境变量获取（抑制 stderr）
            try:
                import subprocess
                result = subprocess.run(
                    ['wmic', 'cpu', 'get', 'NumberOfCores'],
                    capture_output=True, text=True, timeout=5
                )
                if result.returncode == 0:
                    lines = [l.strip() for l in result.stdout.split('\n') if l.strip()]
                    if len(lines) >= 2:
                        info['cores_physical'] = int(lines[1])
            except Exception:
                pass
    except Exception:
        pass

    # 检测是否为 Intel/AMD（使用 subprocess 抑制错误输出）
    try:
        import subprocess
        if sys.platform == 'win32':
            result = subprocess.run(
                ['wmic', 'cpu', 'get', 'Name'],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0 and 'Intel' in result.stdout:
                info['processor_brand'] = 'Intel'
            elif result.returncode == 0 and 'AMD' in result.stdout:
                info['processor_brand'] = 'AMD'
        elif sys.platform == 'linux':
            result = subprocess.run(
                ['grep', 'vendor_id', '/proc/cpuinfo'],
                capture_output=True, text=True, timeout=5
            )
            if 'GenuineIntel' in result.stdout:
                info['processor_brand'] = 'Intel'
            elif 'AuthenticAMD' in result.stdout:
                info['processor_brand'] = 'AMD'
    except Exception:
        pass

    return info


def get_system_memory_mb() -> int:
    """获取系统总内存（MB）"""
    try:
        if sys.platform == 'win32':
            import ctypes
            kernel32 = ctypes.windll.kernel32
            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]
            memory_status = MEMORYSTATUSEX()
            memory_status.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            kernel32.GlobalMemoryStatusEx(ctypes.byref(memory_status))
            return memory_status.ullTotalPhys // (1024 * 1024)
        elif sys.platform == 'linux':
            result = os.popen('grep MemTotal /proc/meminfo').read()
            if result:
                return int(result.split()[1]) // 1024
    except Exception:
        pass

    return 16384  # 默认 16GB


# ============================================================================
# 硬件分级与智能参数推荐
# ============================================================================

def classify_gpu_tier(gpu_info: Dict[str, Any]) -> str:
    """
    将 GPU 分为 5 个等级，用于选择合适的模型和参数。

    等级说明：
        - 'tier_0': 无 GPU / 仅集成显卡 → CPU 模式
        - 'tier_1': 低端 GPU（< 4GB）→ 极小模型
        - 'tier_2': 中端 GPU（4-6GB）→ 小模型  ← RTX 3060 6GB 属于此档
        - 'tier_3': 中高端 GPU（6-8GB）→ 中模型
        - 'tier_4': 高端 GPU（8-12GB）→ 大模型
        - 'tier_5': 旗舰 GPU（> 12GB）→ 超大模型
    """
    if not gpu_info['available']:
        return 'tier_0'

    mem = gpu_info['total_memory_mb']

    # 显存是主要分级依据
    if mem < 2048:        # < 2GB
        return 'tier_0'
    elif mem < 4096:      # 2-4GB
        return 'tier_1'
    elif mem < 6144:      # 4-6GB
        return 'tier_2'
    elif mem < 8192:      # 6-8GB
        return 'tier_3'
    elif mem < 12288:     # 8-12GB
        return 'tier_4'
    else:                  # > 12GB
        return 'tier_5'


def get_gpu_tier_name(tier: str) -> str:
    """获取 GPU 等级的中文名称"""
    names = {
        'tier_0': '无GPU/仅CPU',
        'tier_1': '低端GPU (<4GB)',
        'tier_2': '中端GPU (4-6GB)',
        'tier_3': '中高端GPU (6-8GB)',
        'tier_4': '高端GPU (8-12GB)',
        'tier_5': '旗舰GPU (>12GB)',
    }
    return names.get(tier, '未知')


def recommend_network_params(gpu_tier: str) -> Dict[str, int]:
    """
    根据 GPU 等级推荐网络结构参数。

    不同网络规模的参数量和显存占用：
        - 极小（32通道, 5残差块）:  ~0.1M 参数, ~50MB 显存
        - 小  （64通道, 7残差块）:  ~0.4M 参数, ~150MB 显存  ← RTX3060 推荐
        - 中  （128通道, 7残差块）:  ~1.6M 参数, ~500MB 显存  ← 原版配置
        - 大  （192通道, 10残差块）: ~5.0M 参数, ~1.5GB 显存
        - 超大（256通道, 15残差块）: ~12M 参数, ~3.5GB 显存

    对于 RTX 3060 6GB:
        MCTS batch 推理也消耗显存（400次模拟约占用 1-2GB）
        加上训练 batch（1024 约占用 1-2GB）
        推荐中等网络（128通道, 7残差块）或中大型（192通道, 10残差块）
    """
    params = {
        'tier_0': {'num_filters': 32,  'num_res_blocks': 5,  'policy_channels': 16, 'value_channels': 16, 'value_hidden': 64},
        'tier_1': {'num_filters': 64,  'num_res_blocks': 7,  'policy_channels': 16, 'value_channels': 16, 'value_hidden': 64},
        'tier_2': {'num_filters': 128, 'num_res_blocks': 7,  'policy_channels': 32, 'value_channels': 32, 'value_hidden': 128},  # ← 原版
        'tier_3': {'num_filters': 128, 'num_res_blocks': 10, 'policy_channels': 32, 'value_channels': 32, 'value_hidden': 128},
        'tier_4': {'num_filters': 192, 'num_res_blocks': 10, 'policy_channels': 48, 'value_channels': 48, 'value_hidden': 192},
        'tier_5': {'num_filters': 256, 'num_res_blocks': 15, 'policy_channels': 64, 'value_channels': 64, 'value_hidden': 256},
    }
    return params.get(gpu_tier, params['tier_1'])


def recommend_batch_size(gpu_info: Dict[str, Any], gpu_tier: str,
                         num_filters: int = 128,
                         num_res_blocks: int = 7) -> int:
    """
    根据 GPU 显存和网络大小推荐安全的训练 batch_size。

    使用经验查找表（基于五子棋 ResNet 的实际测试数据）：
      - 128通道/7残差块: batch 1024 在 RTX3060 6GB 上安全
      - 更小网络可更大 batch，更大网络需减小 batch

    策略：
        - 基于 GPU 等级和网络规模查找经验值
        - 调整系数适应不同 GPU
    """
    if gpu_tier == 'tier_0':
        return 256  # CPU 模式用小 batch

    mem_mb = gpu_info['total_memory_mb']

    # 基础 batch 查找表（针对 128ch/7block 的经验安全值）
    base_batch_map = {
        'tier_1': 64,     # 2-4GB
        'tier_2': 64,     # 4-6GB  ← RTX3060 6GB: 64安全（数据增强8→512）
        'tier_3': 128,    # 6-8GB
        'tier_4': 256,    # 8-12GB
        'tier_5': 512,    # >12GB
    }
    base_batch = base_batch_map.get(gpu_tier, 256)

    # 根据网络规模调整
    scale_factor = ((num_filters / 128.0) ** 2) * (num_res_blocks / 7.0) ** 0.5
    adjusted_batch = int(base_batch / max(0.5, scale_factor))

    # 限制为 2 的幂
    batch_sizes = [32, 64, 128, 256, 512, 1024, 2048, 4096]
    safe_batch = 32
    for bs in batch_sizes:
        if bs <= adjusted_batch:
            safe_batch = bs
        else:
            break

    return safe_batch


def recommend_mcts_simulations(gpu_info: Dict[str, Any], gpu_tier: str,
                                cpu_count: int) -> int:
    """
    根据硬件性能推荐 MCTS 模拟次数。

    MCTS 每次模拟都需要一次神经网络前向推理。
    RTX 3060 6GB 的单次推理时间约 2-4ms（128通道网络），
    400 次模拟约 0.8-1.6 秒，属于合理范围。

    对于更快 GPU，可以增加模拟次数以提升棋力。
    """
    if gpu_tier == 'tier_0':
        # CPU 模式：MCTS 受限，减少模拟次数
        return min(200, max(50, cpu_count * 10))

    base_simulations = {
        'tier_0': 50,
        'tier_1': 50,
        'tier_2': 50,     # 默认50次，自动调参会逐步增加
        'tier_3': 100,
        'tier_4': 200,
        'tier_5': 400,
    }

    # 训练模式下不应用 Tensor Core 加成（优先保证速度）
    # 对弈模式下通过 set_timeout 控制搜索强度

    sims = base_simulations.get(gpu_tier, 50)

    # 限制在合理范围内（训练时从低开始，自动调参逐步增加）
    return max(20, min(2000, sims))


def recommend_traditional_ai_depth(gpu_tier: str, cpu_count: int) -> Tuple[int, int, int]:
    """
    推荐传统 AI 搜索参数。

    Returns:
        (初始深度, 最小深度, 最大深度)
    """
    if gpu_tier == 'tier_0':
        # CPU 模式：GPU 加速不可用，搜索较慢
        base = 2 if cpu_count <= 4 else 4
        return (base, 1, min(6, cpu_count))

    # GPU 加速版：搜索速度大幅提升
    depth_map = {
        'tier_0': (1, 1, 4),
        'tier_1': (1, 1, 6),
        'tier_2': (1, 1, 8),     # 训练默认深度1，自动调参逐步增加
        'tier_3': (2, 1, 10),
        'tier_4': (3, 1, 12),
        'tier_5': (4, 1, 16),
    }
    return depth_map.get(gpu_tier, (4, 1, 8))


def recommend_learning_rate(batch_size: int) -> float:
    """
    根据 batch_size 推荐学习率。

    线性缩放规则：lr = base_lr * (batch_size / base_batch)
    其中 base_lr = 0.02, base_batch = 1024
    """
    base_lr = 0.02
    base_batch = 1024
    return base_lr * (batch_size / base_batch)


def recommend_workers(cpu_info: Dict[str, Any], gpu_available: bool) -> int:
    """
    推荐数据加载/并行工作进程数。

    Args:
        cpu_info: CPU 信息字典
        gpu_available: 是否有 GPU

    Returns:
        推荐的工作进程数
    """
    cpu_count = cpu_info.get('cores_physical', cpu_info.get('cores_logical', 4))

    if gpu_available:
        # GPU 模式下：数据加载不是瓶颈，用较少的进程
        return max(1, min(cpu_count // 2, 8))
    else:
        # CPU 模式下：需要更多并行
        return max(1, min(cpu_count, 16))


# ============================================================================
# 完整硬件感知配置
# ============================================================================

def get_hardware_summary() -> Dict[str, Any]:
    """
    获取完整硬件摘要信息（用于显示和日志）。
    """
    gpu = get_gpu_info()
    cpu = get_cpu_info()
    system_mem = get_system_memory_mb()
    gpu_tier = classify_gpu_tier(gpu)

    return {
        'gpu': gpu,
        'cpu': cpu,
        'system_memory_mb': system_mem,
        'gpu_tier': gpu_tier,
        'gpu_tier_name': get_gpu_tier_name(gpu_tier),
    }


def get_auto_config(overrides: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    根据硬件检测结果，自动生成完整的训练配置。

    这是核心函数，被 Trainer 和 GUI 调用以获得硬件自适应配置。

    Args:
        overrides: 用户手动覆盖的参数（优先级最高）

    Returns:
        完整的训练配置字典
    """
    gpu_info = get_gpu_info()
    cpu_info = get_cpu_info()
    system_mem_mb = get_system_memory_mb()
    gpu_tier = classify_gpu_tier(gpu_info)

    # ── 网络结构参数 ──
    net_params = recommend_network_params(gpu_tier)
    num_filters = net_params['num_filters']

    # ── 训练参数 ──
    batch_size = recommend_batch_size(gpu_info, gpu_tier, num_filters,
                                       net_params['num_res_blocks'])
    lr = recommend_learning_rate(batch_size)
    mcts_simulations = recommend_mcts_simulations(gpu_info, gpu_tier, cpu_info['cores_logical'])
    trad_depth, trad_min, trad_max = recommend_traditional_ai_depth(gpu_tier, cpu_info['cores_logical'])
    num_workers = recommend_workers(cpu_info, gpu_info['available'])

    # 根据系统内存调整经验池容量
    if system_mem_mb >= 32768:        # >= 32GB
        replay_capacity = 4000000
    elif system_mem_mb >= 16384:      # >= 16GB  ← 您的配置
        replay_capacity = 2000000
    elif system_mem_mb >= 8192:       # >= 8GB
        replay_capacity = 1000000
    else:
        replay_capacity = 500000

    # 根据总训练步数调整自动评估间隔
    total_steps = 500000
    eval_interval_steps = max(1000, total_steps // 100)  # 每 1% 评估一次

    config = {
        'hardware': {
            'gpu_tier': gpu_tier,
            'gpu_tier_name': get_gpu_tier_name(gpu_tier),
            'gpu_name': gpu_info['name'],
            'gpu_memory_mb': gpu_info['total_memory_mb'],
            'cpu_cores': cpu_info['cores_physical'],
            'cpu_threads': cpu_info['cores_logical'],
            'system_memory_mb': system_mem_mb,
            'tensor_cores': gpu_info['tensor_cores'],
        },
        'network': {
            'num_filters': num_filters,
            'num_res_blocks': net_params['num_res_blocks'],
            'policy_channels': net_params['policy_channels'],
            'value_channels': net_params['value_channels'],
            'value_hidden': net_params['value_hidden'],
        },
        'training': {
            'modes': ['self', 'trad'],
            'learning_rate': lr,
            'batch_size': batch_size,
            'num_mcts_simulations': mcts_simulations,
            'total_steps': total_steps,
            'use_cuda': gpu_info['available'],
            'model_dir': './models',
            'log_dir': './logs',
            'save_interval_minutes': 5,
            'auto_eval_games': 100,
            'eval_interval_steps': eval_interval_steps,
            'gradient_accumulation_steps': 1,
            'mixed_precision': gpu_info['available'],  # 有 GPU 就启用 AMP
            'num_workers': num_workers,
            'self_play': {
                'num_processes': max(1, num_workers // 2),
                'mcts_simulations': mcts_simulations,
            },
            'traditional': {
                'initial_depth': trad_depth,
                'depth_range': [trad_min, trad_max],
                'games_per_adjust': 10,
                'target_win_rate': 0.20,
                'win_rate_window': 100,
                'adjust_step': 1,
            },
            'human': {
                'enabled': False,
                'temperature_for_human_moves': 0.1,
            },
            'replay_buffer': {
                'capacity': replay_capacity,
                'sampling_weights': {
                    'self': 0.6,
                    'trad': 0.3,
                    'human': 0.1,
                },
                'prioritized_alpha': 0.6,   # 优先经验回放 alpha 参数
                'prioritized_beta': 0.4,    # 优先经验回放 beta 参数
            },
            'augmentation': {
                'enabled': True,            # 启用数据增强
                'symmetry': True,           # 8种对称变换
            },
        },
    }

    # ── 应用用户覆盖参数 ──
    if overrides:
        _deep_merge(config, overrides)

    return config


def _deep_merge(base: Dict, override: Dict):
    """
    深度合并两个字典。override 中的值会覆盖 base 中的对应值。
    仅合并字典类型，其他类型直接覆盖。
    """
    for key, value in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value


def format_hardware_summary(hw_summary: Dict[str, Any]) -> str:
    """
    将硬件摘要格式化为可读字符串（用于 GUI 显示和日志）。
    """
    gpu = hw_summary['gpu']
    cpu = hw_summary['cpu']

    lines = []
    lines.append("=" * 50)
    lines.append("🖥️  硬件检测报告")
    lines.append("=" * 50)

    # CPU 信息
    lines.append(f"\nCPU: {cpu.get('processor_brand', 'Unknown')} "
                 f"({cpu['cores_physical']}物理核心 / {cpu['cores_logical']}逻辑线程)")
    lines.append(f"系统内存: {hw_summary['system_memory_mb']} MB")

    # GPU 信息
    if gpu['available']:
        lines.append(f"\nGPU: {gpu['name']}")
        lines.append(f"显存: {gpu['total_memory_mb']} MB")
        cc = gpu['compute_capability']
        lines.append(f"计算能力: {cc[0]}.{cc[1]}")
        if gpu['tensor_cores']:
            lines.append("Tensor Cores: ✅ 支持")
        lines.append(f"CUDA版本: {gpu.get('cuda_version', 'N/A')}")
    else:
        lines.append("\nGPU: ❌ 未检测到 CUDA GPU，使用 CPU 模式")

    lines.append(f"\nGPU等级: {hw_summary['gpu_tier_name']} ({hw_summary['gpu_tier']})")
    lines.append("=" * 50)

    return "\n".join(lines)


def format_auto_config_summary(config: Dict[str, Any]) -> str:
    """
    将自动生成的配置格式化为可读字符串（用于 GUI 显示）。
    """
    train = config.get('training', {})
    net = config.get('network', {})
    hw = config.get('hardware', {})

    lines = []
    lines.append("=" * 50)
    lines.append("⚙️  自动推荐训练配置")
    lines.append("=" * 50)

    lines.append(f"\n📐 网络结构:")
    lines.append(f"   - 卷积通道数: {net.get('num_filters', 128)}")
    lines.append(f"   - 残差块数:   {net.get('num_res_blocks', 7)}")
    lines.append(f"   - 策略头通道: {net.get('policy_channels', 32)}")
    lines.append(f"   - 价值头通道: {net.get('value_channels', 32)}")

    lines.append(f"\n🎯 训练参数:")
    lines.append(f"   - Batch Size:     {train.get('batch_size', 1024)}")
    lines.append(f"   - 学习率:          {train.get('learning_rate', 0.02):.5f}")
    lines.append(f"   - MCTS模拟次数:    {train.get('num_mcts_simulations', 400)}")
    lines.append(f"   - 混合精度(AMP):   {'✅ 开启' if train.get('mixed_precision', False) else '❌ 关闭'}")
    lines.append(f"   - 梯度累积步数:    {train.get('gradient_accumulation_steps', 1)}")
    lines.append(f"   - 数据增强:        {'✅ 开启' if train.get('augmentation', {}).get('enabled', False) else '❌ 关闭'}")

    lines.append(f"\n🤖 传统AI:")
    trad = train.get('traditional', {})
    lines.append(f"   - 初始搜索深度: {trad.get('initial_depth', 4)}")
    lines.append(f"   - 深度范围:     {trad.get('depth_range', [1, 8])}")
    lines.append(f"   - 目标胜率:     {trad.get('target_win_rate', 0.20):.0%}")

    lines.append(f"\n💾 经验回放池:")
    rb = train.get('replay_buffer', {})
    lines.append(f"   - 容量:       {rb.get('capacity', 2000000)}")
    lines.append(f"   - 优先回放:   {'✅ 开启' if rb.get('prioritized_alpha', 0) > 0 else '❌ 关闭'}")

    lines.append("\n" + "=" * 50)
    return "\n".join(lines)


# ============================================================================
# GPU 压力测试（可选，用于精细调优）
# ============================================================================

def benchmark_gpu_memory(num_filters: int = 128, num_res_blocks: int = 7,
                         batch_size: int = 1024) -> Dict[str, Any]:
    """
    GPU 压力测试：测试指定网络配置和 batch_size 下的显存使用和速度。

    返回各项指标，帮助用户精确调优（而非仅靠经验公式）。
    这是一个轻量级测试，运行约 5-10 秒。

    Returns:
        {
            'memory_used_mb': 1234,        # 批处理使用的显存
            'memory_per_sample_kb': 1.23,  # 每样本平均显存
            'forward_time_ms': 3.5,        # 前向推理时间
            'recommended_batch': 1024,      # 推荐的 batch size
            'safe': True,                   # 当前配置是否安全
        }
    """
    if not torch.cuda.is_available():
        return {'safe': False, 'error': 'CUDA not available'}

    result = {}
    device = torch.device('cuda:0')

    try:
        # 构建测试网络
        from network import GomokuNet
        model = GomokuNet(
            num_filters=num_filters,
            num_res_blocks=num_res_blocks,
        ).to(device)

        # 清空缓存
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        mem_before = torch.cuda.memory_allocated(device)

        # 创建测试输入
        test_input = torch.randn(batch_size, 4, 15, 15, device=device)

        # 前向推理 + 计时
        torch.cuda.synchronize()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()
        with torch.no_grad():
            policy, value = model(test_input)
        end_event.record()
        torch.cuda.synchronize()

        # 计算显存使用
        mem_after = torch.cuda.memory_allocated(device)
        mem_used = mem_after - mem_before
        result['memory_used_mb'] = mem_used / (1024 * 1024)
        result['memory_per_sample_kb'] = mem_used / batch_size / 1024

        # 计算耗时
        result['forward_time_ms'] = start_event.elapsed_time(end_event)

        # 清理
        del model, test_input
        torch.cuda.empty_cache()

        # 推荐安全的 batch size
        total_mem = torch.cuda.get_device_properties(0).total_memory
        usable_mem = total_mem * 0.75
        safe_batch = int(usable_mem / (mem_used / batch_size) * 0.85)
        # 取 2 的幂
        batch_sizes = [32, 64, 128, 256, 512, 1024, 2048, 4096]
        result['recommended_batch'] = 32
        for bs in batch_sizes:
            if bs <= safe_batch:
                result['recommended_batch'] = bs

        result['safe'] = True

    except RuntimeError as e:
        if 'out of memory' in str(e).lower():
            result['safe'] = False
            result['error'] = 'OOM'
            result['recommended_batch'] = batch_size // 2
        else:
            result['safe'] = False
            result['error'] = str(e)
        torch.cuda.empty_cache()

    return result


# ============================================================================
# JSON 配置文件读写（一次检测，永久使用，可手动编辑）
# ============================================================================

HARDWARE_JSON = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'hardware_config.json')


def save_hardware_config(config: dict = None, hw_summary: dict = None):
    """
    将硬件信息和推荐配置保存到 JSON 文件。
    下次启动时直接读取，无需重新检测硬件。

    Args:
        config: get_auto_config() 返回的完整配置
        hw_summary: get_hardware_summary() 返回的硬件摘要
    """
    if config is None:
        config = get_auto_config()
    if hw_summary is None:
        hw_summary = get_hardware_summary()

    save_data = {
        '_meta': {
            'version': '2.0',
            'description': '硬件检测配置缓存，可手动编辑调整参数',
            'note': '删除此文件后下次启动会自动重新检测硬件',
        },
        'hardware': hw_summary,
        'training': config.get('training', {}),
        'network': config.get('network', {}),
    }

    try:
        with open(HARDWARE_JSON, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, indent=2, ensure_ascii=False)
        logger.info(f"硬件配置已保存到 {HARDWARE_JSON}")
    except Exception as e:
        logger.warning(f"保存硬件配置失败: {e}")


def load_hardware_config() -> dict:
    """
    从 JSON 文件加载硬件配置。如果文件不存在则重新检测并保存。

    Returns:
        get_auto_config() 格式的完整配置字典
    """
    if not os.path.exists(HARDWARE_JSON):
        logger.info("未找到硬件配置文件，正在检测硬件...")
        config = get_auto_config()
        save_hardware_config(config)
        return config

    try:
        with open(HARDWARE_JSON, 'r', encoding='utf-8') as f:
            saved = json.load(f)

        logger.info(f"从 {HARDWARE_JSON} 加载硬件配置")

        # 用保存的硬件信息重新生成配置（保留用户可能手动修改的值）
        config = get_auto_config()

        # 如果用户手动编辑了 JSON 中的 training 或 network，优先使用
        if 'training' in saved:
            _deep_merge(config['training'], saved['training'])
        if 'network' in saved:
            _deep_merge(config['network'], saved['network'])

        return config

    except Exception as e:
        logger.warning(f"加载硬件配置失败 ({e})，重新检测...")
        config = get_auto_config()
        save_hardware_config(config)
        return config


# ============================================================================
# 主入口：打印硬件信息和推荐配置
# ============================================================================

def main():
    """打印硬件检测报告和推荐配置"""
    # 配置日志
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s [%(levelname)s] %(message)s')

    # 硬件信息
    hw = get_hardware_summary()
    print(format_hardware_summary(hw))

    # 推荐配置
    config = get_auto_config()
    print(format_auto_config_summary(config))

    # 可选：运行 benchmark
    print("\n是否运行 GPU 压力测试（约 5-10 秒）？(y/N): ", end="")
    try:
        choice = input().strip().lower()
        if choice == 'y':
            train_cfg = config['training']
            net_cfg = config['network']
            print("\n运行 GPU 压力测试...")
            bench = benchmark_gpu_memory(
                num_filters=net_cfg['num_filters'],
                num_res_blocks=net_cfg['num_res_blocks'],
                batch_size=train_cfg['batch_size'],
            )
            if bench.get('safe', False):
                print(f"  前向推理: {bench['forward_time_ms']:.2f} ms")
                print(f"  显存使用: {bench['memory_used_mb']:.1f} MB")
                print(f"  推荐 batch: {bench['recommended_batch']}")
            else:
                print(f"  测试失败: {bench.get('error', 'unknown')}")
                print(f"  建议减小 batch_size 或网络规模")
    except EOFError:
        pass

    return config


if __name__ == '__main__':
    main()
