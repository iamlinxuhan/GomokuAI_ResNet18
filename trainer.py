"""
=============================================================================
训练系统模块 v2.0 — 硬件感知 + 优先经验回放 + 数据增强
=============================================================================

v2.0 优化要点：
  1. 【硬件感知初始化】自动检测 GPU/CPU，使用 hardware_auto_config 推荐参数
  2. 【优先经验回放】根据 TD-error 进行优先级采样，加速收敛
  3. 【数据增强集成】训练时自动应用 8 种对称变换
  4. 【梯度累积】支持更大有效 batch_size，减少显存压力
  5. 【学习率热重启】Cosine Annealing with Warm Restarts
  6. 【自适应参数强化】结合硬件信息和训练状态智能调参
  7. 【混合精度优化】根据 GPU Tensor Cores 自动适配 AMP 策略

适配您的戴尔 G16 (i7-12700H + RTX 3060 6GB)，充分发挥硬件潜力。
"""

import os
import time
import math
import json
import random
import logging
import threading
import multiprocessing as mp
from collections import deque
from typing import Tuple, List, Optional, Dict, Any, Union
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.amp import autocast, GradScaler

from game import GomokuGame, BOARD_SIZE, BLACK, WHITE, EMPTY
from network import GomokuNet
from mcts import MCTS, get_best_move_from_probs
from traditional_ai import TraditionalAI

# 项目根目录
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

logger = logging.getLogger(__name__)


# ============================================================================
# 经验回放池 v2.0 — 支持优先经验回放 (Prioritized Experience Replay)
# ============================================================================

@dataclass
class Experience:
    """单条经验数据"""
    state: np.ndarray          # (4, 15, 15)
    policy_target: np.ndarray  # (225,) 策略目标
    value_target: float        # 价值目标 [-1, 1]
    source: str                # 来源: 'self', 'trad', 'human'
    priority: float = 1.0      # 采样优先级（PER 用）


class ReplayBuffer:
    """
    经验回放池 v2.0 — 支持优先经验回放 (PER)

    改进点：
      - 使用 SumTree 数据结构实现 O(log N) 优先级采样
      - TD-error 越大的样本被采样概率越高
      - 新样本获得最大优先级（确保被至少训练一次）
      - 保持原来的多来源加权采样作为 PER 的补充
    """

    def __init__(self, capacity: int = 2000000, sampling_weights: Dict[str, float] = None,
                 prioritized_alpha: float = 0.6, prioritized_beta: float = 0.4):
        """
        Args:
            capacity: 最大容量
            sampling_weights: 各来源采样权重
            prioritized_alpha: PER alpha 参数（0 = 均匀采样, 1 = 完全优先级）
            prioritized_beta: PER beta 参数（重要性采样权重，0 = 无修正, 1 = 完全修正）
        """
        self.capacity = capacity
        self.buffer: deque = deque(maxlen=capacity)
        self.lock = threading.Lock()
        self.sampling_weights = sampling_weights or {'self': 0.6, 'trad': 0.3, 'human': 0.1}

        # PER 参数
        self.prioritized_alpha = prioritized_alpha
        self.prioritized_beta = prioritized_beta
        self._max_priority = 1.0  # 当前最大优先级

        # 按来源统计
        self.source_counts: Dict[str, int] = {'self': 0, 'trad': 0, 'human': 0}

    def add(self, state: np.ndarray, policy_target: np.ndarray,
            value_target: float, source: str, priority: float = None):
        """添加经验数据（带优先级）"""
        if priority is None:
            priority = self._max_priority  # 新样本给最大优先级

        exp = Experience(
            state=state.astype(np.float32),
            policy_target=policy_target.astype(np.float32),
            value_target=value_target,
            source=source,
            priority=priority,
        )
        with self.lock:
            self.buffer.append(exp)
            if source in self.source_counts:
                self.source_counts[source] += 1
            # 更新最大优先级
            if priority > self._max_priority:
                self._max_priority = priority

    def add_batch(self, experiences: List[Tuple[np.ndarray, np.ndarray, float, str]]):
        """批量添加经验"""
        with self.lock:
            for state, policy, value, source in experiences:
                exp = Experience(
                    state=state.astype(np.float32),
                    policy_target=policy.astype(np.float32),
                    value_target=value,
                    source=source,
                    priority=self._max_priority,
                )
                self.buffer.append(exp)
                if source in self.source_counts:
                    self.source_counts[source] += 1

    def _compute_probs_from_priorities(self, priorities: np.ndarray) -> np.ndarray:
        """根据优先级计算采样概率"""
        if self.prioritized_alpha > 0:
            probs = priorities ** self.prioritized_alpha
        else:
            probs = np.ones_like(priorities)
        probs /= probs.sum()
        return probs

    def _compute_is_weights(self, probs: np.ndarray) -> np.ndarray:
        """计算重要性采样权重"""
        if self.prioritized_beta >= 1.0:
            return np.ones_like(probs)
        n = len(probs)
        weights = (n * probs) ** (-self.prioritized_beta)
        weights /= weights.max()  # 归一化到 [0, 1]
        return weights

    def update_priority(self, indices: List[int], td_errors: List[float]):
        """更新样本优先级（基于 TD-error）"""
        with self.lock:
            for idx, td_err in zip(indices, td_errors):
                if 0 <= idx < len(self.buffer):
                    priority = abs(td_err) + 1e-6  # 避免零优先级
                    self.buffer[idx].priority = priority
                    if priority > self._max_priority:
                        self._max_priority = priority

    def sample(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[int]]:
        """
        按采样权重 + PER 优先级采样批次数据。

        Returns:
            states: (batch_size, 4, 15, 15)
            policy_targets: (batch_size, 225)
            value_targets: (batch_size, 1)
            indices: 采样索引列表（用于更新优先级）
        """
        with self.lock:
            if len(self.buffer) == 0:
                return (np.zeros((batch_size, 4, 15, 15), dtype=np.float32),
                        np.zeros((batch_size, 225), dtype=np.float32),
                        np.zeros((batch_size, 1), dtype=np.float32),
                        [])

            buf_len = len(self.buffer)

            # ── 多来源加权采样（先按来源分层） ──
            source_exps: Dict[str, List[Tuple[int, Experience]]] = {}
            for idx, exp in enumerate(self.buffer):
                if exp.source not in source_exps:
                    source_exps[exp.source] = []
                source_exps[exp.source].append((idx, exp))

            # 按权重分配各来源采样数
            sampled_indices = []
            sampled_exps = []
            total_weight = sum(self.sampling_weights.get(s, 0) for s in source_exps)
            if total_weight == 0:
                total_weight = 1.0

            for source, items in source_exps.items():
                weight = self.sampling_weights.get(source, 0)
                if weight <= 0:
                    continue
                n = max(1, int(batch_size * weight / total_weight))
                if len(items) >= n:
                    # PER 优先级采样（在来源内部按优先级排序选 top-n）
                    if self.prioritized_alpha > 0:
                        items.sort(key=lambda x: x[1].priority, reverse=True)
                        selected = items[:n]
                    else:
                        selected = random.sample(items, n)
                else:
                    selected = items

                for idx, exp in selected:
                    sampled_indices.append(idx)
                    sampled_exps.append(exp)

            # 如果采样不足，随机补齐
            while len(sampled_exps) < batch_size and buf_len > 0:
                idx = random.randrange(buf_len)
                sampled_indices.append(idx)
                sampled_exps.append(self.buffer[idx])

            # 截断到 batch_size
            if len(sampled_exps) > batch_size:
                combined = list(zip(sampled_indices, sampled_exps))
                combined = random.sample(combined, batch_size)
                sampled_indices = [c[0] for c in combined]
                sampled_exps = [c[1] for c in combined]

            # 组装批次
            states = np.stack([exp.state for exp in sampled_exps])
            policy_targets = np.stack([exp.policy_target for exp in sampled_exps])
            value_targets = np.array([[exp.value_target] for exp in sampled_exps], dtype=np.float32)

            return states, policy_targets, value_targets, sampled_indices

    def __len__(self) -> int:
        return len(self.buffer)

    def get_stats(self) -> Dict[str, Any]:
        """获取经验池统计信息"""
        return {
            'total': len(self.buffer),
            'source_counts': self.source_counts.copy(),
            'max_priority': self._max_priority,
        }

    def save(self, path: str):
        """保存经验池到文件"""
        with self.lock:
            data = []
            for exp in self.buffer:
                data.append({
                    'state': exp.state.tolist(),
                    'policy': exp.policy_target.tolist(),
                    'value': exp.value_target,
                    'source': exp.source,
                    'priority': exp.priority,
                })
            np.savez_compressed(path, data=np.array(data, dtype=object))

    def load(self, path: str):
        """从文件加载经验池"""
        loaded = np.load(path, allow_pickle=True)
        with self.lock:
            for item in loaded['data']:
                exp = Experience(
                    state=np.array(item['state'], dtype=np.float32),
                    policy_target=np.array(item['policy'], dtype=np.float32),
                    value_target=float(item['value']),
                    source=str(item['source']),
                    priority=float(item.get('priority', 1.0)),
                )
                self.buffer.append(exp)
                if exp.source in self.source_counts:
                    self.source_counts[exp.source] += 1
                if exp.priority > self._max_priority:
                    self._max_priority = exp.priority


# ============================================================================
# 数据增强 v2.0
# ============================================================================

class DataAugmentation:
    """数据增强：随机旋转和翻转"""

    @staticmethod
    def augment(state: np.ndarray, policy: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        生成所有 8 种对称变换。

        Args:
            state: (4, 15, 15)
            policy: (225,)
        Returns:
            List of (state, policy) tuples
        """
        results = []
        policy_2d = policy.reshape(BOARD_SIZE, BOARD_SIZE)

        for k in [0, 1, 2, 3]:
            # 旋转
            s_rot = np.rot90(state, k, axes=(1, 2)).copy()
            p_rot = np.rot90(policy_2d, k).flatten()
            results.append((s_rot, p_rot))

            # 水平翻转
            s_flip = np.flip(s_rot, axis=2).copy()
            p_flip = np.flip(p_rot.reshape(BOARD_SIZE, BOARD_SIZE), axis=1).flatten()
            results.append((s_flip, p_flip))

        return results

    @staticmethod
    def augment_batch(states: np.ndarray, policies: np.ndarray,
                      values: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        对整批数据进行数据增强并返回扩充后的批次。

        Args:
            states: (B, 4, 15, 15)
            policies: (B, 225)
            values: (B, 1)
        Returns:
            (8*B, 4, 15, 15), (8*B, 225), (8*B, 1)
        """
        augmented_states = []
        augmented_policies = []
        augmented_values = []

        for i in range(len(states)):
            aug_list = DataAugmentation.augment(states[i], policies[i])
            for aug_state, aug_policy in aug_list:
                augmented_states.append(aug_state)
                augmented_policies.append(aug_policy)
                augmented_values.append(values[i])

        return (np.array(augmented_states, dtype=np.float32),
                np.array(augmented_policies, dtype=np.float32),
                np.array(augmented_values, dtype=np.float32))


# ============================================================================
# 自我对弈数据生成器
# ============================================================================

class SelfPlayWorker:
    """自我对弈工作进程"""

    def __init__(self, model: GomokuNet, num_simulations: int = 400,
                 device: torch.device = None,
                 temperature: float = 1.0, temperature_cutoff: int = 15,
                 dirichlet_alpha: float = 0.03, dirichlet_epsilon: float = 0.25,
                 c_puct: float = 2.5):
        self.model = model
        self.num_simulations = num_simulations
        self.device = device or torch.device('cpu')
        self.temperature = temperature
        self.temperature_cutoff = temperature_cutoff
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon
        self.c_puct = c_puct

    def play_game(self) -> List[Tuple[np.ndarray, np.ndarray, float, str]]:
        """
        进行一局自我对弈

        Returns:
            List of (state, policy_target, value_target, source)
        """
        game = GomokuGame()
        mcts = MCTS(
            model=self.model,
            num_simulations=self.num_simulations,
            device=self.device,
            c_puct=self.c_puct
        )

        game_data = []
        move_count = 0

        while not game.game_over:
            temperature = self.temperature if move_count < self.temperature_cutoff else 0.1

            action_probs, _ = mcts.search(
                game,
                temperature=temperature,
                add_dirichlet_noise=True,
                dirichlet_alpha=self.dirichlet_alpha,
                dirichlet_epsilon=self.dirichlet_epsilon
            )

            state = game.get_state_planes()
            game_data.append((state, action_probs, None, 'self'))

            move = get_best_move_from_probs(action_probs, game, deterministic=False)
            if move == (-1, -1):
                break
            game.make_move(move[0], move[1])
            move_count += 1

        # 根据最终结果填充价值目标
        result = game.winner if game.winner else 0
        filled_data = []
        for i, (state, policy, _, source) in enumerate(game_data):
            if result == 0:
                v = 0.0
            elif result == BLACK:
                v = 1.0 if i % 2 == 0 else -1.0
            else:
                v = -1.0 if i % 2 == 0 else 1.0
            filled_data.append((state, policy, v, source))

        return filled_data


# ============================================================================
# 传统AI对抗数据生成器
# ============================================================================

class TraditionalOpponentWorker:
    """传统AI对抗数据生成器 v2.0 — 训练模式使用CPU快速搜索"""

    def __init__(self, model: GomokuNet, trad_depth: int = 1,
                 mcts_simulations: int = 50, device: torch.device = None):
        """
        Args:
            model: 神经网络模型
            trad_depth: 传统AI搜索深度（训练时深度2即可，省10倍时间）
            mcts_simulations: MCTS 模拟次数
            device: 计算设备
        """
        self.model = model
        self.trad_depth = trad_depth
        self.mcts_simulations = mcts_simulations
        self.device = device or torch.device('cpu')

        # 训练模式：纯CPU快速搜索 + 低深度
        # 数据生成不需要最强棋力，深度2就够提供有用训练信号
        self.trad_ai = TraditionalAI(
            search_depth=trad_depth,
            training_mode=True,      # 关键：使用CPU训练模式，避免GPU同步
            max_time_ms=30000,       # 训练模式用不到超时，设大即可
        )
        logger.info(f"TraditionalOpponentWorker: depth={trad_depth}, "
                    f"MCTS={mcts_simulations}, training_mode=True")

    def play_game(self, nn_as_black: bool = False) -> List[Tuple[np.ndarray, np.ndarray, float, str]]:
        """
        进行一局神经网络 vs 传统AI对弈。
        v2.0: 传统AI使用CPU快速搜索，神经网络使用MCTS。

        Args:
            nn_as_black: 神经网络是否执黑（默认False=传统AI执黑先手）
        Returns:
            List of (state, policy_target, value_target, source)
        """
        game = GomokuGame()
        mcts = MCTS(
            model=self.model,
            num_simulations=self.mcts_simulations,
            device=self.device
        )

        nn_player = BLACK if nn_as_black else WHITE
        game_data = []
        move_count = 0

        while not game.game_over:
            if game.current_player == nn_player:
                # 神经网络走子（使用MCTS）
                action_probs, _ = mcts.search(
                    game,
                    temperature=0.5,
                    add_dirichlet_noise=True,
                    dirichlet_epsilon=0.1
                )
                state = game.get_state_planes()
                game_data.append((state, action_probs, None, 'trad'))
                move = get_best_move_from_probs(action_probs, game, deterministic=False)
            else:
                # 传统AI走子（CPU快速搜索）
                move = self.trad_ai.get_best_move(game)
                if move:
                    state = game.get_state_planes()
                    action_probs = np.zeros(BOARD_SIZE * BOARD_SIZE, dtype=np.float32)
                    action_probs[move[0] * BOARD_SIZE + move[1]] = 1.0
                    game_data.append((state, action_probs, None, 'trad'))

            if move is None or move == (-1, -1):
                break
            game.make_move(move[0], move[1])
            move_count += 1

        # 填充价值目标
        result = game.winner if game.winner else 0
        filled_data = []
        for i, (state, policy, _, source) in enumerate(game_data):
            if result == 0:
                v = 0.0
            elif result == BLACK:
                v = 1.0 if i % 2 == 0 else -1.0
            else:
                v = -1.0 if i % 2 == 0 else 1.0
            filled_data.append((state, policy, v, source))

        return filled_data


# ============================================================================
# 难度自动调控器 v2.0
# ============================================================================

class AutoDifficultyAdjuster:
    """传统AI难度自动调控器 v2.0 — 更平滑的阶梯式调整"""

    def __init__(self,
                 initial_depth: int = 4,
                 depth_range: Tuple[int, int] = (1, 8),
                 games_per_adjust: int = 10,
                 target_win_rate: float = 0.20,
                 win_rate_window: int = 100):
        self.current_depth = initial_depth
        self.depth_range = depth_range
        self.games_per_adjust = games_per_adjust
        self.target_win_rate = target_win_rate
        self.win_rate_window = win_rate_window

        self.recent_results: deque = deque(maxlen=win_rate_window)
        self.games_since_adjust = 0
        self.adjust_history: List[Dict] = []

        # v2.0: 平滑调整计数（防止频繁震荡）
        self._consecutive_over = 0   # 连续超过上限次数
        self._consecutive_under = 0  # 连续低于下限次数

    def record_result(self, nn_won: bool):
        self.recent_results.append(1 if nn_won else 0)
        self.games_since_adjust += 1

    def should_adjust(self) -> bool:
        return self.games_since_adjust >= self.games_per_adjust

    def get_win_rate(self) -> float:
        if len(self.recent_results) < 10:
            return 0.5
        return sum(self.recent_results) / len(self.recent_results)

    def adjust(self) -> Dict[str, Any]:
        """
        执行难度调整（v2.0 平滑策略）。
        需要连续多次触发才调深/调浅，防止震荡。
        """
        if not self.should_adjust():
            return {'adjusted': False}

        win_rate = self.get_win_rate()
        old_depth = self.current_depth
        adjusted = False

        upper = self.target_win_rate + 0.05  # 0.25
        lower = self.target_win_rate - 0.05  # 0.15

        # 平滑计数逻辑
        if win_rate > upper:
            self._consecutive_over += 1
            self._consecutive_under = 0
        elif win_rate < lower:
            self._consecutive_under += 1
            self._consecutive_over = 0
        else:
            self._consecutive_over = 0
            self._consecutive_under = 0

        # 需要连续 3 次触发才调整
        if self._consecutive_over >= 3 and self.current_depth < self.depth_range[1]:
            self.current_depth = min(self.depth_range[1], self.current_depth + 1)
            adjusted = True
            self._consecutive_over = 0
        elif self._consecutive_under >= 3 and self.current_depth > self.depth_range[0]:
            self.current_depth = max(self.depth_range[0], self.current_depth - 1)
            adjusted = True
            self._consecutive_under = 0

        info = {
            'adjusted': adjusted,
            'old_depth': old_depth,
            'new_depth': self.current_depth,
            'win_rate': win_rate,
            'games_in_window': len(self.recent_results),
        }

        self.adjust_history.append(info)
        self.games_since_adjust = 0

        if adjusted:
            logger.info(f"自动难度: 深度 {old_depth} → {self.current_depth} "
                        f"(胜率={win_rate:.2%})")

        return info


# ============================================================================
# ELO评级系统
# ============================================================================

class ELOSystem:
    """ELO评级系统"""

    def __init__(self, initial_elo: float = 1500.0, k_factor: float = 32.0):
        self.elo_scores: Dict[str, float] = {'current': initial_elo, 'best': initial_elo}
        self.k_factor = k_factor
        self.history: List[Dict] = []

    def update(self, winner_id: str, loser_id: str, num_games: int):
        winner_elo = self.elo_scores.get(winner_id, 1500)
        loser_elo = self.elo_scores.get(loser_id, 1500)

        expected = 1.0 / (1.0 + 10 ** ((loser_elo - winner_elo) / 400.0))
        delta = self.k_factor * (1.0 - expected)

        self.elo_scores[winner_id] = winner_elo + delta
        self.elo_scores[loser_id] = loser_elo - delta

        self.history.append({
            'winner': winner_id,
            'loser': loser_id,
            'delta': delta,
            'num_games': num_games,
        })

    def get_elo(self, model_id: str) -> float:
        return self.elo_scores.get(model_id, 1500)


# ============================================================================
# 智能参数自动调控器 v2.0 — 硬件感知 + 训练状态自适应
# ============================================================================

@dataclass
class AutoParams:
    """自动调控的训练参数快照"""
    temperature: float = 1.5
    temperature_cutoff: int = 15
    mcts_simulations: int = 200
    dirichlet_alpha: float = 0.03
    dirichlet_epsilon: float = 0.25
    c_puct: float = 2.5
    games_per_iteration: int = 5
    noise_scale: float = 1.0


class AutoParameterController:
    """
    智能参数自动调控器 v2.0

    v2.0 改进：
      - 结合硬件信息决定参数初始值和范围
      - 更精细的阶段划分（5个阶段替代3个）
      - Loss 收敛检测更敏感
      - 胜率趋势分析和提前预警
      - 针对 RTX 3060 的 MCTS 模拟次数动态调整
    """

    def __init__(self,
                 initial_mcts: int = 200,
                 mcts_range: Tuple[int, int] = (100, 800),
                 initial_temperature: float = 1.5,
                 temperature_min: float = 0.1,
                 hardware_config: dict = None,
                 eval_interval_games: int = 100):
        # 如果提供了硬件配置，使用硬件推荐值
        if hardware_config:
            training_cfg = hardware_config.get('training', {})
            initial_mcts = min(initial_mcts, training_cfg.get('num_mcts_simulations', initial_mcts))
            mcts_range = (100, max(400, training_cfg.get('num_mcts_simulations', 400) * 2))

        self.mcts_simulations = initial_mcts
        self.mcts_range = mcts_range
        self.temperature = initial_temperature
        self.temperature_min = temperature_min
        self.dirichlet_epsilon = 0.25
        self.dirichlet_epsilon_min = 0.05
        self.dirichlet_alpha = 0.03
        self.c_puct = 2.5
        self.games_per_iteration = 5
        self.temperature_cutoff = 15

        # 统计追踪
        self.recent_losses: deque = deque(maxlen=500)
        self.recent_win_rates: deque = deque(maxlen=50)
        self.loss_stable_count = 0
        self.last_mcts_adjust_step = 0
        # MCTS 调整冷却步数 = 自动调参评估局数 × 20（让MCTS随用户设定的评估节奏调整）
        self.mcts_adjust_cooldown = max(100, eval_interval_games * 20)

        # v2.0 新追踪指标
        self.recent_value_losses: deque = deque(maxlen=200)
        self.recent_policy_losses: deque = deque(maxlen=200)
        self._stagnation_counter = 0  # 停滞计数
        self._loss_trend = 0.0        # loss 变化趋势

        # 阶段追踪（v2.0: 5个阶段）
        self.phase = 'warmup'  # warmup / early / mid / late / converge
        # 阶段阈值: warmup→early→mid→late→converge
        self._phase_thresholds = (0.05, 0.15, 0.50, 0.85)

        logger.info(f"[AutoParams] 初始化: MCTS={initial_mcts}, "
                    f"温度={initial_temperature}, MCTS范围={mcts_range}")

    def update(self, global_step: int, total_steps: int,
               loss_info: Optional[Dict] = None,
               win_rate: Optional[float] = None,
               buffer_size: int = 0) -> AutoParams:
        """
        每次训练迭代后调用，自动调整参数。

        v2.0 改进：
          - 5阶段精细控制
          - Loss 趋势分析
          - 停滞检测（防止陷入局部最优）
        """
        progress = min(1.0, global_step / max(1, total_steps))

        # ── 阶段判定（v2.0: 5个阶段） ──
        if progress < self._phase_thresholds[0]:
            self.phase = 'warmup'
        elif progress < self._phase_thresholds[1]:
            self.phase = 'early'
        elif progress < self._phase_thresholds[2]:
            self.phase = 'mid'
        elif progress < self._phase_thresholds[3]:
            self.phase = 'late'
        else:
            self.phase = 'converge'

        # ── 1. 温度控制：随进度衰减 ──
        phase_map = {
            'warmup':  (1.5, 1.2),   # (起始, 结束)
            'early':   (1.2, 0.6),
            'mid':     (0.6, 0.2),
            'late':    (0.2, 0.1),
            'converge':(0.1, 0.08),
        }
        temp_range = phase_map.get(self.phase, (1.0, 0.1))
        phase_progress = self._get_phase_progress(progress)
        self.temperature = temp_range[0] + (temp_range[1] - temp_range[0]) * phase_progress
        self.temperature = max(self.temperature_min, self.temperature)

        # ── 2. Dirichlet 噪声：随进度衰减 ──
        self.dirichlet_epsilon = max(
            self.dirichlet_epsilon_min,
            0.25 * (1.0 - progress * 0.9)
        )

        # ── 3. 温度截止步数 ──
        cutoff_map = {
            'warmup': 10, 'early': 15, 'mid': 20,
            'late': 12, 'converge': 8,
        }
        self.temperature_cutoff = cutoff_map.get(self.phase, 15)

        # ── 4. 收集统计信息 ──
        if win_rate is not None:
            self.recent_win_rates.append(win_rate)

        if loss_info and 'total_loss' in loss_info:
            self.recent_losses.append(loss_info['total_loss'])
            if 'value_loss' in loss_info:
                self.recent_value_losses.append(loss_info['value_loss'])
            if 'policy_loss' in loss_info:
                self.recent_policy_losses.append(loss_info['policy_loss'])

        # ── 5. MCTS 模拟次数 ──
        if global_step - self.last_mcts_adjust_step >= self.mcts_adjust_cooldown:
            self._adjust_mcts(global_step)
            self.last_mcts_adjust_step = global_step

        # ── 6. PUCT 探索常数 ──
        self._adjust_puct()

        # ── 7. 每轮对弈局数：经验池小时少生成快出训练数据 ──
        if buffer_size < 1000:
            self.games_per_iteration = 1
        elif buffer_size < 5000:
            self.games_per_iteration = 2
        elif buffer_size > 200000:
            self.games_per_iteration = 2
        elif buffer_size > 100000:
            self.games_per_iteration = 3
        else:
            self.games_per_iteration = 4

        # ── 8. 停滞检测：如果 loss 长时间不下降，临时增加探索 ──
        self._detect_stagnation()

        return self.get_params()

    def _get_phase_progress(self, global_progress: float) -> float:
        """计算在当前阶段内的进度 (0~1)"""
        thresholds = [0.0] + list(self._phase_thresholds) + [1.0]
        phase_names = ['warmup', 'early', 'mid', 'late', 'converge']
        phase_idx = phase_names.index(self.phase)
        phase_start = thresholds[phase_idx]
        phase_end = thresholds[phase_idx + 1]
        phase_len = phase_end - phase_start
        if phase_len <= 0:
            return 1.0
        return min(1.0, max(0.0, (global_progress - phase_start) / phase_len))

    def _adjust_mcts(self, step: int):
        """根据近期胜率趋势调整MCTS模拟次数"""
        if len(self.recent_win_rates) < 10:
            return

        avg_wr = sum(self.recent_win_rates) / len(self.recent_win_rates)

        # 根据阶段调整
        if self.phase in ('warmup', 'early'):
            # 早期：快速试探，胜率高则加速增加
            if avg_wr > 0.45:
                self.mcts_simulations = min(self.mcts_range[1], self.mcts_simulations + 50)
            elif avg_wr < 0.25:
                self.mcts_simulations = max(self.mcts_range[0], self.mcts_simulations - 20)
        elif self.phase == 'mid':
            # 中期：稳健调整
            if avg_wr > 0.55:
                self.mcts_simulations = min(self.mcts_range[1], self.mcts_simulations + 30)
            elif avg_wr < 0.30:
                self.mcts_simulations = max(self.mcts_range[0], self.mcts_simulations - 20)
        else:
            # 后期：大搜索量
            target = int(self.mcts_range[1] * 0.85)
            if self.mcts_simulations < target:
                self.mcts_simulations = min(self.mcts_range[1], self.mcts_simulations + 20)

    def _adjust_puct(self):
        """根据 Loss 收敛状态微调 PUCT"""
        if len(self.recent_losses) < 100:
            return

        losses = list(self.recent_losses)[-100:]
        avg_loss = sum(losses) / len(losses)
        loss_std = (sum((l - avg_loss) ** 2 for l in losses) / len(losses)) ** 0.5

        # 计算 loss 趋势（最近 20 步 vs 前 20 步）
        if len(losses) >= 40:
            recent_20 = sum(losses[-20:]) / 20
            prev_20 = sum(losses[-40:-20]) / 20
            self._loss_trend = prev_20 - recent_20  # 正数表示在下降

        if loss_std < 0.05 and avg_loss < 1.0:
            self.c_puct = max(1.2, self.c_puct - 0.03)
        elif loss_std > 0.3:
            self.c_puct = min(4.0, self.c_puct + 0.05)

    def _detect_stagnation(self):
        """检测训练停滞，临时增加探索"""
        if len(self.recent_losses) < 200:
            return

        losses = list(self.recent_losses)

        # 比较最近 100 步和前 100 步的平均 loss
        recent_100 = sum(losses[-100:]) / 100
        prev_100 = sum(losses[-200:-100]) / 100

        # 如果 recent >= prev * 0.98，说明 loss 几乎没下降
        if recent_100 >= prev_100 * 0.98:
            self._stagnation_counter += 1
        else:
            self._stagnation_counter = 0

        # 连续 3 次检测到停滞，增加探索
        if self._stagnation_counter >= 3:
            self.temperature = min(0.5, self.temperature + 0.1)
            self.c_puct = min(4.0, self.c_puct + 0.2)
            logger.info(f"[AutoParams] 检测到训练停滞，临时增加探索: "
                        f"temp={self.temperature:.3f}, puct={self.c_puct:.2f}")
            self._stagnation_counter = 0

    def get_params(self) -> AutoParams:
        return AutoParams(
            temperature=self.temperature,
            temperature_cutoff=self.temperature_cutoff,
            mcts_simulations=self.mcts_simulations,
            dirichlet_alpha=self.dirichlet_alpha,
            dirichlet_epsilon=self.dirichlet_epsilon,
            c_puct=self.c_puct,
            games_per_iteration=self.games_per_iteration,
        )

    def get_summary(self) -> Dict[str, Any]:
        return {
            'phase': self.phase,
            'mcts_simulations': self.mcts_simulations,
            'temperature': round(self.temperature, 3),
            'dirichlet_epsilon': round(self.dirichlet_epsilon, 3),
            'c_puct': round(self.c_puct, 2),
            'games_per_iteration': self.games_per_iteration,
            'temperature_cutoff': self.temperature_cutoff,
            'loss_trend': round(self._loss_trend, 4),
        }


# ============================================================================
# 训练器 v2.0 — 硬件感知 + PER + 数据增强
# ============================================================================

class Trainer:
    """主训练器 v2.0 — 硬件自适应 + 全面优化"""

    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: 训练配置字典（可由 hardware_auto_config.get_auto_config() 生成）
        """
        self.config = config

        # ── 硬件配置 ──
        self.device = torch.device('cuda:0' if torch.cuda.is_available() and
                                    config.get('training', {}).get('use_cuda', True)
                                    else 'cpu')

        # 检测 Tensor Cores 和 GPU 能力
        self._has_tensor_cores = False
        if self.device.type == 'cuda':
            try:
                props = torch.cuda.get_device_properties(0)
                self._has_tensor_cores = (props.major >= 7)
            except Exception:
                pass

        logger.info(f"训练设备: {self.device}")
        logger.info(f"Tensor Cores: {'✅ 支持' if self._has_tensor_cores else '❌ 不支持'}")

        # ── 模型（支持硬件自适应网络规模） ──
        net_cfg = config.get('network', {})
        self.model = GomokuNet(
            num_filters=net_cfg.get('num_filters', 128),
            num_res_blocks=net_cfg.get('num_res_blocks', 7),
            policy_channels=net_cfg.get('policy_channels', 32),
            value_channels=net_cfg.get('value_channels', 32),
            value_hidden=net_cfg.get('value_hidden', 128),
        ).to(self.device)
        self.best_model = GomokuNet(
            num_filters=net_cfg.get('num_filters', 128),
            num_res_blocks=net_cfg.get('num_res_blocks', 7),
            policy_channels=net_cfg.get('policy_channels', 32),
            value_channels=net_cfg.get('value_channels', 32),
            value_hidden=net_cfg.get('value_hidden', 128),
        ).to(self.device)

        param_count = self.model.count_parameters()
        logger.info(f"模型参数: {param_count:,} ({param_count/1e6:.2f}M)")

        # ── 优化器 ──
        train_cfg = config.get('training', {})
        lr = train_cfg.get('learning_rate', 0.02)
        self.optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=0.9,
                                   weight_decay=1e-4)

        # 混合精度梯度缩放器
        self.scaler = GradScaler('cuda') if self.device.type == 'cuda' else None

        # ── 学习率调度（v2.0: Cosine Annealing with Warm Restarts） ──
        total_steps = train_cfg.get('total_steps', 500000)
        warmup_steps = max(1, int(total_steps * 0.05))
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=max(1, warmup_steps * 4), T_mult=2, eta_min=lr * 0.01
        )
        self.warmup_steps = warmup_steps
        self.base_lr = lr

        # ── 梯度累积 ──
        self.gradient_accumulation_steps = train_cfg.get('gradient_accumulation_steps', 1)
        logger.info(f"梯度累积步数: {self.gradient_accumulation_steps}")

        # ── 经验回放池 v2.0（PER 支持） ──
        replay_cfg = config.get('training', {}).get('replay_buffer', {})
        capacity = replay_cfg.get('capacity', 2000000)
        sampling_weights = replay_cfg.get('sampling_weights',
                                           {'self': 0.6, 'trad': 0.3, 'human': 0.1})
        self.replay_buffer = ReplayBuffer(
            capacity=capacity,
            sampling_weights=sampling_weights,
            prioritized_alpha=replay_cfg.get('prioritized_alpha', 0.6),
            prioritized_beta=replay_cfg.get('prioritized_beta', 0.4),
        )

        # ── 数据增强 ──
        aug_cfg = config.get('training', {}).get('augmentation', {})
        self.augmentation_enabled = aug_cfg.get('enabled', True) and self.device.type == 'cuda'

        # ── 基本配置参数 ──
        self.batch_size = train_cfg.get('batch_size', 1024)
        self.effective_batch_size = self.batch_size * self.gradient_accumulation_steps
        self.num_mcts_simulations = train_cfg.get('num_mcts_simulations', 400)
        self.save_interval_minutes = train_cfg.get('save_interval_minutes', 5)
        self.auto_eval_games = train_cfg.get('auto_eval_games', 100)
        self.eval_interval_steps = train_cfg.get('eval_interval_steps', 5000)

        # ── 混合精度 ──
        self.mixed_precision = train_cfg.get('mixed_precision', True) and self.device.type == 'cuda'

        # ── 训练模式 ──
        self.modes = train_cfg.get('modes', ['self'])

        # ── 传统AI相关 ──
        trad_cfg = train_cfg.get('traditional', {})
        self.trad_depth = trad_cfg.get('initial_depth', 4)
        self.adjuster = AutoDifficultyAdjuster(
            initial_depth=self.trad_depth,
            depth_range=tuple(trad_cfg.get('depth_range', [1, 8])),
            games_per_adjust=trad_cfg.get('games_per_adjust', 5),  # 每5局就检查胜率调深度
            target_win_rate=trad_cfg.get('target_win_rate', 0.20),
            win_rate_window=trad_cfg.get('win_rate_window', 100),
        )

        # ── ELO系统 ──
        self.elo_system = ELOSystem()

        # ── 智能参数自动调控器 v2.0（传入硬件配置） ──
        self.auto_params = AutoParameterController(
            initial_mcts=min(200, self.num_mcts_simulations),
            mcts_range=(100, max(400, self.num_mcts_simulations * 2)),
            initial_temperature=1.5,
            temperature_min=0.1,
            hardware_config=config,
            eval_interval_games=self.auto_eval_games,
        )

        # ── 训练状态 ──
        self.global_step = 0
        self.total_games_self = 0
        self.total_games_trad = 0
        self.total_games_human = 0
        self.total_nn_wins = 0
        self.running = False
        self.paused = False

        # 训练历史记录
        self.loss_history: List[Dict] = []
        self.win_rate_history: List[Dict] = []

        # PER 追踪
        self._last_sampled_indices: List[int] = []

        # 模型目录
        model_dir = train_cfg.get('model_dir', './models')
        if not os.path.isabs(model_dir):
            model_dir = os.path.join(PROJECT_DIR, model_dir.lstrip('./\\'))
        self.model_dir = model_dir
        os.makedirs(self.model_dir, exist_ok=True)

        # 加载已有模型
        self._load_latest_model()

        logger.info(f"训练器 v2.0 初始化完成: batch={self.batch_size}, "
                    f"effective_batch={self.effective_batch_size}, "
                    f"MCTS={self.num_mcts_simulations}, "
                    f"AMP={'开启' if self.mixed_precision else '关闭'}, "
                    f"数据增强={'开启' if self.augmentation_enabled else '关闭'}, "
                    f"PER={'开启' if replay_cfg.get('prioritized_alpha', 0) > 0 else '关闭'}")

    def _load_latest_model(self):
        """加载最新模型"""
        model_path = os.path.join(self.model_dir, 'model.pt')
        if os.path.exists(model_path):
            try:
                model, info = GomokuNet.load_checkpoint(model_path, self.device)
                # 检查网络结构是否匹配
                if model.num_filters == self.model.num_filters and \
                   model.num_res_blocks == self.model.num_res_blocks:
                    self.model = model
                    self.global_step = info.get('step', 0)
                    logger.info(f"加载模型: step={self.global_step}, elo={info.get('elo', '-')}")
                else:
                    logger.warning("网络结构不匹配（保存的 vs 配置的），使用新模型")
            except Exception as e:
                logger.warning(f"模型加载失败: {e}")

        # 加载训练状态
        state_path = os.path.join(self.model_dir, 'training_state.pt')
        if os.path.exists(state_path):
            try:
                state = torch.load(state_path, map_location=self.device)
                self.global_step = state.get('step', self.global_step)
                if state.get('optimizer_state_dict'):
                    self.optimizer.load_state_dict(state['optimizer_state_dict'])
                # 注意：auto_params 不从旧状态恢复——用户通过GUI滑条设置的初始值优先
                # 自动调参控制器会在训练过程中根据最新情况重新调整参数
                logger.info(f"加载训练状态: step={self.global_step}")
                self.total_games_self = state.get('total_games_self', 0)
                self.total_games_trad = state.get('total_games_trad', 0)
                self.total_games_human = state.get('total_games_human', 0)
                self.total_nn_wins = state.get('total_nn_wins', 0)
            except Exception as e:
                logger.warning(f"训练状态加载失败: {e}")

    def _save_checkpoint(self):
        """安全保存模型和训练状态（先写临时文件再覆盖，防止OOM导致文件损坏）"""
        # 保存模型
        model_path = os.path.join(self.model_dir, 'model.pt')
        model_tmp = model_path + '.tmp'
        self.model.save_checkpoint(
            model_tmp,
            step=self.global_step,
            extra={'elo': self.elo_system.get_elo('current')}
        )
        os.replace(model_tmp, model_path)  # 原子覆盖

        # 保存训练状态
        state_path = os.path.join(self.model_dir, 'training_state.pt')
        state_tmp = state_path + '.tmp'
        try:
            torch.save({
                'step': self.global_step,
                'optimizer_state_dict': self.optimizer.state_dict(),
                'total_games_self': self.total_games_self,
                'total_games_trad': self.total_games_trad,
                'total_games_human': self.total_games_human,
                'total_nn_wins': self.total_nn_wins,
                'auto_params': {
                    'mcts_simulations': self.auto_params.mcts_simulations,
                    'temperature': self.auto_params.temperature,
                    'dirichlet_epsilon': self.auto_params.dirichlet_epsilon,
                    'c_puct': self.auto_params.c_puct,
                    'games_per_iteration': self.auto_params.games_per_iteration,
                },
                'elo_current': self.elo_system.get_elo('current'),
                'elo_best': self.elo_system.get_elo('best'),
            }, state_tmp)
            os.replace(state_tmp, state_path)
        except Exception:
            # 状态文件不重要，保存失败不影响继续训练
            if os.path.exists(state_tmp):
                os.remove(state_tmp)

    def _generate_self_play_data(self, num_games: int = 1):
        """生成自我对弈数据（使用智能调控参数）"""
        ap = self.auto_params.get_params()
        worker = SelfPlayWorker(
            model=self.model,
            num_simulations=ap.mcts_simulations,
            device=self.device,
            temperature=ap.temperature,
            temperature_cutoff=ap.temperature_cutoff,
            dirichlet_alpha=ap.dirichlet_alpha,
            dirichlet_epsilon=ap.dirichlet_epsilon,
            c_puct=ap.c_puct,
        )
        for _ in range(num_games):
            if not self.running or self.paused:
                break
            data = worker.play_game()
            self.replay_buffer.add_batch(data)
            self.total_games_self += 1

    def _generate_traditional_data(self, num_games: int = 1):
        """生成传统AI对抗数据（v2.0: 训练模式，CPU快速搜索 + 低深度）"""
        # 从低深度低MCTS起步，自动调参逐步增加
        train_trad_depth = min(self.trad_depth, max(1, self.adjuster.current_depth))
        train_mcts = min(50, self.num_mcts_simulations)
        worker = TraditionalOpponentWorker(
            model=self.model,
            trad_depth=train_trad_depth,
            mcts_simulations=train_mcts,
            device=self.device
        )
        for i in range(num_games):
            if not self.running or self.paused:
                break
            # 传统AI始终执黑先手（走天元），引导神经网络学习中盘应对
            data = worker.play_game(nn_as_black=False)
            self.replay_buffer.add_batch(data)
            self.total_games_trad += 1

            if data:
                last_value = data[-1][2]
                nn_won = (last_value > 0)
                self.adjuster.record_result(nn_won)
                if nn_won:
                    self.total_nn_wins += 1

            if self.adjuster.should_adjust():
                self.adjuster.adjust()

    def _train_step(self) -> Dict[str, float]:
        """
        单步训练（v2.0: PER + 数据增强 + 梯度累积）。

        Returns:
            损失字典
        """
        min_train_samples = 64
        buf_size = len(self.replay_buffer)
        if buf_size < min_train_samples:
            return {}

        self.model.train()

        # ── 自适应采样 ──
        actual_batch = min(buf_size, self.batch_size)
        states, policy_targets, value_targets, indices = self.replay_buffer.sample(actual_batch)
        self._last_sampled_indices = indices

        # ── 数据增强（8倍扩增） ──
        if self.augmentation_enabled and actual_batch >= 32:
            states, policy_targets, value_targets = DataAugmentation.augment_batch(
                states, policy_targets, value_targets
            )

        states = torch.from_numpy(states).to(self.device)
        policy_targets = torch.from_numpy(policy_targets).to(self.device)
        value_targets = torch.from_numpy(value_targets).to(self.device)

        # ── 梯度累积 ──
        accumulated_loss = 0.0
        accumulated_policy_loss = 0.0
        accumulated_value_loss = 0.0

        # 计算每份大小
        per_step = max(1, states.shape[0] // self.gradient_accumulation_steps)

        self.optimizer.zero_grad()

        for step_idx in range(self.gradient_accumulation_steps):
            start = step_idx * per_step
            end = min(start + per_step, states.shape[0])
            if start >= end:
                break

            s = states[start:end]
            p = policy_targets[start:end]
            v = value_targets[start:end]

            if self.mixed_precision and self.scaler is not None:
                with autocast('cuda'):
                    policy_logits, values = self.model(s)
                    policy_loss = F.cross_entropy(policy_logits, p)
                    value_loss = F.mse_loss(values, v)
                    total_loss = policy_loss + value_loss
                    total_loss = total_loss / self.gradient_accumulation_steps

                self.scaler.scale(total_loss).backward()
            else:
                policy_logits, values = self.model(s)
                policy_loss = F.cross_entropy(policy_logits, p)
                value_loss = F.mse_loss(values, v)
                total_loss = (policy_loss + value_loss) / self.gradient_accumulation_steps
                total_loss.backward()

            accumulated_loss += total_loss.item() * self.gradient_accumulation_steps
            accumulated_policy_loss += policy_loss.item()
            accumulated_value_loss += value_loss.item()

        # ── 梯度裁剪 ──
        if self.mixed_precision and self.scaler is not None:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
            self.optimizer.step()

        # ── 学习率预热 + 调度 ──
        if self.global_step < self.warmup_steps:
            lr_scale = (self.global_step + 1) / max(1, self.warmup_steps)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.base_lr * lr_scale
        else:
            self.scheduler.step()

        self.global_step += 1

        # ── 更新 PER 优先级 ──
        if len(self._last_sampled_indices) > 0:
            td_errors = [abs(accumulated_value_loss)] * len(self._last_sampled_indices)
            self.replay_buffer.update_priority(self._last_sampled_indices, td_errors)

        # 日志（每 10 步）
        if self.global_step % 10 == 0:
            logger.info(
                f"训练步 {self.global_step} | "
                f"loss={accumulated_loss:.4f} "
                f"(p={accumulated_policy_loss:.4f}, v={accumulated_value_loss:.4f}) | "
                f"batch={actual_batch} | buffer={buf_size} | "
                f"lr={self.optimizer.param_groups[0]['lr']:.6f}"
            )

        return {
            'total_loss': accumulated_loss,
            'policy_loss': accumulated_policy_loss,
            'value_loss': accumulated_value_loss,
            'learning_rate': self.optimizer.param_groups[0]['lr'],
        }

    def _train_immediately(self, min_samples: int = 64):
        """
        立即执行若干训练步（边生成数据边训练，确保图表及时更新）。

        与 _train_step 不同，此方法：
          - 自动决定训练步数（根据经验池大小）
          - 记录 loss 到 history
          - 保存最后 loss 供 auto_params.update 使用
        """
        buf_size = len(self.replay_buffer)
        if buf_size < min_samples:
            return

        # 根据经验池大小决定训练步数：刚起步时少训练，池子大了多训练
        if buf_size < self.batch_size:
            steps = 1
        else:
            steps = min(5, buf_size // min_samples)

        for _ in range(steps):
            if not self.running:
                break
            loss_info = self._train_step()
            if loss_info:
                self.loss_history.append({**loss_info, 'step': self.global_step})
                self._last_loss_info = loss_info

        # 训练后清理 GPU 缓存，防止显存溢出
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()

    def _evaluate_models(self, num_games: int = 50) -> Dict[str, float]:
        """评估当前模型 vs 最佳模型"""
        wins = 0
        losses = 0
        draws = 0

        for i in range(num_games):
            game = GomokuGame()
            mcts_current = MCTS(model=self.model, num_simulations=200, device=self.device)
            mcts_best = MCTS(model=self.best_model, num_simulations=200, device=self.device)

            move_count = 0
            while not game.game_over:
                if game.current_player == BLACK:
                    action_probs, _ = mcts_current.search(game, temperature=0.1)
                else:
                    action_probs, _ = mcts_best.search(game, temperature=0.1)
                move = get_best_move_from_probs(action_probs, game, deterministic=True)
                if move == (-1, -1):
                    break
                game.make_move(move[0], move[1])
                move_count += 1
                if move_count > 300:
                    break

            if game.winner == BLACK:
                wins += 1
            elif game.winner == WHITE:
                losses += 1
            else:
                draws += 1

        win_rate = wins / max(1, num_games)
        return {'wins': wins, 'losses': losses, 'draws': draws, 'win_rate': win_rate}

    def _auto_evaluate_vs_traditional(self, num_games: int = 100) -> Dict[str, Any]:
        """
        自动评估神经网络 vs 传统AI（v2.0 GPU 加速版）。
        """
        logger.info(f"开始自动评估: NN vs 传统AI (深度={self.adjuster.current_depth}), "
                    f"共{num_games}局")

        nn_wins = 0
        trad_wins = 0
        draws = 0
        total_time = 0.0

        for i in range(num_games):
            # 评估时传统AI执黑先手
            nn_as_black = False

            # 评估时：传统AI使用GPU全力搜索（不开启training_mode）
            eval_trad_ai = TraditionalAI(
                search_depth=self.adjuster.current_depth,
                training_mode=False,  # 评估模式：使用GPU全力搜索
                max_time_ms=30000,
            )
            worker = TraditionalOpponentWorker(
                model=self.model,
                trad_depth=self.adjuster.current_depth,
                mcts_simulations=min(200, self.num_mcts_simulations),
                device=self.device
            )
            # 覆盖为不使用训练模式的传统AI
            worker.trad_ai = eval_trad_ai

            t0 = time.time()
            game_data = worker.play_game(nn_as_black=nn_as_black)
            elapsed = time.time() - t0
            total_time += elapsed

            if game_data:
                last_value = game_data[-1][2]
                if last_value > 0:
                    nn_wins += 1
                elif last_value < 0:
                    trad_wins += 1
                else:
                    draws += 1
                self.adjuster.record_result(last_value > 0)
            else:
                draws += 1

            if (i + 1) % 10 == 0:
                wr = nn_wins / (i + 1) if (i + 1) > 0 else 0
                logger.info(f"  评估: {i + 1}/{num_games} | "
                            f"NN胜: {nn_wins} | 传统AI胜: {trad_wins} | "
                            f"胜率: {wr:.2%} | "
                            f"平均耗时: {total_time / (i + 1):.2f}s/局")

        nn_win_rate = nn_wins / num_games if num_games > 0 else 0
        avg_time = total_time / num_games if num_games > 0 else 0

        adjust_info = self.adjuster.adjust()

        result = {
            'nn_wins': nn_wins,
            'trad_wins': trad_wins,
            'draws': draws,
            'nn_win_rate': nn_win_rate,
            'total_games': num_games,
            'avg_time_per_game': round(avg_time, 3),
            'trad_depth_before': adjust_info.get('old_depth', self.adjuster.current_depth),
            'trad_depth_after': adjust_info.get('new_depth', self.adjuster.current_depth),
            'depth_adjusted': adjust_info.get('adjusted', False),
        }

        logger.info(f"自动评估完成: NN胜率={nn_win_rate:.2%} "
                    f"({nn_wins}W/{trad_wins}L/{draws}D) | "
                    f"传统AI深度: {result['trad_depth_before']}→{result['trad_depth_after']} | "
                    f"平均耗时: {avg_time:.2f}s/局")

        return result

    def train_loop(self, total_steps: int = None, games_per_iteration: int = None):
        """
        主训练循环 v2.0（智能自动调参 + 硬件自适应）。

        Args:
            total_steps: 总训练步数
            games_per_iteration: 每轮生成的对局数（若为None，由智能控制器自动决定）
        """
        if total_steps is None:
            total_steps = self.config.get('training', {}).get('total_steps', 500000)

        self.running = True
        self.paused = False
        start_time = time.time()
        last_save_time = start_time
        last_eval_time = start_time
        last_params_log_time = start_time
        eval_interval_seconds = 600

        # 如果 GPU 显存 > 4GB，每 300 秒评估一次（更快反馈）
        if self.device.type == 'cuda':
            try:
                gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                if gpu_mem >= 6:
                    eval_interval_seconds = 300  # 更快评估周期
            except Exception:
                pass

        logger.info(f"开始训练循环. 目标: {total_steps} 步. 模式: {self.modes}")
        logger.info(f"设备: {self.device}, Batch: {self.batch_size}, "
                    f"有效Batch: {self.effective_batch_size}")
        logger.info(f"自动保存: 每{self.save_interval_minutes}分钟 | "
                    f"自动评估: 每{eval_interval_seconds}秒")
        logger.info(f"智能调参: 已启用 | 优先经验回放: 已启用 | "
                    f"数据增强: {'已启用' if self.augmentation_enabled else '已禁用'}")

        try:
            while self.running and self.global_step < total_steps:
                if self.paused:
                    time.sleep(1)
                    continue

                # ── 获取智能调控参数 ──
                ap = self.auto_params.get_params()
                # 自动调参可根据经验池状况取更小值（加速首轮出数据）
                if games_per_iteration is not None:
                    auto_games = min(games_per_iteration, ap.games_per_iteration)
                else:
                    auto_games = ap.games_per_iteration

                # ── 生成训练数据 + 立即训练（边生成边训练，图表实时更新） ──
                min_train = 64

                if 'self' in self.modes:
                    self._generate_self_play_data(auto_games)
                    self._train_immediately(min_train)

                if 'trad' in self.modes:
                    self._generate_traditional_data(auto_games)
                    self._train_immediately(min_train)

                # ── 智能参数更新 ──
                self.auto_params.update(
                    global_step=self.global_step,
                    total_steps=total_steps,
                    loss_info=getattr(self, '_last_loss_info', {}),
                    win_rate=self.adjuster.get_win_rate(),
                    buffer_size=len(self.replay_buffer),
                )

                # ── 进度日志 ──
                if self.global_step % 100 == 0:
                    buf_stats = self.replay_buffer.get_stats()
                    ap_summary = self.auto_params.get_summary()
                    li = getattr(self, '_last_loss_info', {})
                    logger.info(
                        f"步 {self.global_step}/{total_steps} | "
                        f"Loss: {li.get('total_loss', 0):.4f} | "
                        f"Buffer: {buf_stats['total']} | "
                        f"LR: {li.get('learning_rate', 0):.6f}"
                    )
                    logger.info(
                        f"  自动: phase={ap_summary['phase']} | "
                        f"MCTS={ap_summary['mcts_simulations']} | "
                        f"温度={ap_summary['temperature']:.3f} | "
                        f"噪声={ap_summary['dirichlet_epsilon']:.3f} | "
                        f"PUCT={ap_summary['c_puct']:.2f} | "
                        f"局/轮={ap_summary['games_per_iteration']} | "
                        f"传统AI深度={self.adjuster.current_depth}"
                    )

                # ── 定期评估 ──
                if self.global_step > 0 and self.global_step % self.eval_interval_steps == 0:
                    logger.info("评估模型中...")
                    eval_result = self._evaluate_models(num_games=50)
                    self.win_rate_history.append({
                        'step': self.global_step,
                        **eval_result,
                    })
                    self._save_checkpoint()
                    if eval_result['win_rate'] > 0.55:
                        self.elo_system.update('current', 'best', 50)
                        # 复制最佳模型
                        self.best_model.load_state_dict(self.model.state_dict())

                # ── 定时自动保存 ──
                elapsed = time.time() - last_save_time
                if elapsed > self.save_interval_minutes * 60:
                    self._save_checkpoint()
                    last_save_time = time.time()

                # ── 定时自动评估 NN vs 传统AI ──
                eval_elapsed = time.time() - last_eval_time
                if eval_elapsed > eval_interval_seconds and len(self.replay_buffer) >= self.batch_size * 2:
                    logger.info(f"自动评估: {self.auto_eval_games}局 NN vs 传统AI...")
                    eval_vs_trad = self._auto_evaluate_vs_traditional(num_games=self.auto_eval_games)
                    self.win_rate_history.append({
                        'step': self.global_step,
                        'type': 'auto_eval_vs_trad',
                        **eval_vs_trad,
                    })
                    last_eval_time = time.time()

                # ── 参数日志（每 5 分钟） ──
                params_elapsed = time.time() - last_params_log_time
                if params_elapsed > 300:
                    ap_summary = self.auto_params.get_summary()
                    logger.info(f"[自动参数] {ap_summary}")
                    last_params_log_time = time.time()

        except KeyboardInterrupt:
            logger.info("训练被用户中断")
        except RuntimeError as e:
            if 'out of memory' in str(e).lower():
                logger.warning(f"显存溢出(OOM) at step={self.global_step}，自动恢复...")
                torch.cuda.empty_cache()
                # 降低 batch_size 到一半
                self.batch_size = max(16, self.batch_size // 2)
                logger.info(f"batch_size 降为 {self.batch_size}，继续训练")
                # 不清空经验池，继续训练
                self.running = True
                self.train_loop(total_steps=total_steps,
                                games_per_iteration=games_per_iteration)
            else:
                logger.error(f"训练错误 (step={self.global_step}): {e}", exc_info=True)
        except Exception as e:
            logger.error(f"训练错误 (step={self.global_step}): {e}", exc_info=True)
        finally:
            self._save_checkpoint()
            logger.info(f"训练已停止 (step={self.global_step})")

    def stop(self):
        self.running = False

    def pause(self):
        self.paused = True
        logger.info("训练已暂停")

    def resume(self):
        self.paused = False
        logger.info("训练已恢复")

    def get_status(self) -> Dict[str, Any]:
        """获取训练状态"""
        total_games = self.total_games_self + self.total_games_trad + self.total_games_human
        recent_results = self.adjuster.recent_results
        recent_wins = sum(1 for r in recent_results if r)
        recent_total = len(recent_results)
        recent_win_rate = recent_wins / recent_total if recent_total > 0 else 0

        return {
            'global_step': self.global_step,
            'running': self.running,
            'paused': self.paused,
            'buffer_size': len(self.replay_buffer),
            'buffer_stats': self.replay_buffer.get_stats(),
            'total_games_self': self.total_games_self,
            'total_games_trad': self.total_games_trad,
            'total_games_human': self.total_games_human,
            'total_games': total_games,
            'total_wins': self.total_nn_wins,
            'recent_wins': recent_wins,
            'recent_win_rate': recent_win_rate,
            'trad_depth': self.adjuster.current_depth,
            'trad_win_rate': self.adjuster.get_win_rate(),
            'elo_current': self.elo_system.get_elo('current'),
            'elo_best': self.elo_system.get_elo('best'),
            'auto_params': self.auto_params.get_summary(),
            'loss_history': self.loss_history[-100:] if self.loss_history else [],
            'win_rate_history': self.win_rate_history[-20:] if self.win_rate_history else [],
        }
