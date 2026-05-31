"""
训练系统模块
支持三种训练模式：自我对弈、传统AI对抗、人机对弈
混合经验回放池 + 可配置采样权重
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
from typing import Tuple, List, Optional, Dict, Any
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
# 经验回放池
# ============================================================================

@dataclass
class Experience:
    """单条经验数据"""
    state: np.ndarray          # (4, 15, 15)
    policy_target: np.ndarray  # (225,) 策略目标
    value_target: float        # 价值目标 [-1, 1]
    source: str                # 来源: 'self', 'trad', 'human'


class ReplayBuffer:
    """经验回放池 - 基于循环缓冲区的多来源经验存储"""

    def __init__(self, capacity: int = 2000000, sampling_weights: Dict[str, float] = None):
        """
        Args:
            capacity: 最大容量
            sampling_weights: 各来源采样权重
        """
        self.capacity = capacity
        self.buffer: deque = deque(maxlen=capacity)
        self.lock = threading.Lock()
        self.sampling_weights = sampling_weights or {'self': 0.6, 'trad': 0.3, 'human': 0.1}

        # 按来源统计
        self.source_counts: Dict[str, int] = {'self': 0, 'trad': 0, 'human': 0}

    def add(self, state: np.ndarray, policy_target: np.ndarray,
            value_target: float, source: str):
        """添加经验数据"""
        exp = Experience(
            state=state.astype(np.float32),
            policy_target=policy_target.astype(np.float32),
            value_target=value_target,
            source=source
        )
        with self.lock:
            self.buffer.append(exp)
            if source in self.source_counts:
                self.source_counts[source] += 1

    def add_batch(self, experiences: List[Tuple[np.ndarray, np.ndarray, float, str]]):
        """批量添加经验"""
        with self.lock:
            for state, policy, value, source in experiences:
                exp = Experience(
                    state=state.astype(np.float32),
                    policy_target=policy.astype(np.float32),
                    value_target=value,
                    source=source
                )
                self.buffer.append(exp)
                if source in self.source_counts:
                    self.source_counts[source] += 1

    def sample(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        按采样权重采样批次数据

        Returns:
            states: (batch_size, 4, 15, 15)
            policy_targets: (batch_size, 225)
            value_targets: (batch_size, 1)
        """
        with self.lock:
            if len(self.buffer) == 0:
                return (np.zeros((batch_size, 4, 15, 15), dtype=np.float32),
                        np.zeros((batch_size, 225), dtype=np.float32),
                        np.zeros((batch_size, 1), dtype=np.float32))

            # 按来源分类
            source_exps: Dict[str, List[Experience]] = {}
            for exp in self.buffer:
                if exp.source not in source_exps:
                    source_exps[exp.source] = []
                source_exps[exp.source].append(exp)

            # 按权重采样
            sampled = []
            total_weight = sum(self.sampling_weights.get(s, 0) for s in source_exps)
            if total_weight == 0:
                total_weight = 1.0

            for source, exps in source_exps.items():
                weight = self.sampling_weights.get(source, 0)
                if weight > 0 and exps:
                    n = max(1, int(batch_size * weight / total_weight))
                    if len(exps) >= n:
                        sampled.extend(random.sample(exps, n))
                    else:
                        sampled.extend(exps)

            # 如果采样不足，随机补齐
            while len(sampled) < batch_size and len(self.buffer) > 0:
                sampled.append(random.choice(list(self.buffer)))

            # 截断到batch_size
            if len(sampled) > batch_size:
                sampled = random.sample(sampled, batch_size)

            # 组装批次
            states = np.stack([exp.state for exp in sampled])
            policy_targets = np.stack([exp.policy_target for exp in sampled])
            value_targets = np.array([[exp.value_target] for exp in sampled], dtype=np.float32)

            return states, policy_targets, value_targets

    def __len__(self) -> int:
        return len(self.buffer)

    def get_stats(self) -> Dict[str, Any]:
        """获取经验池统计信息"""
        return {
            'total': len(self.buffer),
            'source_counts': self.source_counts.copy(),
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
                    source=str(item['source'])
                )
                self.buffer.append(exp)
                if exp.source in self.source_counts:
                    self.source_counts[exp.source] += 1


# ============================================================================
# 数据增强
# ============================================================================

class DataAugmentation:
    """数据增强：随机旋转和翻转"""

    @staticmethod
    def augment(state: np.ndarray, policy: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        生成所有8种对称变换
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
            # 智能温度：前N步使用探索温度，之后用极小温度
            temperature = self.temperature if move_count < self.temperature_cutoff else 0.1

            action_probs, _ = mcts.search(
                game,
                temperature=temperature,
                add_dirichlet_noise=True,
                dirichlet_alpha=self.dirichlet_alpha,
                dirichlet_epsilon=self.dirichlet_epsilon
            )

            # 保存状态和MCTS策略目标
            state = game.get_state_planes()
            game_data.append((state, action_probs, None, 'self'))  # value稍后填充

            # 选择落子
            move = get_best_move_from_probs(action_probs, game, deterministic=False)
            if move == (-1, -1):
                break
            game.make_move(move[0], move[1])
            move_count += 1

        # 根据最终结果填充价值目标
        result = game.winner if game.winner else 0
        filled_data = []
        for state, policy, _, source in game_data:
            if result == BLACK:
                # 黑棋赢，检查该状态是哪方
                # 从state planes判断：plane[3]全1表示当前玩家视角
                # 简化处理：交替赋值
                pass
            filled_data.append((state, policy, 0.0, source))

        # 根据结果交替赋值价值
        for i in range(len(filled_data)):
            state, policy, _, source = filled_data[i]
            # 从终局视角：黑棋(BLACK=1)胜则黑棋各步价值为正
            if result == 0:
                v = 0.0
            elif result == BLACK:
                # 黑棋第0,2,4...步（从黑棋视角）价值为正
                v = 1.0 if i % 2 == 0 else -1.0
            else:
                # 白棋胜
                v = -1.0 if i % 2 == 0 else 1.0
            filled_data[i] = (state, policy, v, source)

        return filled_data


# ============================================================================
# 传统AI对抗数据生成器
# ============================================================================

class TraditionalOpponentWorker:
    """传统AI对抗数据生成器"""

    def __init__(self, model: GomokuNet, trad_depth: int = 4,
                 mcts_simulations: int = 200, device: torch.device = None):
        self.model = model
        self.trad_depth = trad_depth
        self.mcts_simulations = mcts_simulations
        self.device = device or torch.device('cpu')
        self.trad_ai = TraditionalAI(search_depth=trad_depth)

    def play_game(self, nn_as_black: bool = True) -> List[Tuple[np.ndarray, np.ndarray, float, str]]:
        """
        进行一局神经网络 vs 传统AI对弈
        Args:
            nn_as_black: 神经网络是否执黑
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
                # 神经网络走子
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
                # 传统AI走子
                move = self.trad_ai.get_best_move(game)
                # 为传统AI走子也记录数据
                if move:
                    state = game.get_state_planes()
                    # 策略目标：传统AI选择的位置为one-hot
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
# 难度自动调控器
# ============================================================================

class AutoDifficultyAdjuster:
    """传统AI难度自动调控器"""

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

        # 统计窗口
        self.recent_results: deque = deque(maxlen=win_rate_window)
        self.games_since_adjust = 0
        self.adjust_history: List[Dict] = []

    def record_result(self, nn_won: bool):
        """记录一场对弈结果"""
        self.recent_results.append(1 if nn_won else 0)
        self.games_since_adjust += 1

    def should_adjust(self) -> bool:
        return self.games_since_adjust >= self.games_per_adjust

    def get_win_rate(self) -> float:
        """获取神经网络近期胜率"""
        if len(self.recent_results) < 10:
            return 0.5
        return sum(self.recent_results) / len(self.recent_results)

    def adjust(self) -> Dict[str, Any]:
        """执行难度调整"""
        if not self.should_adjust():
            return {'adjusted': False}

        win_rate = self.get_win_rate()
        old_depth = self.current_depth
        adjusted = False

        upper = self.target_win_rate + 0.05
        lower = self.target_win_rate - 0.05

        if win_rate > upper and self.current_depth < self.depth_range[1]:
            self.current_depth = min(self.depth_range[1], self.current_depth + 1)
            adjusted = True
        elif win_rate < lower and self.current_depth > self.depth_range[0]:
            self.current_depth = max(self.depth_range[0], self.current_depth - 1)
            adjusted = True

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
            logger.info(f"Auto-difficulty: depth {old_depth} -> {self.current_depth} "
                        f"(win_rate={win_rate:.2%})")

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
        """更新ELO分数"""
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
# 智能参数自动调控器
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
    noise_scale: float = 1.0      # 数据增强噪声缩放


class AutoParameterController:
    """
    智能参数自动调控器

    根据训练进度自动调整：
    - MCTS 搜索温度：随训练进度逐步降低，控制探索/利用平衡
    - MCTS 模拟次数：随模型能力提升逐步增加
    - Dirichlet 噪声强度：随训练进度衰减
    - PUCT 探索常数：根据 loss 收敛情况微调
    - 每轮对弈局数：根据经验池健康度动态调整
    """

    def __init__(self,
                 initial_mcts: int = 200,
                 mcts_range: Tuple[int, int] = (100, 800),
                 initial_temperature: float = 1.5,
                 temperature_min: float = 0.1):
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
        self.recent_losses: deque = deque(maxlen=200)
        self.recent_win_rates: deque = deque(maxlen=20)
        self.loss_stable_count = 0
        self.last_mcts_adjust_step = 0
        self.mcts_adjust_cooldown = 2000

        # 阶段追踪
        self.phase = 'early'  # early / mid / late
        self._phase_thresholds = (0.15, 0.60)  # early→mid→late

        logger.info(f"AutoParameterController 初始化: "
                    f"mcts={initial_mcts}, temp={initial_temperature}, "
                    f"puct={self.c_puct}")

    def update(self, global_step: int, total_steps: int,
               loss_info: Optional[Dict] = None,
               win_rate: Optional[float] = None,
               buffer_size: int = 0) -> AutoParams:
        """
        每次训练迭代后调用，自动调整参数。

        Returns:
            AutoParams: 更新后的参数快照
        """
        progress = min(1.0, global_step / max(1, total_steps))

        # ── 阶段判定 ──
        if progress < self._phase_thresholds[0]:
            self.phase = 'early'
        elif progress < self._phase_thresholds[1]:
            self.phase = 'mid'
        else:
            self.phase = 'late'

        # ── 1. MCTS搜索温度：随进度衰减 ──
        # early: 1.5→0.8, mid: 0.8→0.2, late: 0.2→0.1
        if self.phase == 'early':
            self.temperature = 1.5 - progress / self._phase_thresholds[0] * 0.7
        elif self.phase == 'mid':
            p = (progress - self._phase_thresholds[0]) / (self._phase_thresholds[1] - self._phase_thresholds[0])
            self.temperature = 0.8 - p * 0.6
        else:
            self.temperature = max(self.temperature_min, 0.2 * (1.0 - progress) + self.temperature_min)

        # ── 2. Dirichlet 噪声：随进度衰减 ──
        self.dirichlet_epsilon = max(
            self.dirichlet_epsilon_min,
            0.25 * (1.0 - progress * 0.85)
        )

        # ── 3. MCTS模拟次数：根据胜率+loss趋势动态调整 ──
        if win_rate is not None:
            self.recent_win_rates.append(win_rate)

        if loss_info and 'total_loss' in loss_info:
            self.recent_losses.append({'total_loss': loss_info['total_loss'], 'step': global_step})

        # 每 cooldown 步检查一次 MCTS 调整
        if global_step - self.last_mcts_adjust_step >= self.mcts_adjust_cooldown:
            self._adjust_mcts(global_step)
            self.last_mcts_adjust_step = global_step

        # ── 4. PUCT探索常数：根据loss收敛状态微调 ──
        if len(self.recent_losses) >= 100:
            recent_100 = list(self.recent_losses)[-100:]
            losses = [r['total_loss'] if isinstance(r, dict) else r for r in recent_100]
            avg_loss = sum(losses) / len(losses)
            loss_std = (sum((l - avg_loss) ** 2 for l in losses) / len(losses)) ** 0.5

            if loss_std < 0.05 and avg_loss < 1.0:
                # Loss已稳定→减少探索，专注优化
                self.c_puct = max(1.2, self.c_puct - 0.05)
            elif loss_std > 0.3:
                # Loss波动大→增加探索
                self.c_puct = min(4.0, self.c_puct + 0.1)

        # ── 5. 每轮对弈局数：根据经验池健康度 ──
        if buffer_size < 10000:
            self.games_per_iteration = 10  # 经验池小时多收集
        elif buffer_size > 100000:
            self.games_per_iteration = 3   # 经验池充足时少收集
        else:
            self.games_per_iteration = 5

        # ── 6. 温度截止步数：中期开始逐步增大探索步数 ──
        if self.phase == 'early':
            self.temperature_cutoff = 12
        elif self.phase == 'mid':
            self.temperature_cutoff = 18
        else:
            self.temperature_cutoff = 10  # 后期缩小，更确定性

        return self.get_params()

    def _adjust_mcts(self, step: int):
        """根据近期胜率趋势调整MCTS模拟次数"""
        if len(self.recent_win_rates) < 10:
            return

        avg_wr = sum(self.recent_win_rates) / len(self.recent_win_rates)

        if self.phase == 'early':
            # 早期：快速试探，胜率高则加速增加
            if avg_wr > 0.45:
                self.mcts_simulations = min(self.mcts_range[1], self.mcts_simulations + 50)
        elif self.phase == 'mid':
            # 中期：稳健调整
            if avg_wr > 0.55:
                self.mcts_simulations = min(self.mcts_range[1], self.mcts_simulations + 30)
            elif avg_wr < 0.30:
                self.mcts_simulations = max(self.mcts_range[0], self.mcts_simulations - 20)
        else:
            # 后期：大搜索量
            self.mcts_simulations = min(self.mcts_range[1], max(self.mcts_simulations, int(self.mcts_range[1] * 0.85)))

    def get_params(self) -> AutoParams:
        """获取当前参数快照"""
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
        """获取控制器状态摘要"""
        return {
            'phase': self.phase,
            'mcts_simulations': self.mcts_simulations,
            'temperature': round(self.temperature, 3),
            'dirichlet_epsilon': round(self.dirichlet_epsilon, 3),
            'c_puct': round(self.c_puct, 2),
            'games_per_iteration': self.games_per_iteration,
            'temperature_cutoff': self.temperature_cutoff,
        }


# ============================================================================
# 训练器
# ============================================================================

class Trainer:
    """主训练器"""

    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: 训练配置字典
        """
        self.config = config
        self.device = torch.device('cuda:0' if torch.cuda.is_available() and
                                    config.get('training', {}).get('use_cuda', True)
                                    else 'cpu')

        logger.info(f"Training on device: {self.device}")

        # 模型
        self.model = GomokuNet().to(self.device)
        self.best_model = GomokuNet().to(self.device)

        # 优化器
        train_cfg = config.get('training', {})
        lr = train_cfg.get('learning_rate', 0.02)
        self.optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=0.9,
                                    weight_decay=1e-4)
        self.scaler = GradScaler('cuda')  # 混合精度

        # 学习率调度器
        total_steps = train_cfg.get('total_steps', 500000)
        warmup_steps = int(total_steps * 0.05)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=total_steps - warmup_steps
        )
        self.warmup_steps = warmup_steps
        self.base_lr = lr

        # 经验回放池
        replay_cfg = config.get('training', {}).get('replay_buffer', {})
        capacity = replay_cfg.get('capacity', 2000000)
        sampling_weights = replay_cfg.get('sampling_weights',
                                           {'self': 0.6, 'trad': 0.3, 'human': 0.1})
        self.replay_buffer = ReplayBuffer(capacity=capacity, sampling_weights=sampling_weights)

        # 配置参数
        self.batch_size = train_cfg.get('batch_size', 1024)
        self.num_mcts_simulations = train_cfg.get('num_mcts_simulations', 400)
        self.save_interval_minutes = train_cfg.get('save_interval_minutes', 5)
        self.auto_eval_games = train_cfg.get('auto_eval_games', 100)

        # 训练模式
        self.modes = train_cfg.get('modes', ['self'])

        # 传统AI相关
        trad_cfg = train_cfg.get('traditional', {})
        self.trad_depth = trad_cfg.get('initial_depth', 4)
        self.adjuster = AutoDifficultyAdjuster(
            initial_depth=self.trad_depth,
            depth_range=tuple(trad_cfg.get('depth_range', [1, 8])),
            games_per_adjust=trad_cfg.get('games_per_adjust', 10),
            target_win_rate=trad_cfg.get('target_win_rate', 0.20),
            win_rate_window=trad_cfg.get('win_rate_window', 100),
        )

        # ELO系统
        self.elo_system = ELOSystem()

        # 智能参数自动调控器
        self.auto_params = AutoParameterController(
            initial_mcts=min(200, self.num_mcts_simulations),
            mcts_range=(100, 800),
            initial_temperature=1.5,
            temperature_min=0.1,
        )

        # 训练状态
        self.global_step = 0
        self.total_games_self = 0
        self.total_games_trad = 0
        self.total_games_human = 0
        self.running = False
        self.paused = False

        # 训练历史记录
        self.loss_history: List[Dict] = []
        self.win_rate_history: List[Dict] = []

        # 模型目录（优先绝对路径，否则基于项目根目录）
        model_dir = train_cfg.get('model_dir', './models')
        if not os.path.isabs(model_dir):
            model_dir = os.path.join(PROJECT_DIR, model_dir.lstrip('./\\'))
        self.model_dir = model_dir
        os.makedirs(self.model_dir, exist_ok=True)

        # 加载已有模型
        self._load_latest_model()

    def _load_latest_model(self):
        """加载最新模型（覆盖策略：只有一个model.pt）"""
        model_path = os.path.join(self.model_dir, 'model.pt')
        if os.path.exists(model_path):
            try:
                model, info = GomokuNet.load_checkpoint(model_path, self.device)
                self.model = model
                self.global_step = info.get('step', 0)
                logger.info(f"Loaded model from step {self.global_step}, elo={info.get('elo', '-')}")
            except Exception as e:
                logger.warning(f"Failed to load model: {e}")

        # 加载训练状态（优化器、全局步数等）
        state_path = os.path.join(self.model_dir, 'training_state.pt')
        if os.path.exists(state_path):
            try:
                state = torch.load(state_path, map_location=self.device)
                self.global_step = state.get('step', self.global_step)
                if state.get('optimizer_state_dict'):
                    self.optimizer.load_state_dict(state['optimizer_state_dict'])
                # 恢复auto_params状态
                if state.get('auto_params'):
                    ap = state['auto_params']
                    self.auto_params.mcts_simulations = ap.get('mcts_simulations', 200)
                    self.auto_params.temperature = ap.get('temperature', 1.5)
                    self.auto_params.dirichlet_epsilon = ap.get('dirichlet_epsilon', 0.25)
                    self.auto_params.c_puct = ap.get('c_puct', 2.5)
                    self.auto_params.games_per_iteration = ap.get('games_per_iteration', 5)
                logger.info(f"Loaded training state from step {self.global_step}")
            except Exception as e:
                logger.warning(f"Failed to load training state: {e}")

    def _save_checkpoint(self):
        """保存模型和训练状态（覆盖策略：始终覆盖旧文件）"""
        # 保存模型（覆盖旧模型）
        model_path = os.path.join(self.model_dir, 'model.pt')
        self.model.save_checkpoint(
            model_path,
            step=self.global_step,
            extra={'elo': self.elo_system.get_elo('current')}
        )

        # 保存训练状态（覆盖，用于断点续训）
        state_path = os.path.join(self.model_dir, 'training_state.pt')
        torch.save({
            'step': self.global_step,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'auto_params': {
                'mcts_simulations': self.auto_params.mcts_simulations,
                'temperature': self.auto_params.temperature,
                'dirichlet_epsilon': self.auto_params.dirichlet_epsilon,
                'c_puct': self.auto_params.c_puct,
                'games_per_iteration': self.auto_params.games_per_iteration,
            },
            'elo_current': self.elo_system.get_elo('current'),
            'elo_best': self.elo_system.get_elo('best'),
        }, state_path)

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
        """生成传统AI对抗数据"""
        worker = TraditionalOpponentWorker(
            model=self.model,
            trad_depth=self.adjuster.current_depth,
            mcts_simulations=min(200, self.num_mcts_simulations),
            device=self.device
        )
        for i in range(num_games):
            if not self.running or self.paused:
                break
            nn_as_black = (i % 2 == 0)
            data = worker.play_game(nn_as_black=nn_as_black)
            self.replay_buffer.add_batch(data)
            self.total_games_trad += 1

            # 记录胜负结果用于难度调控
            if data:
                # 判断神经网络是否获胜（数据中最后一个状态的价值）
                last_value = data[-1][2]
                nn_won = (last_value > 0)
                self.adjuster.record_result(nn_won)

            # 检查是否需要调整难度
            if self.adjuster.should_adjust():
                self.adjuster.adjust()

    def _train_step(self) -> Dict[str, float]:
        """单步训练"""
        if len(self.replay_buffer) < self.batch_size:
            return {}

        self.model.train()

        # 采样批次
        states, policy_targets, value_targets = self.replay_buffer.sample(self.batch_size)

        states = torch.from_numpy(states).to(self.device)
        policy_targets = torch.from_numpy(policy_targets).to(self.device)
        value_targets = torch.from_numpy(value_targets).to(self.device)

        self.optimizer.zero_grad()

        # 混合精度训练
        with autocast('cuda'):
            policy_logits, values = self.model(states)
            policy_loss = F.cross_entropy(policy_logits, policy_targets)
            value_loss = F.mse_loss(values, value_targets)
            total_loss = policy_loss + value_loss

        self.scaler.scale(total_loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

        # 学习率预热
        if self.global_step < self.warmup_steps:
            lr_scale = (self.global_step + 1) / max(1, self.warmup_steps)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.base_lr * lr_scale
        else:
            self.scheduler.step()

        self.global_step += 1

        return {
            'total_loss': total_loss.item(),
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'learning_rate': self.optimizer.param_groups[0]['lr'],
        }

    def _evaluate_models(self, num_games: int = 50) -> Dict[str, float]:
        """评估当前模型 vs 最佳模型"""
        wins = 0
        losses = 0
        draws = 0

        worker_self = SelfPlayWorker(model=self.model, num_simulations=200, device=self.device)
        # 这里简化：直接用模型对弈判断
        # 实际应让两个模型分别作为不同方对弈

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
                    break  # 防止无限循环

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
        自动评估神经网络 vs 传统AI（GPU加速版）
        运行 num_games 局对弈，统计胜率，自动调整传统AI搜索深度

        Args:
            num_games: 评估局数（默认100局）
        Returns:
            评估结果字典
        """
        logger.info(f"开始自动评估: NN vs 传统AI (深度={self.adjuster.current_depth}), "
                    f"共{num_games}局 [GPU加速]")

        nn_wins = 0
        trad_wins = 0
        draws = 0
        total_time = 0.0

        for i in range(num_games):
            nn_as_black = (i % 2 == 0)

            worker = TraditionalOpponentWorker(
                model=self.model,
                trad_depth=self.adjuster.current_depth,
                mcts_simulations=min(200, self.num_mcts_simulations),
                device=self.device
            )

            t0 = time.time()
            game_data = worker.play_game(nn_as_black=nn_as_black)
            elapsed = time.time() - t0
            total_time += elapsed

            # 判断胜负
            if game_data:
                last_value = game_data[-1][2]
                if last_value > 0:
                    nn_wins += 1
                elif last_value < 0:
                    trad_wins += 1
                else:
                    draws += 1

                # 记录到难度调节器
                self.adjuster.record_result(last_value > 0)
            else:
                draws += 1

            # 每10局打印进度
            if (i + 1) % 10 == 0:
                wr = nn_wins / (i + 1) if (i + 1) > 0 else 0
                logger.info(f"  评估进度: {i + 1}/{num_games} | "
                            f"NN胜: {nn_wins} | 传统AI胜: {trad_wins} | "
                            f"平: {draws} | 当前胜率: {wr:.2%} | "
                            f"平均耗时: {total_time / (i + 1):.2f}s/局")

        nn_win_rate = nn_wins / num_games if num_games > 0 else 0
        avg_time = total_time / num_games if num_games > 0 else 0

        # 自动调整传统AI难度
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
                    f"平均耗时: {avg_time:.2f}s/局 [GPU]")

        return result

    def train_loop(self, total_steps: int = None, games_per_iteration: int = None):
        """
        主训练循环（智能自动调参）

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
        # 自动评估间隔（秒）
        eval_interval_seconds = 600

        logger.info(f"Starting training loop. Target: {total_steps} steps. "
                    f"Modes: {self.modes}")
        logger.info(f"Device: {self.device}, Batch size: {self.batch_size}")
        logger.info(f"Auto-save: every {self.save_interval_minutes}min | "
                    f"Auto-eval: {self.auto_eval_games} games vs Traditional AI")
        logger.info(f"Smart Auto-Params: ENABLED (temperature/MCTS/noise/PUCT auto-adjusted)")

        try:
            while self.running and self.global_step < total_steps:
                if self.paused:
                    time.sleep(1)
                    continue

                # 获取智能调控参数
                ap = self.auto_params.get_params()
                auto_games = games_per_iteration if games_per_iteration is not None else ap.games_per_iteration

                # 生成训练数据
                if 'self' in self.modes:
                    self._generate_self_play_data(auto_games)
                if 'trad' in self.modes:
                    self._generate_traditional_data(auto_games)

                # 训练步骤
                loss_info = {}
                if len(self.replay_buffer) >= self.batch_size:
                    for _ in range(10):  # 每轮训练10步
                        if not self.running:
                            break
                        loss_info = self._train_step()
                        if loss_info:
                            self.loss_history.append({**loss_info, 'step': self.global_step})

                    # ── 智能参数自动更新 ──
                    self.auto_params.update(
                        global_step=self.global_step,
                        total_steps=total_steps,
                        loss_info=loss_info,
                        win_rate=self.adjuster.get_win_rate(),
                        buffer_size=len(self.replay_buffer),
                    )

                    # 打印进度（含自动调控参数）
                    if self.global_step % 100 == 0:
                        buf_stats = self.replay_buffer.get_stats()
                        ap_summary = self.auto_params.get_summary()
                        logger.info(
                            f"Step {self.global_step}/{total_steps} | "
                            f"Loss: {loss_info.get('total_loss', 0):.4f} | "
                            f"Buffer: {buf_stats['total']} | "
                            f"LR: {loss_info.get('learning_rate', 0):.6f}"
                        )
                        logger.info(
                            f"  Auto: phase={ap_summary['phase']} | "
                            f"MCTS={ap_summary['mcts_simulations']} | "
                            f"temp={ap_summary['temperature']:.3f} | "
                            f"noise_eps={ap_summary['dirichlet_epsilon']:.3f} | "
                            f"PUCT={ap_summary['c_puct']:.2f} | "
                            f"games/iter={ap_summary['games_per_iteration']} | "
                            f"trad_depth={self.adjuster.current_depth}"
                        )

                # 定期评估和保存（每5000步）
                if self.global_step > 0 and self.global_step % 5000 == 0:
                    logger.info("Evaluating models...")
                    eval_result = self._evaluate_models(num_games=50)
                    logger.info(f"Eval: {eval_result}")

                    self.win_rate_history.append({
                        'step': self.global_step,
                        **eval_result,
                    })

                    # 覆盖保存模型
                    self._save_checkpoint()
                    if eval_result['win_rate'] > 0.55:
                        self.elo_system.update('current', 'best', 50)

                # 定时自动保存（覆盖旧模型）
                elapsed = time.time() - last_save_time
                if elapsed > self.save_interval_minutes * 60:
                    logger.info(f"Auto-save: 覆盖保存模型 (间隔: {self.save_interval_minutes}分钟)")
                    self._save_checkpoint()
                    last_save_time = time.time()

                # 定时自动评估 NN vs 传统AI
                eval_elapsed = time.time() - last_eval_time
                if eval_elapsed > eval_interval_seconds and len(self.replay_buffer) >= self.batch_size * 2:
                    logger.info(f"Auto-eval: 开始{self.auto_eval_games}局 NN vs 传统AI 评估 (GPU加速)...")
                    eval_vs_trad = self._auto_evaluate_vs_traditional(
                        num_games=self.auto_eval_games
                    )
                    self.win_rate_history.append({
                        'step': self.global_step,
                        'type': 'auto_eval_vs_trad',
                        **eval_vs_trad,
                    })
                    last_eval_time = time.time()

                # 定期输出参数调控日志（每5分钟）
                params_elapsed = time.time() - last_params_log_time
                if params_elapsed > 300:
                    ap_summary = self.auto_params.get_summary()
                    logger.info(f"[AutoParams] {ap_summary}")
                    last_params_log_time = time.time()

        except KeyboardInterrupt:
            logger.info("Training interrupted by user")
        finally:
            self._save_checkpoint()
            logger.info(f"Training stopped at step {self.global_step}")

    def stop(self):
        """停止训练"""
        self.running = False

    def pause(self):
        """暂停训练"""
        self.paused = True
        logger.info("Training paused")

    def resume(self):
        """恢复训练"""
        self.paused = False
        logger.info("Training resumed")

    def get_status(self) -> Dict[str, Any]:
        """获取训练状态"""
        return {
            'global_step': self.global_step,
            'running': self.running,
            'paused': self.paused,
            'buffer_size': len(self.replay_buffer),
            'buffer_stats': self.replay_buffer.get_stats(),
            'total_games_self': self.total_games_self,
            'total_games_trad': self.total_games_trad,
            'total_games_human': self.total_games_human,
            'trad_depth': self.adjuster.current_depth,
            'trad_win_rate': self.adjuster.get_win_rate(),
            'elo_current': self.elo_system.get_elo('current'),
            'elo_best': self.elo_system.get_elo('best'),
            'auto_params': self.auto_params.get_summary(),
            'loss_history': self.loss_history[-100:] if self.loss_history else [],
            'win_rate_history': self.win_rate_history[-20:] if self.win_rate_history else [],
        }
