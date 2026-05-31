"""
蒙特卡洛树搜索 (MCTS)
基于神经网络引导，使用PUCT公式进行节点选择
支持虚拟损失（Virtual Loss）用于并行评估
"""

import math
import numpy as np
import torch
import torch.nn.functional as F
from typing import Tuple, List, Optional, Dict
from game import GomokuGame, BOARD_SIZE, EMPTY, get_center_weights


class MCTSNode:
    """MCTS树节点"""

    def __init__(self, parent: Optional['MCTSNode'] = None, prior: float = 0.0,
                 move: Optional[Tuple[int, int]] = None):
        self.parent = parent
        self.children: Dict[Tuple[int, int], 'MCTSNode'] = {}
        self.prior = prior          # 先验概率 P(s,a)
        self.visit_count = 0        # 访问次数 N
        self.total_value = 0.0      # 累计价值 W
        self.mean_value = 0.0       # 平均价值 Q = W/N
        self.move = move            # 到达此节点的落子

    def is_leaf(self) -> bool:
        return len(self.children) == 0

    def is_root(self) -> bool:
        return self.parent is None

    def get_value(self) -> float:
        """获取节点平均价值"""
        if self.visit_count == 0:
            return 0.0
        return self.total_value / self.visit_count


class MCTS:
    """
    蒙特卡洛树搜索
    """

    def __init__(self,
                 model: torch.nn.Module,
                 num_simulations: int = 400,
                 c_puct: float = 2.5,
                 virtual_loss: float = 3.0,
                 device: torch.device = None):
        """
        Args:
            model: 神经网络模型
            num_simulations: 每次搜索的模拟次数
            c_puct: PUCT探索常数
            virtual_loss: 虚拟损失（用于并行评估）
            device: 计算设备
        """
        self.model = model
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.virtual_loss = virtual_loss
        self.device = device or torch.device('cpu')
        self.root: Optional[MCTSNode] = None

    def search(self, game: GomokuGame, temperature: float = 1.0,
               add_dirichlet_noise: bool = False, dirichlet_alpha: float = 0.03,
               dirichlet_epsilon: float = 0.25) -> Tuple[np.ndarray, float]:
        """
        执行MCTS搜索

        Args:
            game: 当前游戏状态
            temperature: 温度参数（控制探索程度）
            add_dirichlet_noise: 是否在根节点添加Dirichlet噪声
            dirichlet_alpha: Dirichlet分布的alpha参数
            dirichlet_epsilon: 噪声混合比例

        Returns:
            action_probs: (225,) 各动作的访问概率分布
            root_value: 根节点价值
        """
        if game.game_over:
            return np.zeros(BOARD_SIZE * BOARD_SIZE, dtype=np.float32), 0.0

        self.root = MCTSNode(prior=0.0)

        # 第一次评估根节点
        state_planes = game.get_state_planes()
        state_tensor = torch.from_numpy(state_planes).unsqueeze(0).to(self.device)
        self.model.eval()
        with torch.no_grad():
            policy_logits, value = self.model(state_tensor)
        policy_probs = F.softmax(policy_logits, dim=1).cpu().numpy().flatten()

        # 扩展到根节点的合法动作
        legal_moves = game.get_legal_moves()
        legal_indices = [r * BOARD_SIZE + c for r, c in legal_moves]

        # 计算中心位置权重（开局时更强）
        move_count = len(game.move_history)
        center_weights = get_center_weights(move_count, strength=0.8)

        for (r, c), idx in zip(legal_moves, legal_indices):
            prior = policy_probs[idx] * center_weights[idx]
            self.root.children[(r, c)] = MCTSNode(
                parent=self.root,
                prior=prior,
                move=(r, c)
            )

        # 添加Dirichlet噪声
        if add_dirichlet_noise:
            noise = np.random.dirichlet([dirichlet_alpha] * len(legal_moves))
            for (r, c), n in zip(legal_moves, noise):
                child = self.root.children[(r, c)]
                child.prior = (1 - dirichlet_epsilon) * child.prior + dirichlet_epsilon * n

        # 执行模拟
        for _ in range(self.num_simulations):
            node, search_path, sim_game = self._select(self.root, game.clone())
            value = self._evaluate_and_expand(node, sim_game)
            self._backpropagate(search_path, value)

        # 计算访问概率
        action_probs = np.zeros(BOARD_SIZE * BOARD_SIZE, dtype=np.float32)
        for (r, c), child in self.root.children.items():
            idx = r * BOARD_SIZE + c
            if temperature == 0:
                # 贪婪选择
                action_probs[idx] = 1.0 if child.visit_count == max(
                    c.visit_count for c in self.root.children.values()) else 0.0
            else:
                action_probs[idx] = child.visit_count ** (1.0 / temperature)

        # 归一化
        total = action_probs.sum()
        if total > 0:
            action_probs /= total

        root_value = self.root.get_value()
        return action_probs, root_value

    def _select(self, node: MCTSNode, game: GomokuGame) -> Tuple[MCTSNode, List[MCTSNode], GomokuGame]:
        """
        选择阶段：沿树向下遍历到叶节点
        返回 (叶节点, 搜索路径, 对应游戏状态)
        """
        search_path = [node]

        while not node.is_leaf() and not game.game_over:
            # 选择最佳子节点
            best_move, best_child = self._select_child(node)
            game.make_move(best_move[0], best_move[1])
            node = best_child
            search_path.append(node)

        return node, search_path, game

    def _select_child(self, node: MCTSNode) -> Tuple[Tuple[int, int], MCTSNode]:
        """使用PUCT公式选择最佳子节点（随机破平，保证非确定性探索）"""
        best_score = -float('inf')
        best_move = None
        best_child = None

        sqrt_parent_n = math.sqrt(node.visit_count + 1)

        # 打乱子节点顺序以实现随机破平——当多个子节点 visit_count=0、
        # mean_value=0 时 PUCT 全相等，必须随机选择否则整个搜索树是确定的
        items = list(node.children.items())
        np.random.shuffle(items)

        for move, child in items:
            # PUCT公式
            q_value = child.mean_value
            u_value = self.c_puct * child.prior * sqrt_parent_n / (1 + child.visit_count)
            score = q_value + u_value

            if score > best_score:
                best_score = score
                best_move = move
                best_child = child

        return best_move, best_child

    def _evaluate_and_expand(self, node: MCTSNode, game: GomokuGame) -> float:
        """
        评估叶节点并扩展
        返回评估价值（从当前玩家视角）
        """
        if game.game_over:
            if game.winner == 0:
                return 0.0  # 平局
            # 游戏结束时轮到的是当前玩家（因游戏结束不再切换）
            # 返回负值表示上一步落子方获胜
            return -1.0  # 当前局面下当前玩家已输

        # 网络评估
        state_planes = game.get_state_planes()
        state_tensor = torch.from_numpy(state_planes).unsqueeze(0).to(self.device)
        self.model.eval()
        with torch.no_grad():
            policy_logits, value = self.model(state_tensor)
        policy_probs = F.softmax(policy_logits, dim=1).cpu().numpy().flatten()
        value = value.item()

        # 扩展节点
        legal_moves = game.get_legal_moves()
        move_count = len(game.move_history)
        center_weights = get_center_weights(move_count, strength=0.8)
        for r, c in legal_moves:
            idx = r * BOARD_SIZE + c
            prior = policy_probs[idx] * center_weights[idx]
            node.children[(r, c)] = MCTSNode(
                parent=node,
                prior=prior,
                move=(r, c)
            )

        return value

    def _backpropagate(self, search_path: List[MCTSNode], value: float):
        """
        回溯：沿搜索路径更新节点统计信息
        value从当前玩家视角，交替翻转
        """
        for node in reversed(search_path):
            node.visit_count += 1
            node.total_value += value
            node.mean_value = node.total_value / node.visit_count
            value = -value  # 切换视角

    def get_action_visits(self) -> Dict[Tuple[int, int], int]:
        """获取根节点各动作的访问次数"""
        if self.root is None:
            return {}
        return {move: child.visit_count for move, child in self.root.children.items()}


def get_best_move_from_probs(action_probs: np.ndarray, game: GomokuGame,
                              deterministic: bool = False) -> Tuple[int, int]:
    """
    从概率分布中选择最佳落子

    Args:
        action_probs: (225,) 概率分布
        game: 当前游戏状态
        deterministic: 是否确定性选择（贪婪）

    Returns:
        (row, col)
    """
    legal_moves = game.get_legal_moves()
    if not legal_moves:
        return (-1, -1)

    if deterministic:
        # 选择概率最高的合法动作
        best_idx = -1
        best_prob = -1
        for r, c in legal_moves:
            idx = r * BOARD_SIZE + c
            if action_probs[idx] > best_prob:
                best_prob = action_probs[idx]
                best_idx = idx
        return (best_idx // BOARD_SIZE, best_idx % BOARD_SIZE)
    else:
        # 按概率采样
        probs = np.array([action_probs[r * BOARD_SIZE + c] for r, c in legal_moves])
        probs = probs / probs.sum()
        idx = np.random.choice(len(legal_moves), p=probs)
        return legal_moves[idx]
