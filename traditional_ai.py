"""
传统AI引擎 - PyTorch CUDA 加速版
Alpha-Beta剪枝 + Zobrist哈希置换表 + 启发式排序 + 迭代加深

所有计算密集部分（五连检测、棋型评分、候选排序、局面评估）使用 GPU 张量批处理。
Alpha-Beta 递归保持 CPU 控制流，每层节点就地修改 GPU 张量。
"""

import time
import random
from collections import OrderedDict
from typing import Tuple, List, Optional

import numpy as np
import torch
import torch.nn.functional as F

from game import GomokuGame, BOARD_SIZE, EMPTY, BLACK, WHITE

# ==================== 方向定义 ====================
DIRS = [(1, 0), (0, 1), (1, 1), (1, -1)]

# ==================== Zobrist 哈希表（CPU，uint64 不适合 GPU） ====================
_zobrist_table = np.random.randint(0, 2 ** 63, size=(2, BOARD_SIZE, BOARD_SIZE), dtype=np.uint64)


def _zobrist_hash(board_np: np.ndarray) -> np.uint64:
    """计算棋盘 Zobrist 哈希值 (board_np: 0=空,1=黑,2=白)"""
    h = np.uint64(0)
    b1 = (board_np == 1)
    b2 = (board_np == 2)
    h ^= _zobrist_table[0][b1].sum(dtype=np.uint64) if b1.any() else np.uint64(0)
    h ^= _zobrist_table[1][b2].sum(dtype=np.uint64) if b2.any() else np.uint64(0)
    return h


# ==================== 棋盘格式转换 ====================
def _to_internal(board_game: np.ndarray) -> np.ndarray:
    """game.py 格式 (0/1/-1) → AI 内部格式 (0/1/2)"""
    b = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int8)
    b[board_game == BLACK] = 1
    b[board_game == WHITE] = 2
    return b


def _player_to_internal(player: int) -> int:
    return 1 if player == BLACK else 2


def _player_from_internal(p: int) -> int:
    return BLACK if p == 1 else WHITE


# ==================== 置换表（LRU 缓存） ====================
TT_SIZE = 1 << 18
_tt = OrderedDict()

FLAG_EXACT = 0
FLAG_UPPER = 1
FLAG_LOWER = 2


def _tt_store(hash_key, depth, value, flag, best_move):
    if len(_tt) >= TT_SIZE:
        _tt.popitem(last=False)
    _tt[hash_key] = (depth, value, flag, best_move)


def _tt_lookup(hash_key, depth, alpha, beta):
    if hash_key not in _tt:
        return None, None, False
    stored_depth, stored_value, flag, best_move = _tt[hash_key]
    _tt.move_to_end(hash_key)
    if stored_depth >= depth:
        if flag == FLAG_EXACT:
            return stored_value, best_move, True
        elif flag == FLAG_UPPER and stored_value <= alpha:
            return stored_value, best_move, True
        elif flag == FLAG_LOWER and stored_value >= beta:
            return stored_value, best_move, True
    return None, best_move, False


# ============================================================================
# GPU 加速核心
# ============================================================================

class GPUEvaluator:
    """GPU 加速棋盘评估器 - 所有张量操作在 GPU 上"""

    def __init__(self, device: torch.device = None):
        self.device = device or torch.device(
            'cuda:0' if torch.cuda.is_available() else 'cpu')
        self._gpu = (self.device.type == 'cuda')

        # ---- conv2d 五连检测核 (4, 1, 5, 5) ----
        ones_h_5 = torch.zeros(1, 1, 5, 5, device=self.device)
        ones_h_5[:, :, 2:3, :] = 1.0  # 中间行的 5 列
        ones_v_5 = torch.zeros(1, 1, 5, 5, device=self.device)
        ones_v_5[:, :, :, 2:3] = 1.0  # 中间列的 5 行
        eye5 = torch.eye(5, device=self.device)
        ones_d = eye5.view(1, 1, 5, 5)              # 主对角线
        ones_a = eye5.flip(1).view(1, 1, 5, 5)      # 反对角线
        self.win_kernels = torch.cat([ones_h_5, ones_v_5, ones_d, ones_a], dim=0)

        # ---- 中心距离预计算 ----
        center = BOARD_SIZE // 2
        r_idx = torch.arange(BOARD_SIZE, device=self.device).float()
        c_idx = torch.arange(BOARD_SIZE, device=self.device).float()
        dist_r = (r_idx.view(-1, 1) - center).abs()
        dist_c = (c_idx.view(1, -1) - center).abs()
        self.center_dist = dist_r + dist_c  # (15, 15)

    # ------------------------------------------------------------------
    # 1. 五连检测（conv2d 批量 4 方向）
    # ------------------------------------------------------------------
    def check_win(self, board: torch.Tensor, player: int) -> bool:
        """检查 player 是否五连"""
        mask = (board == player).float().view(1, 1, BOARD_SIZE, BOARD_SIZE)
        convs = F.conv2d(mask, self.win_kernels)  # (1, 4, h', w')
        return (convs >= 4.5).any().item()

    def find_winning_moves(self, board: torch.Tensor, player: int,
                           candidates: torch.Tensor) -> torch.Tensor:
        """
        批量检测每个候选是否让 player 五连
        Returns: (N,) bool tensor
        """
        if candidates.numel() == 0:
            return torch.zeros(0, dtype=torch.bool, device=self.device)

        N = candidates.shape[0]
        rows = candidates[:, 0]
        cols = candidates[:, 1]
        result = torch.zeros(N, dtype=torch.bool, device=self.device)

        for dr, dc in DIRS:
            # 提取 (N, 9) 线段
            line = torch.zeros(N, 9, dtype=torch.int8, device=self.device)
            for k in range(9):
                rk = rows + dr * (k - 4)
                ck = cols + dc * (k - 4)
                valid = (rk >= 0) & (rk < BOARD_SIZE) & (ck >= 0) & (ck < BOARD_SIZE)
                rc = rk.clamp(0, BOARD_SIZE - 1)
                cc = ck.clamp(0, BOARD_SIZE - 1)
                v = board[rc, cc]
                # 越界位置用对方颜色填充
                v = torch.where(valid, v, torch.tensor(3 - player, dtype=torch.int8, device=self.device))
                line[:, k] = v

            # 滑动窗口检测连续 5 个 player
            for start_k in range(5):
                w = line[:, start_k:start_k + 5]
                result |= (w == player).all(dim=1)

        return result

    # ------------------------------------------------------------------
    # 2. 批量候选评分（核心加速点）
    # ------------------------------------------------------------------
    def batch_score_moves(self, board: torch.Tensor, candidates: torch.Tensor,
                          player: int) -> torch.Tensor:
        """
        一次 GPU 调用评估所有候选位置
        Returns: (N,) float tensor of scores
        """
        if candidates.numel() == 0:
            return torch.zeros(0, device=self.device)

        N = candidates.shape[0]
        opp = 3 - player
        rows = candidates[:, 0].long()
        cols = candidates[:, 1].long()

        attack_total = torch.zeros(N, device=self.device, dtype=torch.float32)
        defense_total = torch.zeros(N, device=self.device, dtype=torch.float32)

        for dr, dc in DIRS:
            # 提取 (N, 9) 线段
            line = torch.zeros(N, 9, dtype=torch.int8, device=self.device)
            for k in range(9):
                rk = rows + dr * (k - 4)
                ck = cols + dc * (k - 4)
                valid = (rk >= 0) & (rk < BOARD_SIZE) & (ck >= 0) & (ck < BOARD_SIZE)
                rc = rk.clamp(0, BOARD_SIZE - 1)
                cc = ck.clamp(0, BOARD_SIZE - 1)
                v = board[rc, cc]
                v = torch.where(valid, v, torch.tensor(-1, dtype=torch.int8, device=self.device))
                line[:, k] = v

            # 评分：己方和对方分别计算
            atk = self._score_line_batch(line, player)      # (N,)
            dfs = self._score_line_batch(line, opp)          # (N,)
            attack_total += atk
            defense_total += dfs

        # 中心加成
        center_bonus = (BOARD_SIZE - 1 - self.center_dist[rows, cols]).clamp(min=0) * 5.0

        return attack_total * 2.0 + defense_total + center_bonus

    def _score_line_batch(self, line: torch.Tensor, player: int) -> torch.Tensor:
        """
        对 (N, 9) 线段批量评分 (单方向)
        line[:, 4] 是落子位置
        """
        N = line.shape[0]
        own = (line == player).int()         # (N, 9)
        empty = (line == 0).int()
        opp_block = ((line != player) & (line != 0)).int()

        # 从中心(idx=4)向正方向(idx=5→8)数连续己方棋子
        p1 = own[:, 5]
        p2 = own[:, 6] * p1
        p3 = own[:, 7] * p2
        p4 = own[:, 8] * p3

        # 从中心向反方向(idx=3→0)数连续己方棋子
        n1 = own[:, 3]
        n2 = own[:, 2] * n1
        n3 = own[:, 1] * n2
        n4 = own[:, 0] * n3

        pos_cnt = p1 + p2 + p3 + p4       # 正方向连续数
        neg_cnt = n1 + n2 + n3 + n4       # 反方向连续数
        total_cnt = pos_cnt + neg_cnt + 1  # 总连续数

        # 开放端检测：
        # 正方向：连续pos_cnt个后的下一个位置是否为空
        # pos_cnt ∈ [0,4]，对应检查 idx ∈ [5,9]，idx=9时越界→无开放端
        pos_empty = torch.zeros(N, device=self.device)
        for check_cnt in range(1, 5):
            mask = (pos_cnt == check_cnt)
            if mask.any():
                idx = 5 + check_cnt  # 5→9, idx=9 means off the line
                if idx < 9:
                    pos_empty += mask.float() * empty[:, idx].float()

        neg_empty = torch.zeros(N, device=self.device)
        for check_cnt in range(1, 5):
            mask = (neg_cnt == check_cnt)
            if mask.any():
                idx = 3 - check_cnt  # 2→-1, idx=-1 means off the line
                if idx >= 0:
                    neg_empty += mask.float() * empty[:, idx].float()

        # cnt=0: 正方向紧邻=idx5, 反方向紧邻=idx3
        pos_empty += (pos_cnt == 0).float() * empty[:, 5].float()
        neg_empty += (neg_cnt == 0).float() * empty[:, 3].float()

        open_ends = pos_empty + neg_empty  # (N,) 0-2
        is_live = (open_ends >= 2).float()

        # 跳活检测
        has_jump = torch.zeros(N, device=self.device)
        # 正方向跳活：idx=5空 + idx=6为己方
        has_jump += (empty[:, 5] * own[:, 6]).float()
        # 反方向跳活：idx=3空 + idx=2为己方
        has_jump += (empty[:, 3] * own[:, 2]).float()
        has_jump = (has_jump > 0).float()

        # 按棋型查分
        scores = torch.zeros(N, device=self.device)

        # 五连以上
        scores += (total_cnt >= 5).float() * 100000000

        # 活四
        scores += (total_cnt == 4).float() * is_live * (1 - has_jump) * 1000000
        scores += (total_cnt == 4).float() * is_live * has_jump * 10000000        # 活四+跳

        # 冲四
        scores += (total_cnt == 4).float() * (1 - is_live) * (1 - has_jump) * 10000
        scores += (total_cnt == 4).float() * (1 - is_live) * has_jump * 100000    # 冲四+跳

        # 活三
        scores += (total_cnt == 3).float() * is_live * (1 - has_jump) * 10000
        scores += (total_cnt == 3).float() * is_live * has_jump * 1000000         # 双活三

        # 眠三
        scores += (total_cnt == 3).float() * (1 - is_live) * (1 - has_jump) * 500
        scores += (total_cnt == 3).float() * (1 - is_live) * has_jump * 1000

        # 活二
        scores += (total_cnt == 2).float() * is_live * (1 - has_jump) * 200
        scores += (total_cnt == 2).float() * is_live * has_jump * 1000

        # 眠二
        scores += (total_cnt == 2).float() * (1 - is_live) * 50

        # 活一
        scores += (total_cnt == 1).float() * is_live * 10

        # 眠一
        scores += (total_cnt == 1).float() * (1 - is_live) * 1

        return scores

    # ------------------------------------------------------------------
    # 3. 局面评估
    # ------------------------------------------------------------------
    def evaluate_board(self, board: torch.Tensor, ai_player: int) -> float:
        """
        完整局面评估（正值对 AI 有利）
        """
        opp = 3 - ai_player

        if self.check_win(board, ai_player):
            return 100000000.0
        if self.check_win(board, opp):
            return -100000000.0

        # 活四必胜
        if self._has_live_four_gpu(board, ai_player):
            return 50000000.0

        ai_score = self._evaluate_player_gpu(board, ai_player)
        opp_score = self._evaluate_player_gpu(board, opp)
        return ai_score - opp_score * 0.95

    def _evaluate_player_gpu(self, board: torch.Tensor, player: int) -> float:
        """GPU 全盘棋型评分 — 对所有己方棋子批量评分后求和"""
        own_mask = (board == player)

        # 找出所有己方棋子的位置
        positions = torch.nonzero(own_mask)  # (M, 2)
        if positions.numel() == 0:
            return 0.0

        M = positions.shape[0]
        rows = positions[:, 0].long()
        cols = positions[:, 1].long()
        N = M

        total_score = torch.zeros(N, device=self.device, dtype=torch.float32)

        for dr, dc in DIRS:
            # 提取每个位置的 9 格线段
            line = torch.zeros(N, 9, dtype=torch.int8, device=self.device)
            for k in range(9):
                rk = rows + dr * (k - 4)
                ck = cols + dc * (k - 4)
                valid = (rk >= 0) & (rk < BOARD_SIZE) & (ck >= 0) & (ck < BOARD_SIZE)
                rc = rk.clamp(0, BOARD_SIZE - 1)
                cc = ck.clamp(0, BOARD_SIZE - 1)
                v = board[rc, cc]
                v = torch.where(valid, v, torch.tensor(-1, dtype=torch.int8, device=self.device))
                line[:, k] = v

            # 评分该方向
            dir_score = self._score_line_batch(line, player)
            total_score += dir_score

        # 中心加成
        center_bonus = ((BOARD_SIZE - 1) - self.center_dist[rows, cols]).clamp(min=0) * 0.05
        total_score += center_bonus.float()

        return total_score.sum().item()

    # ------------------------------------------------------------------
    # 4. 活四检测
    # ------------------------------------------------------------------
    def _has_live_four_gpu(self, board: torch.Tensor, player: int) -> bool:
        """GPU 活四检测"""
        candidates = self._generate_moves_gpu(board)
        N = candidates.shape[0]
        for i in range(min(N, 100)):
            r, c = int(candidates[i, 0].item()), int(candidates[i, 1].item())
            board[r, c] = player
            found = self._check_live_four_at(board, r, c, player)
            board[r, c] = 0
            if found:
                return True
        return False

    def _check_live_four_at(self, board: torch.Tensor, r: int, c: int, player: int) -> bool:
        """检查 (r,c) 落子后沿各方向是否形成真活四"""
        for dr, dc in DIRS:
            pos_cnt = 0
            for k in range(1, 6):
                nr, nc = r + dr * k, c + dc * k
                if 0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE and board[nr, nc].item() == player:
                    pos_cnt += 1
                else:
                    break
            neg_cnt = 0
            for k in range(1, 6):
                nr, nc = r - dr * k, c - dc * k
                if 0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE and board[nr, nc].item() == player:
                    neg_cnt += 1
                else:
                    break

            total = pos_cnt + neg_cnt + 1
            if total == 4:
                nr_pos = r + dr * (pos_cnt + 1)
                nc_pos = c + dc * (pos_cnt + 1)
                pos_empty = (0 <= nr_pos < BOARD_SIZE and 0 <= nc_pos < BOARD_SIZE and
                             board[nr_pos, nc_pos].item() == 0)
                nr_neg = r - dr * (neg_cnt + 1)
                nc_neg = c - dc * (neg_cnt + 1)
                neg_empty = (0 <= nr_neg < BOARD_SIZE and 0 <= nc_neg < BOARD_SIZE and
                             board[nr_neg, nc_neg].item() == 0)
                if pos_empty and neg_empty:
                    return True
        return False

    # ------------------------------------------------------------------
    # 5. 候选位置生成（GPU max_pool2d 膨胀）
    # ------------------------------------------------------------------
    def _generate_moves_gpu(self, board: torch.Tensor) -> torch.Tensor:
        """
        GPU 候选位置生成：已有棋子周围 2 格的空位
        Returns: (N, 2) int tensor
        """
        occupied = (board != 0).float()
        dilated = F.max_pool2d(
            occupied.view(1, 1, BOARD_SIZE, BOARD_SIZE),
            kernel_size=5, stride=1, padding=2
        ).squeeze()
        candidates_mask = (dilated > 0) & (board == 0)

        if not candidates_mask.any():
            return torch.tensor([[BOARD_SIZE // 2, BOARD_SIZE // 2]], device=self.device)

        return torch.nonzero(candidates_mask).long()


# ============================================================================
# 全局单例 GPU 评估器
# ============================================================================
_gpu_eval: Optional[GPUEvaluator] = None


def _get_gpu_eval() -> GPUEvaluator:
    global _gpu_eval
    if _gpu_eval is None:
        _gpu_eval = GPUEvaluator()
    return _gpu_eval


# ============================================================================
# TraditionalAI 接口类
# ============================================================================

class TraditionalAI:
    """传统五子棋 AI（PyTorch GPU 加速版）"""

    def __init__(self, search_depth: int = 4):
        self.search_depth = max(1, min(8, search_depth))
        self.nodes_evaluated = 0
        self._last_time = 0.0
        self._eval = _get_gpu_eval()

    def set_depth(self, depth: int):
        self.search_depth = max(1, min(8, depth))

    def get_best_move(self, game: GomokuGame) -> Optional[Tuple[int, int]]:
        """获取最佳落子（GPU 加速版）"""
        if game.game_over:
            return None

        board_np = _to_internal(game.board)
        board = torch.from_numpy(board_np.astype(np.int8)).to(self._eval.device)
        ai_player = _player_to_internal(game.current_player)

        _tt.clear()

        t0 = time.perf_counter()
        move = self._ai_move_gpu(board, ai_player)
        self._last_time = time.perf_counter() - t0
        self.nodes_evaluated += 1

        return move

    def get_stats(self) -> dict:
        return {
            'depth': self.search_depth,
            'time': round(self._last_time, 3),
            'total_evals': self.nodes_evaluated,
            'device': str(self._eval.device),
            'gpu': self._eval._gpu,
        }

    # ================================================================
    # 核心搜索（GPU 版）
    # ================================================================

    def _ai_move_gpu(self, board: torch.Tensor, ai_player: int) -> Tuple[int, int]:
        """AI 主入口"""
        # 立即威胁检测
        threat = self._check_immediate_threat_gpu(board, ai_player)
        if threat is not None:
            return threat

        candidates = self._eval._generate_moves_gpu(board)
        if candidates.numel() == 0:
            return (BOARD_SIZE // 2, BOARD_SIZE // 2)

        # 第一步天元
        stone_count = int((board != 0).sum().item())
        if stone_count <= 1:
            center = BOARD_SIZE // 2
            if int(board[center, center].item()) == 0:
                return (center, center)
            idx = random.randint(0, int(candidates.shape[0]) - 1)
            return (int(candidates[idx, 0].item()), int(candidates[idx, 1].item()))

        # 批量评分 + 排序
        scores = self._eval.batch_score_moves(board, candidates, ai_player)
        sorted_idx = torch.argsort(scores, descending=True)

        max_branch = 15 if self.search_depth >= 2 else 10
        top_k = min(int(candidates.shape[0]), max_branch)
        top_candidates = candidates[sorted_idx[:top_k]]

        board_cpu = board.cpu().numpy()
        base_hash = _zobrist_hash(board_cpu)

        best_move = (int(top_candidates[0, 0].item()), int(top_candidates[0, 1].item()))
        best_val = float('-inf')

        for i in range(top_k):
            r = int(top_candidates[i, 0].item())
            c = int(top_candidates[i, 1].item())

            board_cpu[r, c] = ai_player
            new_hash = base_hash ^ _zobrist_table[ai_player - 1, r, c]

            board[r, c] = ai_player
            if self._eval.check_win(board, ai_player):
                board[r, c] = 0
                board_cpu[r, c] = 0
                return (r, c)

            val = self._alpha_beta_gpu(board, self.search_depth - 1,
                                       float('-inf'), float('inf'),
                                       False, ai_player, new_hash)
            board[r, c] = 0
            board_cpu[r, c] = 0

            if val > best_val:
                best_val = val
                best_move = (r, c)

        return best_move

    def _alpha_beta_gpu(self, board: torch.Tensor, depth: int,
                        alpha: float, beta: float,
                        maximizing: bool, ai_player: int,
                        hash_key) -> float:
        """Alpha-Beta 剪枝（评估全在 GPU，控制流 CPU）"""
        opp = 3 - ai_player

        # 置换表
        cached_val, _, hit = _tt_lookup(hash_key, depth, alpha, beta)
        if hit and depth > 0:
            return cached_val

        # 终局
        if self._eval.check_win(board, ai_player):
            return 10000000 + depth
        if self._eval.check_win(board, opp):
            return -10000000 - depth
        if depth == 0:
            return self._eval.evaluate_board(board, ai_player)

        candidates = self._eval._generate_moves_gpu(board)
        if candidates.numel() == 0:
            return 0

        current_player = ai_player if maximizing else opp

        # 批量评分排序
        scores = self._eval.batch_score_moves(board, candidates, current_player)
        sorted_idx = torch.argsort(scores, descending=True)

        max_branch = 25 if depth <= 1 else 20
        top_k = min(int(candidates.shape[0]), max_branch)

        if maximizing:
            best_val = float('-inf')
            for idx in range(top_k):
                i = int(sorted_idx[idx].item())
                r = int(candidates[i, 0].item())
                c = int(candidates[i, 1].item())

                board[r, c] = ai_player
                new_hash = hash_key ^ _zobrist_table[ai_player - 1, r, c]
                val = self._alpha_beta_gpu(board, depth - 1, alpha, beta,
                                           False, ai_player, new_hash)
                board[r, c] = 0

                best_val = max(best_val, val)
                alpha = max(alpha, val)
                if beta <= alpha:
                    break

            _tt_store(hash_key, depth, best_val,
                      FLAG_LOWER if best_val >= beta else FLAG_EXACT, None)
            return best_val
        else:
            best_val = float('inf')
            for idx in range(top_k):
                i = int(sorted_idx[idx].item())
                r = int(candidates[i, 0].item())
                c = int(candidates[i, 1].item())

                board[r, c] = opp
                new_hash = hash_key ^ _zobrist_table[opp - 1, r, c]
                val = self._alpha_beta_gpu(board, depth - 1, alpha, beta,
                                           True, ai_player, new_hash)
                board[r, c] = 0

                best_val = min(best_val, val)
                beta = min(beta, val)
                if beta <= alpha:
                    break

            _tt_store(hash_key, depth, best_val,
                      FLAG_UPPER if best_val <= alpha else FLAG_EXACT, None)
            return best_val

    def _check_immediate_threat_gpu(self, board: torch.Tensor, player: int
                                    ) -> Optional[Tuple[int, int]]:
        """立即威胁检测（GPU 版）"""
        opp = 3 - player
        candidates = self._eval._generate_moves_gpu(board)

        # 1. 自己能五连
        winning = self._eval.find_winning_moves(board, player, candidates)
        if winning.any():
            idx = int(torch.nonzero(winning)[0].item())
            return (int(candidates[idx, 0].item()), int(candidates[idx, 1].item()))

        # 2. 对手能五连 → 必须堵
        opp_win = self._eval.find_winning_moves(board, opp, candidates)
        if opp_win.any():
            idx = int(torch.nonzero(opp_win)[0].item())
            return (int(candidates[idx, 0].item()), int(candidates[idx, 1].item()))

        # 3. 自己能活四
        N = int(candidates.shape[0])
        for i in range(min(N, 100)):
            r = int(candidates[i, 0].item())
            c = int(candidates[i, 1].item())
            board[r, c] = player
            found = self._eval._check_live_four_at(board, r, c, player)
            board[r, c] = 0
            if found:
                return (r, c)

        # 4. 对手能活四
        for i in range(min(N, 100)):
            r = int(candidates[i, 0].item())
            c = int(candidates[i, 1].item())
            board[r, c] = opp
            found = self._eval._check_live_four_at(board, r, c, opp)
            board[r, c] = 0
            if found:
                return (r, c)

        return None
