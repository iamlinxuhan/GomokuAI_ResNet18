"""
五子棋游戏核心模块
15×15标准棋盘，无禁手规则，横/竖/斜四方向五连判定
"""

import numpy as np
from typing import Tuple, List, Optional

BOARD_SIZE = 15
BLACK = 1   # 黑棋先手
WHITE = -1  # 白棋后手
EMPTY = 0

# 方向向量：水平、垂直、对角线(\)、反对角线(/)
DIRECTIONS = [(1, 0), (0, 1), (1, 1), (1, -1)]


class GomokuGame:
    """五子棋游戏逻辑"""

    def __init__(self):
        self.board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int8)
        self.current_player = BLACK  # 黑棋先手
        self.move_history: List[Tuple[int, int]] = []
        self.game_over = False
        self.winner: Optional[int] = None
        self.win_line: List[Tuple[int, int]] = []  # 获胜连线坐标

    def reset(self):
        """重置棋盘"""
        self.board.fill(EMPTY)
        self.current_player = BLACK
        self.move_history.clear()
        self.game_over = False
        self.winner = None
        self.win_line.clear()

    def clone(self) -> 'GomokuGame':
        """深拷贝当前游戏状态"""
        g = GomokuGame()
        g.board = self.board.copy()
        g.current_player = self.current_player
        g.move_history = self.move_history.copy()
        g.game_over = self.game_over
        g.winner = self.winner
        g.win_line = self.win_line.copy()
        return g

    def get_legal_moves(self) -> List[Tuple[int, int]]:
        """获取所有合法落子位置"""
        if self.game_over:
            return []
        return [(r, c) for r in range(BOARD_SIZE) for c in range(BOARD_SIZE)
                if self.board[r, c] == EMPTY]

    def is_legal_move(self, row: int, col: int) -> bool:
        """判断落子是否合法"""
        if self.game_over:
            return False
        if not (0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE):
            return False
        return self.board[row, col] == EMPTY

    def make_move(self, row: int, col: int) -> bool:
        """落子，返回是否成功"""
        if not self.is_legal_move(row, col):
            return False
        self.board[row, col] = self.current_player
        self.move_history.append((row, col))
        self._check_winner(row, col)
        if not self.game_over:
            self.current_player = -self.current_player
        return True

    def undo_move(self) -> bool:
        """悔棋一步，返回是否成功"""
        if not self.move_history:
            return False
        row, col = self.move_history.pop()
        self.board[row, col] = EMPTY
        self.current_player = -self.current_player
        self.game_over = False
        self.winner = None
        self.win_line.clear()
        return True

    def _check_winner(self, row: int, col: int):
        """检查是否五连获胜"""
        player = self.board[row, col]
        if player == EMPTY:
            return

        for dr, dc in DIRECTIONS:
            line = [(row, col)]
            # 正方向延伸
            for i in range(1, 5):
                r, c = row + dr * i, col + dc * i
                if 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE and self.board[r, c] == player:
                    line.append((r, c))
                else:
                    break
            # 反方向延伸
            for i in range(1, 5):
                r, c = row - dr * i, col - dc * i
                if 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE and self.board[r, c] == player:
                    line.append((r, c))
                else:
                    break
            if len(line) >= 5:
                self.game_over = True
                self.winner = player
                self.win_line = line
                return

        # 检查是否平局
        if len(self.move_history) == BOARD_SIZE * BOARD_SIZE:
            self.game_over = True
            self.winner = 0  # 0表示平局

    def get_state_planes(self) -> np.ndarray:
        """
        获取当前状态的4通道平面表示（从当前玩家视角）
        通道0: 当前执子方棋子位置
        通道1: 对方棋子位置
        通道2: 上一步落子位置
        通道3: 当前玩家颜色常数平面（全1）
        返回 shape (4, 15, 15)
        """
        planes = np.zeros((4, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)

        cur = self.current_player
        opp = -cur

        # 通道0：当前玩家棋子
        planes[0] = (self.board == cur).astype(np.float32)
        # 通道1：对方棋子
        planes[1] = (self.board == opp).astype(np.float32)
        # 通道2：上一步落子位置
        if self.move_history:
            last_r, last_c = self.move_history[-1]
            planes[2, last_r, last_c] = 1.0
        # 通道3：颜色常数平面
        planes[3] = 1.0

        return planes

    def get_last_move(self) -> Optional[Tuple[int, int]]:
        """获取上一步落子坐标"""
        return self.move_history[-1] if self.move_history else None

    def move_to_str(self, row: int, col: int) -> str:
        """将坐标转换为可读字符串，如 H8"""
        return f"{chr(ord('A') + col)}{row + 1}"

    def __str__(self) -> str:
        """打印棋盘"""
        symbol = {BLACK: '●', WHITE: '○', EMPTY: '·'}
        lines = []
        header = '   ' + ' '.join(chr(ord('A') + i) for i in range(BOARD_SIZE))
        lines.append(header)
        for r in range(BOARD_SIZE):
            row_str = f'{r + 1:2d} ' + ' '.join(symbol[self.board[r, c]] for c in range(BOARD_SIZE))
            lines.append(row_str)
        return '\n'.join(lines)
