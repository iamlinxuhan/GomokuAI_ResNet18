"""
五子棋AI - 图形用户界面 (PyQt6)
支持人机对弈、AI对弈、训练面板
深色主题，棋盘木纹风格
"""

import os
import sys
import time
import json
import threading
import queue
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Tuple, List

import numpy as np
import torch

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QMenuBar, QMenu, QStatusBar, QLabel, QPushButton, QComboBox,
    QGroupBox, QSplitter, QListWidget, QListWidgetItem, QFileDialog,
    QMessageBox, QDockWidget, QProgressBar, QCheckBox, QSlider,
    QDoubleSpinBox, QSpinBox, QGridLayout, QFrame, QSizePolicy,
    QDialog, QDialogButtonBox, QFormLayout, QTabWidget, QTextEdit,
    QScrollArea,
)
from PyQt6.QtGui import (
    QPainter, QPen, QBrush, QColor, QFont, QMouseEvent,
    QPaintEvent, QAction, QIcon, QPalette, QLinearGradient,
    QRadialGradient, QPixmap,
)
from PyQt6.QtCore import (
    Qt, QTimer, QThread, pyqtSignal, QRect, QPoint, QPointF, QSize,
    QEvent, QObject,
)

# matplotlib 图表嵌入
import matplotlib
matplotlib.use('QtAgg')
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
# 设置中文字体（如果可用）
try:
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
except Exception:
    pass

from game import GomokuGame, BOARD_SIZE, BLACK, WHITE, EMPTY, get_center_weights
from network import GomokuNet
from mcts import MCTS, get_best_move_from_probs
from traditional_ai import TraditionalAI
from trainer import Trainer, ReplayBuffer

# 日志
import logging
logger = logging.getLogger(__name__)

# 硬件自动配置（首次导入时检测硬件）
try:
    from hardware_auto_config import get_auto_config, get_hardware_summary, format_hardware_summary
    _HW_CONFIG = get_auto_config()
    _HW_SUMMARY = get_hardware_summary()
    _HW_SUMMARY_TEXT = format_hardware_summary(_HW_SUMMARY)
except Exception:
    _HW_CONFIG = {}
    _HW_SUMMARY = {'gpu': {'available': False, 'name': '未知'}, 'gpu_tier_name': '未知'}
    _HW_SUMMARY_TEXT = "硬件信息检测失败"

# 项目根目录（所有路径以此为基础）
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))


# ============================================================================
# 棋盘绘制组件
# ============================================================================

class BoardWidget(QWidget):
    """棋盘绘制组件"""

    move_made = pyqtSignal(int, int)  # (row, col)
    game_state_changed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.game = GomokuGame()
        self.setMinimumSize(560, 560)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        # 显示参数
        self.board_margin = 40
        self.cell_size = 35
        self.stone_radius = 14
        self.highlight_cell = None  # 鼠标悬停高亮
        self.win_line = []
        self.show_coordinates = True

        # 棋子动画
        self.animation_stone = None
        self.animation_alpha = 0.0
        self.animation_timer = QTimer(self)
        self.animation_timer.timeout.connect(self._animate)
        self.animation_timer.setInterval(20)

        # 颜色方案
        self.board_color = QColor(220, 179, 92)  # 木色
        self.board_border_color = QColor(139, 90, 43)  # 深木色
        self.line_color = QColor(50, 50, 50)
        self.black_stone_color = QColor(30, 30, 30)
        self.white_stone_color = QColor(240, 240, 240)
        self.highlight_color = QColor(255, 215, 0, 100)  # 金色半透明
        self.win_highlight_color = QColor(255, 50, 50, 180)  # 红色半透明

        self.setMouseTracking(True)

    def _board_to_widget(self, row: int, col: int) -> QPoint:
        """棋盘坐标转widget坐标"""
        x = self.board_margin + col * self.cell_size
        y = self.board_margin + row * self.cell_size
        return QPoint(x, y)

    def _widget_to_board(self, x: int, y: int) -> Optional[Tuple[int, int]]:
        """widget坐标转棋盘坐标"""
        col = round((x - self.board_margin) / self.cell_size)
        row = round((y - self.board_margin) / self.cell_size)
        if 0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE:
            # 检查是否足够接近交叉点
            px, py = self._board_to_widget(row, col).x(), self._board_to_widget(row, col).y()
            if abs(x - px) < self.cell_size * 0.45 and abs(y - py) < self.cell_size * 0.45:
                return (row, col)
        return None

    def paintEvent(self, event: QPaintEvent):
        """绘制棋盘"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        self._draw_background(painter)
        self._draw_grid(painter)
        self._draw_star_points(painter)
        self._draw_stones(painter)
        self._draw_win_line(painter)
        self._draw_highlight(painter)
        self._draw_animation_stone(painter)
        self._draw_coordinates(painter)

    def _draw_background(self, painter: QPainter):
        """绘制棋盘背景"""
        # 外边框
        board_width = (BOARD_SIZE - 1) * self.cell_size
        board_rect = QRect(
            self.board_margin - 20,
            self.board_margin - 20,
            board_width + 40,
            board_width + 40
        )

        # 木纹渐变
        tl = board_rect.topLeft()
        br = board_rect.bottomRight()
        gradient = QLinearGradient(QPointF(tl), QPointF(br))
        gradient.setColorAt(0.0, QColor(230, 190, 110))
        gradient.setColorAt(0.5, QColor(210, 165, 85))
        gradient.setColorAt(1.0, QColor(225, 185, 100))

        painter.setPen(QPen(self.board_border_color, 3))
        painter.setBrush(QBrush(gradient))
        painter.drawRoundedRect(board_rect, 8, 8)

    def _draw_grid(self, painter: QPainter):
        """绘制网格线"""
        painter.setPen(QPen(self.line_color, 1))
        board_width = (BOARD_SIZE - 1) * self.cell_size

        for i in range(BOARD_SIZE):
            # 水平线
            y = self.board_margin + i * self.cell_size
            painter.drawLine(
                self.board_margin, y,
                self.board_margin + board_width, y
            )
            # 垂直线
            x = self.board_margin + i * self.cell_size
            painter.drawLine(
                x, self.board_margin,
                x, self.board_margin + board_width
            )

    def _draw_star_points(self, painter: QPainter):
        """绘制星位点"""
        star_points = [(3, 3), (3, 7), (3, 11),
                       (7, 3), (7, 7), (7, 11),
                       (11, 3), (11, 7), (11, 11)]

        painter.setBrush(QBrush(self.line_color))
        painter.setPen(Qt.PenStyle.NoPen)
        for row, col in star_points:
            p = self._board_to_widget(row, col)
            painter.drawEllipse(p, 3, 3)

    def _draw_stones(self, painter: QPainter):
        """绘制棋子"""
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                stone = self.game.board[r, c]
                if stone == EMPTY:
                    continue
                p = self._board_to_widget(r, c)
                if stone == BLACK:
                    self._draw_black_stone(painter, p)
                elif stone == WHITE:
                    self._draw_white_stone(painter, p)

        # 标记最后一步
        if self.game.move_history:
            last_r, last_c = self.game.move_history[-1]
            p = self._board_to_widget(last_r, last_c)
            painter.setPen(QPen(QColor(255, 50, 50), 2))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRect(p.x() - 5, p.y() - 5, 10, 10)

    def _draw_black_stone(self, painter: QPainter, center: QPoint):
        """绘制黑棋（立体效果）"""
        # 阴影
        painter.setBrush(QBrush(QColor(20, 20, 20, 80)))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(center.x() - self.stone_radius + 2,
                            center.y() - self.stone_radius + 2,
                            self.stone_radius * 2, self.stone_radius * 2)

        # 主体
        gradient = QRadialGradient(
            center.x() - 3, center.y() - 4, self.stone_radius * 1.2
        )
        gradient.setColorAt(0.0, QColor(90, 90, 90))
        gradient.setColorAt(0.7, QColor(40, 40, 40))
        gradient.setColorAt(1.0, QColor(15, 15, 15))
        painter.setBrush(QBrush(gradient))
        painter.drawEllipse(center, self.stone_radius, self.stone_radius)

    def _draw_white_stone(self, painter: QPainter, center: QPoint):
        """绘制白棋（立体效果）"""
        # 阴影
        painter.setBrush(QBrush(QColor(50, 50, 50, 60)))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(center.x() - self.stone_radius + 2,
                            center.y() - self.stone_radius + 2,
                            self.stone_radius * 2, self.stone_radius * 2)

        # 主体
        gradient = QRadialGradient(
            center.x() - 3, center.y() - 4, self.stone_radius * 1.2
        )
        gradient.setColorAt(0.0, QColor(255, 255, 255))
        gradient.setColorAt(0.6, QColor(230, 230, 230))
        gradient.setColorAt(1.0, QColor(180, 180, 180))
        painter.setBrush(QBrush(gradient))
        painter.setPen(QPen(QColor(150, 150, 150), 1))
        painter.drawEllipse(center, self.stone_radius, self.stone_radius)

    def _draw_win_line(self, painter: QPainter):
        """绘制获胜连线高亮"""
        if self.game.win_line and len(self.game.win_line) >= 5:
            painter.setPen(QPen(self.win_highlight_color, 3))
            for r, c in self.game.win_line:
                p = self._board_to_widget(r, c)
                painter.drawEllipse(p, self.stone_radius + 3, self.stone_radius + 3)

    def _draw_highlight(self, painter: QPainter):
        """绘制鼠标悬停高亮"""
        if self.highlight_cell and not self.game.game_over:
            r, c = self.highlight_cell
            if self.game.is_legal_move(r, c):
                p = self._board_to_widget(r, c)
                painter.setBrush(QBrush(self.highlight_color))
                painter.setPen(Qt.PenStyle.NoPen)
                painter.drawEllipse(p, self.stone_radius, self.stone_radius)

    def _draw_animation_stone(self, painter: QPainter):
        """绘制动画中的棋子"""
        if self.animation_stone and self.animation_alpha > 0:
            r, c, player = self.animation_stone
            p = self._board_to_widget(r, c)
            painter.save()
            painter.setOpacity(self.animation_alpha)
            if player == BLACK:
                self._draw_black_stone(painter, p)
            else:
                self._draw_white_stone(painter, p)
            painter.restore()

    def _draw_coordinates(self, painter: QPainter):
        """绘制坐标标签"""
        painter.setPen(QPen(QColor(80, 80, 80)))
        font = QFont('Arial', 8)
        painter.setFont(font)

        for i in range(BOARD_SIZE):
            # 列标签 A-O
            x = self.board_margin + i * self.cell_size
            y = self.board_margin - 18
            label = chr(ord('A') + i)
            painter.drawText(QRect(x - 10, y, 20, 15), Qt.AlignmentFlag.AlignCenter, label)

            # 行标签 1-15
            x = self.board_margin - 25
            y = self.board_margin + i * self.cell_size - 7
            painter.drawText(QRect(x, y, 20, 15), Qt.AlignmentFlag.AlignCenter, str(i + 1))

    def _animate(self):
        """棋子动画"""
        if self.animation_alpha < 1.0:
            self.animation_alpha = min(1.0, self.animation_alpha + 0.1)
            self.update()
        else:
            self.animation_timer.stop()
            self.animation_stone = None
            self.update()

    def mouseMoveEvent(self, event: QMouseEvent):
        """鼠标移动事件"""
        pos = event.position()
        cell = self._widget_to_board(int(pos.x()), int(pos.y()))
        if cell != self.highlight_cell:
            self.highlight_cell = cell
            self.update()

    def mousePressEvent(self, event: QMouseEvent):
        """鼠标点击事件"""
        if event.button() == Qt.MouseButton.LeftButton:
            pos = event.position()
            cell = self._widget_to_board(int(pos.x()), int(pos.y()))
            if cell:
                self.move_made.emit(cell[0], cell[1])

    def make_move(self, row: int, col: int, animate: bool = True) -> bool:
        """落子"""
        player = self.game.current_player
        if self.game.make_move(row, col):
            if animate:
                self.animation_stone = (row, col, player)
                self.animation_alpha = 0.0
                self.animation_timer.start()
            self.game_state_changed.emit()
            self.update()
            return True
        return False

    def undo_move(self) -> bool:
        """悔棋"""
        if self.game.undo_move():
            self.game_state_changed.emit()
            self.update()
            return True
        return False

    def reset_board(self):
        """重置棋盘"""
        self.game.reset()
        self.highlight_cell = None
        self.win_line = []
        self.animation_stone = None
        self.animation_alpha = 0.0
        self.game_state_changed.emit()
        self.update()

    def resizeEvent(self, event):
        """窗口大小变化时调整棋盘大小"""
        size = min(self.width(), self.height())
        self.cell_size = max(20, (size - self.board_margin * 2) // (BOARD_SIZE - 1))
        self.stone_radius = max(8, self.cell_size // 2 - 2)
        super().resizeEvent(event)


# ============================================================================
# AI推理线程
# ============================================================================

class AIThread(QThread):
    """AI推理后台线程"""
    move_ready = pyqtSignal(int, int)  # (row, col)
    thinking_started = pyqtSignal()
    thinking_finished = pyqtSignal()

    def __init__(self, model: GomokuNet, game: GomokuGame,
                 num_simulations: int = 400, device: torch.device = None):
        super().__init__()
        self.model = model
        self.game = game.clone()
        self.num_simulations = num_simulations
        self.device = device or torch.device('cpu')

    def run(self):
        """执行AI推理"""
        self.thinking_started.emit()
        try:
            mcts = MCTS(
                model=self.model,
                num_simulations=self.num_simulations,
                device=self.device
            )
            # Dirichlet噪声保证非确定性探索，避免每局棋步完全相同
            action_probs, _ = mcts.search(self.game, temperature=0.1,
                                          add_dirichlet_noise=True)
            move = get_best_move_from_probs(action_probs, self.game, deterministic=True)
            if move and move != (-1, -1):
                self.move_ready.emit(move[0], move[1])
        except Exception as e:
            print(f"AI推理错误: {e}")
        finally:
            self.thinking_finished.emit()


# ============================================================================
# 多局并发AI对弈线程（训练数据收集模式专用）
# ============================================================================

class MultiGameAIThread(QThread):
    """
    5局AI vs AI并发对弈，不阻塞UI。
    其中第0局作为参考渲染到棋盘，其余4局后台静默运行。
    """
    reference_move = pyqtSignal(int, int)        # (row, col) 参考局的每一步
    reference_reset = pyqtSignal()                # 参考局重置
    all_games_completed = pyqtSignal(list)        # 全部5局的数据 [(state, policy, value, source), ...]
    game_progress = pyqtSignal(int, int)          # (completed, total)
    thinking_status = pyqtSignal(str)             # 状态消息

    def __init__(self, model: GomokuNet, num_simulations: int = 400,
                 device: torch.device = None, num_games: int = 5,
                 replay_buffer: 'ReplayBuffer' = None):
        super().__init__()
        self.model = model
        self.num_simulations = num_simulations
        self.device = device or torch.device('cpu')
        self.num_games = num_games
        self.replay_buffer = replay_buffer

    def run(self):
        all_data = []
        completed_count = [0]  # 用 list 实现可变闭包捕获
        lock = threading.Lock()
        move_queue = queue.Queue()

        def play_one_game(game_id: int) -> list:
            """在子线程中运行一局AI vs AI"""
            game = GomokuGame()
            mcts = MCTS(model=self.model, num_simulations=self.num_simulations,
                        device=self.device)
            game_data = []

            while not game.game_over:
                action_probs, _ = mcts.search(game, temperature=0.1,
                                              add_dirichlet_noise=True)
                state = game.get_state_planes()
                game_data.append((state, action_probs, None, 'self'))

                move = get_best_move_from_probs(action_probs, game, deterministic=True)
                if move == (-1, -1):
                    break
                game.make_move(move[0], move[1])

                # 参考局(0号)每一步都发送到队列用于UI渲染
                if game_id == 0:
                    move_queue.put((move[0], move[1]))

            # 价值回填
            result = game.winner if game.winner else 0
            for i in range(len(game_data)):
                state, policy, _, source = game_data[i]
                if result == 0:
                    v = 0.0
                elif result == BLACK:
                    v = 1.0 if i % 2 == 0 else -1.0
                else:
                    v = -1.0 if i % 2 == 0 else 1.0
                game_data[i] = (state, policy, v, source)

            with lock:
                all_data.extend(game_data)
                completed_count[0] += 1
                self.game_progress.emit(completed_count[0], self.num_games)

            return game_data

        self.thinking_status.emit(f"开始{self.num_games}局并发对弈 (参考局渲染中)...")

        with ThreadPoolExecutor(max_workers=self.num_games) as executor:
            futures = [executor.submit(play_one_game, i) for i in range(self.num_games)]

            # 主循环：在等待所有对弈完成的同时，处理参考局的UI更新
            while completed_count[0] < self.num_games:
                try:
                    row, col = move_queue.get(timeout=0.08)
                    self.reference_move.emit(row, col)
                except queue.Empty:
                    pass

        self.thinking_status.emit(
            f"{self.num_games}局对弈完成，共收集{len(all_data)}条训练数据"
        )
        self.all_games_completed.emit(all_data)


# ============================================================================
# 训练线程
# ============================================================================

class TrainingThread(QThread):
    """训练后台线程"""
    status_update = pyqtSignal(dict)
    log_message = pyqtSignal(str)

    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self.trainer: Optional[Trainer] = None

    def run(self):
        """运行训练（后台线程跑 train_loop，本线程轮询状态）"""
        try:
            self.trainer = Trainer(self.config)
            games_per_iter = self.config.get('training', {}).get('games_per_iteration', 5)
            self._train_error = None

            def _run_train():
                try:
                    self.trainer.train_loop(games_per_iteration=games_per_iter)
                except Exception as e:
                    self._train_error = e

            bg = threading.Thread(target=_run_train, daemon=True)
            bg.start()

            # 每 2 秒轮询训练状态并发送到 GUI
            last_emit_total = -1
            while bg.is_alive():
                if self.trainer:
                    status = self.trainer.get_status()
                    self.status_update.emit(status)
                    total = status.get('total_games', 0)
                    if total > last_emit_total:
                        self.log_message.emit(
                            f"已对弈 {total} 局 | 经验池 {status.get('buffer_size', 0)} | "
                            f"训练步 {status.get('global_step', 0)}"
                        )
                        last_emit_total = total
                self.msleep(2000)

            bg.join()
            if self._train_error:
                raise self._train_error
            if self.trainer:
                self.status_update.emit(self.trainer.get_status())
        except Exception as e:
            self.log_message.emit(f"训练错误: {e}")

    def stop_training(self):
        if self.trainer:
            self.trainer.stop()

    def pause_training(self):
        if self.trainer:
            self.trainer.pause()

    def resume_training(self):
        if self.trainer:
            self.trainer.resume()


# ============================================================================
# 训练面板对话框
# ============================================================================

class TrainingPanel(QDialog):
    """训练监控面板 v2.0 — 新增硬件信息显示和自动配置推荐"""

    def __init__(self, parent=None, config: dict = None):
        super().__init__(parent)
        self.setWindowTitle("训练监控面板 v2.0")
        self.setMinimumSize(800, 700)
        self.config = config or {}
        self.training_thread: Optional[TrainingThread] = None
        self._weight_adjusting = False  # 防止采样权重联动时的递归信号

        # 加载硬件自动配置
        try:
            from hardware_auto_config import get_hardware_summary, get_auto_config, format_hardware_summary, format_auto_config_summary
            self.hw_summary = get_hardware_summary()
            self.auto_config = get_auto_config()
            self.hw_summary_text = format_hardware_summary(self.hw_summary)
            self.auto_config_text = format_auto_config_summary(self.auto_config)
        except Exception:
            self.hw_summary = {'gpu': {'available': False, 'name': '未知'}, 'gpu_tier_name': '未知', 'system_memory_mb': 0}
            self.auto_config = {}
            self.hw_summary_text = "硬件检测失败"
            self.auto_config_text = "配置生成失败"

        self._init_ui()
        self._start_status_update()

    def _init_ui(self):
        layout = QVBoxLayout(self)

        # 标签页
        tabs = QTabWidget()

        # --- 训练控制 ---
        control_tab = QWidget()
        control_layout = QVBoxLayout(control_tab)

        # 模式选择
        mode_group = QGroupBox("训练模式")
        mode_layout = QHBoxLayout()
        self.cb_self = QCheckBox("自我对弈 (self)")
        self.cb_self.setChecked(True)
        self.cb_trad = QCheckBox("传统AI对抗 (trad)")
        self.cb_trad.setChecked(True)
        self.cb_human = QCheckBox("人机对弈 (human)")
        self.cb_human.setChecked(False)
        # 三个勾选框共用同一个联动处理：取消勾选时对应权重归零再分配
        self.cb_self.toggled.connect(lambda c: self._on_mode_toggled('self', c))
        self.cb_trad.toggled.connect(lambda c: self._on_mode_toggled('trad', c))
        self.cb_human.toggled.connect(lambda c: self._on_mode_toggled('human', c))
        mode_layout.addWidget(self.cb_self)
        mode_layout.addWidget(self.cb_trad)
        mode_layout.addWidget(self.cb_human)
        mode_group.setLayout(mode_layout)
        control_layout.addWidget(mode_group)

        # 超参数
        param_group = QGroupBox("超参数")
        param_layout = QGridLayout()

        # 学习率 - 滑动条
        param_layout.addWidget(QLabel("学习率:"), 0, 0)
        self.slider_lr = QSlider(Qt.Orientation.Horizontal)
        self.slider_lr.setRange(1, 1000)  # 0.0001 ~ 0.1
        self.slider_lr.setValue(200)       # 0.02
        self.slider_lr.setSingleStep(10)
        self.lbl_lr = QLabel("0.02000")
        self.lbl_lr.setMinimumWidth(60)
        self.slider_lr.valueChanged.connect(
            lambda v: self.lbl_lr.setText(f"{v / 10000:.5f}"))
        param_layout.addWidget(self.slider_lr, 0, 1)
        param_layout.addWidget(self.lbl_lr, 0, 2)

        # 批量大小 - 滑动条
        param_layout.addWidget(QLabel("批量大小:"), 0, 3)
        self.slider_batch = QSlider(Qt.Orientation.Horizontal)
        self.slider_batch.setRange(64, 4096)
        self.slider_batch.setValue(1024)
        self.slider_batch.setSingleStep(64)
        self.slider_batch.setTickInterval(512)
        self.lbl_batch = QLabel("1024")
        self.lbl_batch.setMinimumWidth(50)
        self.slider_batch.valueChanged.connect(
            lambda v: self.lbl_batch.setText(str(v)))
        param_layout.addWidget(self.slider_batch, 0, 4)
        param_layout.addWidget(self.lbl_batch, 0, 5)

        # MCTS模拟次数 - 滑动条（默认50，自动调参逐步增加）
        param_layout.addWidget(QLabel("MCTS模拟:"), 1, 0)
        self.slider_mcts = QSlider(Qt.Orientation.Horizontal)
        self.slider_mcts.setRange(10, 800)
        self.slider_mcts.setValue(50)
        self.slider_mcts.setSingleStep(50)
        self.slider_mcts.setTickInterval(100)
        self.slider_mcts.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.lbl_mcts_val = QLabel("50")
        self.lbl_mcts_val.setMinimumWidth(50)
        self.slider_mcts.valueChanged.connect(
            lambda v: self.lbl_mcts_val.setText(str(v)))
        param_layout.addWidget(self.slider_mcts, 1, 1)
        param_layout.addWidget(self.lbl_mcts_val, 1, 2)

        # 总训练步数 - 滑动条
        param_layout.addWidget(QLabel("总训练步数:"), 1, 3)
        self.slider_steps = QSlider(Qt.Orientation.Horizontal)
        self.slider_steps.setRange(10000, 1000000)
        self.slider_steps.setValue(500000)
        self.slider_steps.setSingleStep(10000)
        self.slider_steps.setTickInterval(100000)
        self.lbl_steps = QLabel("500000")
        self.lbl_steps.setMinimumWidth(60)
        self.slider_steps.valueChanged.connect(
            lambda v: self.lbl_steps.setText(str(v)))
        param_layout.addWidget(self.slider_steps, 1, 4)
        param_layout.addWidget(self.lbl_steps, 1, 5)

        # 自动保存间隔(分钟) - 滑动条
        param_layout.addWidget(QLabel("自动保存间隔:"), 2, 0)
        self.slider_save = QSlider(Qt.Orientation.Horizontal)
        self.slider_save.setRange(1, 60)
        self.slider_save.setValue(5)
        self.slider_save.setSingleStep(1)
        self.slider_save.setTickInterval(5)
        self.slider_save.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.lbl_save = QLabel("5分钟")
        self.lbl_save.setMinimumWidth(50)
        self.slider_save.valueChanged.connect(
            lambda v: self.lbl_save.setText(f"{v}分钟"))
        param_layout.addWidget(self.slider_save, 2, 1)
        param_layout.addWidget(self.lbl_save, 2, 2)

        # 自动调参评估局数
        param_layout.addWidget(QLabel("自动调参局数:"), 2, 3)
        self.slider_eval_games = QSlider(Qt.Orientation.Horizontal)
        self.slider_eval_games.setRange(10, 200)
        self.slider_eval_games.setValue(100)
        self.slider_eval_games.setSingleStep(10)
        self.slider_eval_games.setTickInterval(20)
        self.lbl_eval_games = QLabel("100局")
        self.lbl_eval_games.setMinimumWidth(50)
        self.slider_eval_games.valueChanged.connect(
            lambda v: self.lbl_eval_games.setText(f"{v}局"))
        param_layout.addWidget(self.slider_eval_games, 2, 4)
        param_layout.addWidget(self.lbl_eval_games, 2, 5)

        # 传统AI初始深度 — 训练时传统AI对抗的起始搜索深度（默认1，自动调参逐步增加）
        param_layout.addWidget(QLabel("传统AI初始深度:"), 3, 0)
        self.slider_trad_init_depth = QSlider(Qt.Orientation.Horizontal)
        self.slider_trad_init_depth.setRange(1, 8)
        self.slider_trad_init_depth.setValue(1)
        self.slider_trad_init_depth.setSingleStep(1)
        self.slider_trad_init_depth.setTickInterval(1)
        self.slider_trad_init_depth.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.lbl_trad_init_depth = QLabel("1")
        self.lbl_trad_init_depth.setMinimumWidth(30)
        self.slider_trad_init_depth.valueChanged.connect(
            lambda v: self.lbl_trad_init_depth.setText(str(v)))
        param_layout.addWidget(self.slider_trad_init_depth, 3, 1)
        param_layout.addWidget(self.lbl_trad_init_depth, 3, 2)

        # 每轮对弈局数 — 每个训练迭代生成多少局对局数据（自动调参可调整）
        param_layout.addWidget(QLabel("每轮对弈局数:"), 3, 3)
        self.slider_parallel = QSlider(Qt.Orientation.Horizontal)
        self.slider_parallel.setRange(1, 20)
        self.slider_parallel.setValue(5)
        self.slider_parallel.setSingleStep(1)
        self.slider_parallel.setTickInterval(2)
        self.slider_parallel.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.lbl_parallel = QLabel("5")
        self.lbl_parallel.setMinimumWidth(30)
        self.slider_parallel.valueChanged.connect(
            lambda v: self.lbl_parallel.setText(str(v)))
        param_layout.addWidget(self.slider_parallel, 3, 4)
        param_layout.addWidget(self.lbl_parallel, 3, 5)

        param_group.setLayout(param_layout)
        control_layout.addWidget(param_group)

        # 混合比例
        mix_group = QGroupBox("数据采样权重")
        mix_layout = QGridLayout()
        mix_layout.addWidget(QLabel("Self:"), 0, 0)
        self.slider_self = QSlider(Qt.Orientation.Horizontal)
        self.slider_self.setRange(0, 100)
        self.slider_self.setValue(60)
        self.lbl_self = QLabel("60%")
        mix_layout.addWidget(self.slider_self, 0, 1)
        mix_layout.addWidget(self.lbl_self, 0, 2)

        mix_layout.addWidget(QLabel("Trad:"), 1, 0)
        self.slider_trad = QSlider(Qt.Orientation.Horizontal)
        self.slider_trad.setRange(0, 100)
        self.slider_trad.setValue(30)
        self.lbl_trad = QLabel("30%")
        mix_layout.addWidget(self.slider_trad, 1, 1)
        mix_layout.addWidget(self.lbl_trad, 1, 2)

        mix_layout.addWidget(QLabel("Human:"), 2, 0)
        self.slider_human = QSlider(Qt.Orientation.Horizontal)
        self.slider_human.setRange(0, 100)
        self.slider_human.setValue(10)
        self.lbl_human = QLabel("10%")
        mix_layout.addWidget(self.slider_human, 2, 1)
        mix_layout.addWidget(self.lbl_human, 2, 2)
        mix_group.setLayout(mix_layout)
        control_layout.addWidget(mix_group)

        # 连接滑块信号 — 联动绑定：调整任意一项，其余按比例补偿使总和=100
        self.slider_self.valueChanged.connect(self._on_weight_self_changed)
        self.slider_trad.valueChanged.connect(self._on_weight_trad_changed)
        self.slider_human.valueChanged.connect(self._on_weight_human_changed)

        # 控制按钮
        btn_layout = QHBoxLayout()
        self.btn_start = QPushButton("开始训练")
        self.btn_start.clicked.connect(self._start_training)
        self.btn_pause = QPushButton("暂停")
        self.btn_pause.clicked.connect(self._pause_training)
        self.btn_pause.setEnabled(False)
        self.btn_stop = QPushButton("停止训练")
        self.btn_stop.clicked.connect(self._stop_training)
        self.btn_stop.setEnabled(False)
        self.btn_save = QPushButton("保存检查点")
        self.btn_save.clicked.connect(self._save_checkpoint)

        btn_layout.addWidget(self.btn_start)
        btn_layout.addWidget(self.btn_pause)
        btn_layout.addWidget(self.btn_stop)
        btn_layout.addWidget(self.btn_save)
        control_layout.addLayout(btn_layout)

        tabs.addTab(control_tab, "训练控制")

        # --- 硬件信息 ---
        hw_tab = QWidget()
        hw_layout = QVBoxLayout(hw_tab)

        # 硬件检测报告
        hw_text = QTextEdit()
        hw_text.setReadOnly(True)
        hw_text.setFont(QFont('Consolas', 9))
        hw_text.setText(self.hw_summary_text)
        hw_layout.addWidget(QLabel("📋 硬件检测报告:"))
        hw_layout.addWidget(hw_text)

        # 推荐配置
        config_text = QTextEdit()
        config_text.setReadOnly(True)
        config_text.setFont(QFont('Consolas', 9))
        config_text.setText(self.auto_config_text)
        hw_layout.addWidget(QLabel("⚙️ 自动推荐配置:"))
        hw_layout.addWidget(config_text)

        # 应用推荐配置按钮
        btn_apply_hw = QPushButton("应用推荐配置到训练参数")
        btn_apply_hw.clicked.connect(self._apply_hardware_config)
        hw_layout.addWidget(btn_apply_hw)

        # GPU基准测试按钮
        btn_benchmark = QPushButton("运行GPU基准测试（约10秒）")
        btn_benchmark.clicked.connect(self._run_gpu_benchmark)
        hw_layout.addWidget(btn_benchmark)

        tabs.addTab(hw_tab, "硬件信息")

        # --- 监控（含图表） ---
        monitor_tab = QWidget()
        monitor_layout = QVBoxLayout(monitor_tab)

        # 顶部信息栏（紧凑横排）
        info_bar = QHBoxLayout()
        self.lbl_step = QLabel("步数: 0")
        self.lbl_step.setFont(QFont('Arial', 10, QFont.Weight.Bold))
        info_bar.addWidget(self.lbl_step)
        self.lbl_buffer = QLabel("经验池: 0")
        info_bar.addWidget(self.lbl_buffer)
        self.lbl_loss = QLabel("损失: -")
        info_bar.addWidget(self.lbl_loss)
        self.lbl_lr = QLabel("LR: -")
        info_bar.addWidget(self.lbl_lr)
        self.lbl_trad_wr = QLabel("胜率: -")
        info_bar.addWidget(self.lbl_trad_wr)
        self.lbl_trad_depth = QLabel("深度: -")
        info_bar.addWidget(self.lbl_trad_depth)
        self.lbl_elo = QLabel("ELO: -")
        info_bar.addWidget(self.lbl_elo)
        info_bar.addStretch()
        monitor_layout.addLayout(info_bar)

        # 图表区域（3个子图：损失、胜率、自动参数）
        self.chart_widget = QWidget()
        self.chart_layout = QVBoxLayout(self.chart_widget)
        self.chart_layout.setContentsMargins(0, 0, 0, 0)

        # 创建 matplotlib 图表
        self.fig = Figure(figsize=(8, 4.5), dpi=100)
        self.fig.patch.set_facecolor('#2b2b2b')  # 深色背景

        # 损失曲线 (左上)
        self.ax_loss = self.fig.add_subplot(2, 2, 1)
        self._style_ax(self.ax_loss, '损失曲线', '步数', '损失')
        self.line_total, = self.ax_loss.plot([], [], 'r-', lw=1.5, label='总损失')
        self.line_policy, = self.ax_loss.plot([], [], 'g-', lw=0.8, alpha=0.7, label='策略')
        self.line_value, = self.ax_loss.plot([], [], 'b-', lw=0.8, alpha=0.7, label='价值')
        self.ax_loss.legend(fontsize=7, loc='upper right')

        # 胜率曲线 (右上)
        self.ax_wr = self.fig.add_subplot(2, 2, 2)
        self._style_ax(self.ax_wr, 'NN vs 传统AI 胜率', '局数', '胜率')
        self.line_wr, = self.ax_wr.plot([], [], 'c-', lw=1.5)
        self.ax_wr.axhline(0.20, color='y', ls='--', lw=0.8, alpha=0.5, label='目标')
        self.ax_wr.set_ylim(0, 1.0)
        self.ax_wr.legend(fontsize=7, loc='upper left')

        # MCTS/温度曲线 (左下)
        self.ax_mcts = self.fig.add_subplot(2, 2, 3)
        self._style_ax(self.ax_mcts, '自动参数趋势', '步数', 'MCTS模拟')
        self.line_mcts, = self.ax_mcts.plot([], [], 'm-', lw=1.5, label='MCTS')
        self.ax_mcts_twin = self.ax_mcts.twinx()
        self.ax_mcts_twin.set_ylabel('温度', color='orange', fontsize=8)
        self.line_temp, = self.ax_mcts_twin.plot([], [], 'orange', lw=1, label='温度')
        self.ax_mcts.legend(fontsize=7, loc='upper left')

        # 学习率 (右下)
        self.ax_lr = self.fig.add_subplot(2, 2, 4)
        self._style_ax(self.ax_lr, '学习率', '步数', 'LR')
        self.line_lr, = self.ax_lr.plot([], [], 'w-', lw=1.5)

        self.fig.tight_layout(pad=1.5)

        self.chart_canvas = FigureCanvas(self.fig)
        self.chart_canvas.setMinimumHeight(280)
        self.chart_layout.addWidget(self.chart_canvas)
        monitor_layout.addWidget(self.chart_widget, stretch=1)

        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximumHeight(18)
        monitor_layout.addWidget(self.progress_bar)

        # 日志
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(120)
        self.log_text.setStyleSheet("background-color: #1e1e1e; color: #d4d4d4; font-size: 11px;")
        monitor_layout.addWidget(QLabel("训练日志:"))
        monitor_layout.addWidget(self.log_text)

        tabs.addTab(monitor_tab, "监控")

        layout.addWidget(tabs)

    def _start_training(self):
        """开始训练"""
        # 防止重复启动
        if self.training_thread and self.training_thread.isRunning():
            self._append_log("训练已在运行中")
            return

        modes = []
        if self.cb_self.isChecked():
            modes.append('self')
        if self.cb_trad.isChecked():
            modes.append('trad')
        if self.cb_human.isChecked():
            modes.append('human')

        if not modes:
            QMessageBox.warning(self, "警告", "请至少选择一种训练模式")
            return

        # 从硬件自动配置获取补充字段
        hw_config = getattr(self, 'auto_config', {})
        hw_net = hw_config.get('network', {})
        hw_training = hw_config.get('training', {})

        # 计算实际采样权重：未勾选的模式权重强制归 0，已勾选的按比例重分配
        mode_checked = {
            'self':  self.cb_self.isChecked(),
            'trad':  self.cb_trad.isChecked(),
            'human': self.cb_human.isChecked(),
        }
        raw_weights = {
            'self':  self.slider_self.value() / 100.0,
            'trad':  self.slider_trad.value() / 100.0,
            'human': self.slider_human.value() / 100.0,
        }
        # 未勾选的强制归零
        checked_sum = sum(raw_weights[m] for m in mode_checked if mode_checked[m])
        if checked_sum > 0:
            w_self  = raw_weights['self']  / checked_sum if mode_checked['self']  else 0.0
            w_trad  = raw_weights['trad']  / checked_sum if mode_checked['trad']  else 0.0
            w_human = raw_weights['human'] / checked_sum if mode_checked['human'] else 0.0
        else:
            w_self = w_trad = w_human = 0.0

        config = {
            # 网络结构（硬件自适应）
            'network': {
                'num_filters': hw_net.get('num_filters', 128),
                'num_res_blocks': hw_net.get('num_res_blocks', 7),
                'policy_channels': hw_net.get('policy_channels', 32),
                'value_channels': hw_net.get('value_channels', 32),
                'value_hidden': hw_net.get('value_hidden', 128),
            },
            'training': {
                'modes': modes,
                'learning_rate': self.slider_lr.value() / 10000.0,
                'batch_size': self.slider_batch.value(),
                'num_mcts_simulations': self.slider_mcts.value(),
                'total_steps': self.slider_steps.value(),
                'use_cuda': torch.cuda.is_available(),
                # 以下字段从硬件推荐配置继承默认值
                'mixed_precision': hw_training.get('mixed_precision', True),
                'gradient_accumulation_steps': hw_training.get('gradient_accumulation_steps', 1),
                'eval_interval_steps': hw_training.get('eval_interval_steps', 5000),
                'num_workers': hw_training.get('num_workers', 4),
                'model_dir': os.path.join(PROJECT_DIR, 'models'),
                'log_dir': os.path.join(PROJECT_DIR, 'logs'),
                'save_interval_minutes': self.slider_save.value(),
                'auto_eval_games': self.slider_eval_games.value(),
                'games_per_iteration': self.slider_parallel.value(),
                # 自我对弈
                'self_play': {
                    'num_processes': hw_training.get('self_play', {}).get('num_processes', 4),
                    'mcts_simulations': self.slider_mcts.value(),
                },
                # 经验回放池（含 PER）
                'replay_buffer': {
                    'capacity': 2000000,
                    'prioritized_alpha': 0.6,
                    'prioritized_beta': 0.4,
                    'sampling_weights': {
                        'self': w_self,
                        'trad': w_trad,
                        'human': w_human,
                    },
                },
                # 数据增强
                'augmentation': {
                    'enabled': hw_training.get('augmentation', {}).get('enabled', True),
                    'symmetry': True,
                },
                # 传统AI
                'traditional': {
                    'initial_depth': self.slider_trad_init_depth.value(),
                    'depth_range': [1, 8],
                    'games_per_adjust': 10,
                    'target_win_rate': 0.20,
                    'win_rate_window': 100,
                    'adjust_step': 1,
                },
                # 人机对弈
                'human': {
                    'enabled': mode_checked['human'],
                    'temperature_for_human_moves': 0.1,
                },
            },
        }

        self.training_thread = TrainingThread(config)
        self.training_thread.status_update.connect(self._update_status)
        self.training_thread.log_message.connect(self._append_log)
        self.training_thread.start()

        self.btn_start.setEnabled(False)
        self.btn_pause.setEnabled(True)
        self.btn_stop.setEnabled(True)

        # 训练启动后隐藏主窗口GUI，只保留监控面板
        main_window = self.parent()
        if main_window:
            main_window.hide()

        self._append_log("训练已启动...")
        self._append_log(f"模式: {modes}")
        self._append_log(f"设备: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
        self._append_log(f"MCTS模拟: {self.slider_mcts.value()}次 | "
                         f"学习率: {self.slider_lr.value() / 10000:.5f} | "
                         f"自动保存: 每{self.slider_save.value()}分钟")
        self._append_log(f"自动调参: 每{self.slider_eval_games.value()}局评估")

    def _pause_training(self):
        """暂停/恢复训练"""
        if self.training_thread:
            if self.training_thread.trainer and not self.training_thread.trainer.paused:
                self.training_thread.pause_training()
                self.btn_pause.setText("恢复")
                self._append_log("训练已暂停")
            else:
                self.training_thread.resume_training()
                self.btn_pause.setText("暂停")
                self._append_log("训练已恢复")

    def _stop_training(self):
        """停止训练"""
        if self.training_thread:
            self.training_thread.stop_training()
            self._append_log("训练正在停止...")
        self.btn_start.setEnabled(True)
        self.btn_pause.setEnabled(False)
        self.btn_stop.setEnabled(False)

        # 训练结束后恢复显示主窗口GUI
        main_window = self.parent()
        if main_window:
            main_window.show()

    def _save_checkpoint(self):
        """保存检查点"""
        if self.training_thread and self.training_thread.trainer:
            self.training_thread.trainer._save_checkpoint()
            self._append_log("模型已覆盖保存")

    def _update_status(self, status: dict):
        """更新训练状态显示（紧凑信息栏 + 图表 + 控制滑条同步）"""
        # ── 信息栏 ──
        self.lbl_step.setText(f"步数: {status.get('global_step', 0)}")
        self.lbl_buffer.setText(f"缓冲: {status.get('buffer_size', 0)}")
        self.lbl_trad_depth.setText(f"深度: {status.get('trad_depth', '-')}")
        wr = status.get('trad_win_rate', 0)
        self.lbl_trad_wr.setText(f"胜率: {wr:.0%}")
        self.lbl_elo.setText(f"ELO: {status.get('elo_current', 0):.0f}")

        if status.get('loss_history'):
            last = status['loss_history'][-1]
            self.lbl_loss.setText(f"损失: {last.get('total_loss', 0):.3f}")
            self.lbl_lr.setText(f"LR: {last.get('learning_rate', 0):.5f}")

        total_steps = self.slider_steps.value()
        if total_steps > 0:
            self.progress_bar.setValue(
                int(status.get('global_step', 0) / total_steps * 100)
            )

        # ── 同步控制滑条（自动调参结果 → UI） ──
        ap = status.get('auto_params', {})
        if ap:
            # MCTS 模拟次数：自动调参可能调整了它
            auto_mcts = ap.get('mcts_simulations')
            if auto_mcts and auto_mcts != self.slider_mcts.value():
                self.slider_mcts.blockSignals(True)
                self.slider_mcts.setValue(auto_mcts)
                self.lbl_mcts_val.setText(str(auto_mcts))
                self.slider_mcts.blockSignals(False)

            # 每轮对弈局数：自动调参根据经验池调整
            auto_games = ap.get('games_per_iteration')
            if auto_games and auto_games != self.slider_parallel.value():
                self.slider_parallel.blockSignals(True)
                self.slider_parallel.setValue(auto_games)
                self.lbl_parallel.setText(str(auto_games))
                self.slider_parallel.blockSignals(False)

        # 传统AI深度：自动难度调控可能调整了它
        trad_depth = status.get('trad_depth')
        if trad_depth and trad_depth != self.slider_trad_init_depth.value():
            self.slider_trad_init_depth.blockSignals(True)
            self.slider_trad_init_depth.setValue(trad_depth)
            self.lbl_trad_init_depth.setText(str(trad_depth))
            self.slider_trad_init_depth.blockSignals(False)

        # ── 刷新图表 ──
        self._update_charts(status)

    @staticmethod
    def _style_ax(ax, title, xlabel, ylabel):
        """给图表坐标轴设置深色主题样式"""
        ax.set_facecolor('#2b2b2b')
        ax.spines['bottom'].set_color('#555')
        ax.spines['top'].set_color('#555')
        ax.spines['left'].set_color('#555')
        ax.spines['right'].set_color('#555')
        ax.tick_params(colors='#aaa', labelsize=7)
        ax.set_title(title, color='#ddd', fontsize=9)
        ax.set_xlabel(xlabel, color='#aaa', fontsize=7)
        ax.set_ylabel(ylabel, color='#aaa', fontsize=7)
        ax.grid(True, alpha=0.15, linestyle='-')

    def _update_charts(self, status: dict):
        """更新图表数据"""
        loss_hist = status.get('loss_history', [])
        wr_hist = status.get('win_rate_history', [])

        # ── 损失曲线 ──
        if len(loss_hist) >= 2:
            steps = [h.get('step', 0) for h in loss_hist]
            totals = [h.get('total_loss', 0) for h in loss_hist]
            policies = [h.get('policy_loss', 0) for h in loss_hist]
            values = [h.get('value_loss', 0) for h in loss_hist]
            self.line_total.set_data(steps, totals)
            self.line_policy.set_data(steps, policies)
            self.line_value.set_data(steps, values)
            self.ax_loss.relim()
            self.ax_loss.autoscale_view()
            # 只显示最近 200 步的细节
            if len(steps) > 200:
                self.ax_loss.set_xlim(steps[-200], steps[-1])

        # ── 胜率曲线 ──
        if len(wr_hist) >= 2:
            idx = list(range(len(wr_hist)))
            win_rates = []
            for h in wr_hist:
                if 'win_rate' in h:
                    win_rates.append(h['win_rate'])
                elif 'nn_win_rate' in h:
                    win_rates.append(h['nn_win_rate'])
                else:
                    win_rates.append(0.5)
            self.line_wr.set_data(idx, win_rates)
            self.ax_wr.relim()
            self.ax_wr.autoscale_view()

        # ── 自动参数趋势（MCTS + 温度） ──
        ap = status.get('auto_params', {})
        if ap:
            # 从 loss_history 中提取时间线上的 MCTS 值（每100步采样）
            param_steps = []
            mcts_vals = []
            temp_vals = []
            for h in loss_hist[::20]:  # 每20步采一个点
                param_steps.append(h.get('step', 0))
                # 从 status 获取最新 auto_params（简化：所有点用当前值）
                mcts_vals.append(ap.get('mcts_simulations', 200))
                temp_vals.append(ap.get('temperature', 1.0))

            if param_steps:
                self.line_mcts.set_data(param_steps, mcts_vals)
                self.line_temp.set_data(param_steps, temp_vals)
                self.ax_mcts.relim()
                self.ax_mcts.autoscale_view()

        # ── 学习率 ──
        if len(loss_hist) >= 2:
            lr_steps = [h.get('step', 0) for h in loss_hist]
            lr_vals = [h.get('learning_rate', 0) for h in loss_hist]
            if any(lr_vals):
                self.line_lr.set_data(lr_steps, lr_vals)
                self.ax_lr.relim()
                self.ax_lr.autoscale_view()

        # 刷新画布
        self.chart_canvas.draw_idle()

    def _append_log(self, msg: str):
        """添加日志"""
        from datetime import datetime
        ts = datetime.now().strftime("%H:%M:%S")
        self.log_text.append(f"[{ts}] {msg}")

    def _on_mode_toggled(self, source: str, checked: bool):
        """
        训练模式勾选切换 → 自动同步对应采样权重。

        三个模式（self/trad/human）共用此方法：
          - 取消勾选 → 该模式权重置 0，释放的比例按原比例分配给其余两个
          - 勾选     → 从其余两个按比例匀出 10 给该模式
        """
        self._weight_adjusting = True

        sliders = {'self': self.slider_self, 'trad': self.slider_trad, 'human': self.slider_human}
        labels  = {'self': self.lbl_self,  'trad': self.lbl_trad,  'human': self.lbl_human}
        others  = [s for s in sliders if s != source]

        if not checked:
            # ── 取消勾选：该模式置 0，余量按比例分给其余两个 ──
            old_val = sliders[source].value()
            sliders[source].setValue(0)
            labels[source].setText("0%")

            if old_val > 0:
                total_other = sum(sliders[s].value() for s in others)
                if total_other > 0:
                    for i, s in enumerate(others):
                        if i < len(others) - 1:
                            new_val = round(sliders[s].value() + old_val * (sliders[s].value() / total_other))
                        else:
                            new_val = 100 - sliders[others[0]].value()  # 最后一个用减法确保总和=100
                        new_val = max(0, min(100, new_val))
                        sliders[s].setValue(new_val)
                        labels[s].setText(f"{new_val}%")
        else:
            # ── 勾选：该模式从 0→10，其余两个等比缩减 ──
            if sliders[source].value() == 0:
                current_others = sum(sliders[s].value() for s in others)
                if current_others > 0:
                    for i, s in enumerate(others):
                        if i < len(others) - 1:
                            new_val = max(0, round(sliders[s].value() * 90 / current_others))
                        else:
                            new_val = 90 - sliders[others[0]].value()
                        new_val = max(0, min(100, new_val))
                        sliders[s].setValue(new_val)
                        labels[s].setText(f"{new_val}%")
                sliders[source].setValue(10)
                labels[source].setText("10%")

        self._weight_adjusting = False

    def _on_weight_self_changed(self, v):
        """Self采样权重变化 → 联动调整Trad和Human，保持总和=100。"""
        self.lbl_self.setText(f"{v}%")
        if self._weight_adjusting:
            return
        self._balance_weights('self', v)

    def _on_weight_trad_changed(self, v):
        """Trad采样权重变化 → 联动调整Self和Human。"""
        self.lbl_trad.setText(f"{v}%")
        if self._weight_adjusting:
            return
        self._balance_weights('trad', v)

    def _on_weight_human_changed(self, v):
        """Human采样权重变化 → 联动调整Self和Trad。"""
        self.lbl_human.setText(f"{v}%")
        if self._weight_adjusting:
            return
        self._balance_weights('human', v)

    def _balance_weights(self, changed_source, new_val):
        """调整另两个采样权重滑条，使三者总和恒为100。"""
        self._weight_adjusting = True
        remaining = 100 - new_val

        sources = ['self', 'trad', 'human']
        sliders = {'self': self.slider_self, 'trad': self.slider_trad, 'human': self.slider_human}
        labels = {'self': self.lbl_self, 'trad': self.lbl_trad, 'human': self.lbl_human}

        others = [s for s in sources if s != changed_source]
        old_sum = sum(sliders[s].value() for s in others)

        if old_sum == 0:
            # 另两项都为0，均匀分配
            for i, s in enumerate(others):
                val = remaining // 2 if i == 0 else remaining - remaining // 2
                val = max(0, min(100, val))
                sliders[s].setValue(val)
                labels[s].setText(f"{val}%")
        else:
            # 按原有比例重新分配，最后一项用减法确保总和=100
            prev_values = []
            for i, s in enumerate(others):
                if i < len(others) - 1:
                    val = max(0, min(100, round(remaining * sliders[s].value() / old_sum)))
                else:
                    val = max(0, min(100, remaining - sum(prev_values)))
                prev_values.append(val)
                sliders[s].setValue(val)
                labels[s].setText(f"{val}%")

        self._weight_adjusting = False

    def _apply_hardware_config(self):
        """将硬件推荐配置应用到训练参数滑块"""
        try:
            train = self.auto_config.get('training', {})

            # 学习率
            lr = train.get('learning_rate', 0.02)
            self.slider_lr.setValue(int(lr * 10000))

            # Batch Size
            self.slider_batch.setValue(train.get('batch_size', 1024))

            # MCTS 模拟次数
            self.slider_mcts.setValue(train.get('num_mcts_simulations', 400))

            # 传统AI初始深度
            trad = train.get('traditional', {})
            self.slider_trad_init_depth.setValue(trad.get('initial_depth', 4))

            # 每轮对弈局数（根据硬件CPU核心数推荐）
            cpu_count = self.auto_config.get('hardware', {}).get('cpu_threads', 8)
            recommended_games = max(2, min(10, cpu_count // 4))
            self.slider_parallel.setValue(recommended_games)

            self._append_log("✅ 已应用硬件推荐配置参数")
            self._append_log(f"  Batch: {train.get('batch_size', 1024)}, "
                             f"MCTS: {train.get('num_mcts_simulations', 400)}, "
                             f"LR: {train.get('learning_rate', 0.02):.5f}")
        except Exception as e:
            self._append_log(f"❌ 应用配置失败: {e}")

    def _run_gpu_benchmark(self):
        """运行 GPU 基准测试"""
        self._append_log("开始 GPU 基准测试（约10秒）...")
        try:
            from hardware_auto_config import benchmark_gpu_memory
            if torch.cuda.is_available():
                # 获取当前网络配置
                net = self.auto_config.get('network', {})
                train = self.auto_config.get('training', {})
                result = benchmark_gpu_memory(
                    num_filters=net.get('num_filters', 128),
                    num_res_blocks=net.get('num_res_blocks', 7),
                    batch_size=train.get('batch_size', 1024),
                )
                if result.get('safe', False):
                    self._append_log(f"  ✅ 基准测试完成:")
                    self._append_log(f"    前向推理: {result['forward_time_ms']:.2f} ms")
                    self._append_log(f"    显存使用: {result['memory_used_mb']:.1f} MB")
                    self._append_log(f"    推荐 Batch: {result['recommended_batch']}")
                    # 自动更新 batch 滑块
                    self.slider_batch.setValue(result['recommended_batch'])
                else:
                    self._append_log(f"  ⚠️ 测试不通过: {result.get('error', '未知')}")
            else:
                self._append_log("  没有检测到 GPU，跳过基准测试")
        except Exception as e:
            self._append_log(f"  ❌ 基准测试失败: {e}")

    def closeEvent(self, event):
        """关闭面板时恢复主窗口GUI"""
        main_window = self.parent()
        if main_window and not main_window.isVisible():
            main_window.show()
        super().closeEvent(event)

    def _start_status_update(self):
        """定期更新状态"""
        self.status_timer = QTimer(self)
        self.status_timer.timeout.connect(self._poll_status)
        self.status_timer.start(2000)

    def _poll_status(self):
        """轮询训练状态"""
        if self.training_thread and self.training_thread.trainer:
            status = self.training_thread.trainer.get_status()
            self._update_status(status)


# ============================================================================
# 主窗口
# ============================================================================

class MainWindow(QMainWindow):
    """五子棋AI主窗口"""

    def __init__(self):
        super().__init__()

        # ── 硬件自适应初始化 ──
        gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
        hw_tier = _HW_SUMMARY.get('gpu_tier_name', '未知')
        self.setWindowTitle(f"五子棋神经网络AI — {gpu_name} [{hw_tier}]")
        self.setMinimumSize(900, 650)

        # 显示硬件信息
        hw_gpu = _HW_SUMMARY.get('gpu', {})
        hw_cpu = _HW_SUMMARY.get('cpu', {})
        logger.info(f"硬件检测: GPU={hw_gpu.get('name', 'N/A')} "
                    f"显存={hw_gpu.get('total_memory_mb', 0)}MB "
                    f"CPU={hw_cpu.get('cores_physical', '?')}核")

        # ── 从硬件配置读取推荐参数 ──
        train_cfg = _HW_CONFIG.get('training', {})
        self._recommended_mcts = train_cfg.get('num_mcts_simulations', 400)
        trad_cfg = train_cfg.get('traditional', {})
        self._recommended_trad_depth = trad_cfg.get('initial_depth', 4)

        # 游戏状态
        self.game_mode = 'human_black'
        self.training_mode = False
        self.traditional_is_black = True
        self._ref_game_board = GomokuGame()

        # AI引擎
        self.model: Optional[GomokuNet] = None
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.ai_thread: Optional[AIThread] = None
        self.ai_thinking = False
        self.num_mcts_simulations = self._recommended_mcts

        # 传统AI（使用硬件推荐的搜索深度）
        self.traditional_ai = TraditionalAI(search_depth=self._recommended_trad_depth)

        # 并行对弈局数
        self.num_parallel_games = min(5, max(1, hw_cpu.get('cores_physical', 4) // 2))

        # 对弈计时器
        self.ai_timer = QTimer(self)
        self.ai_timer.timeout.connect(self._trigger_ai_move)

        self._init_ui()
        self._load_model()
        self._apply_dark_theme()

        # 状态栏显示硬件信息
        self.status_bar.showMessage(
            f"硬件: {hw_gpu.get('name', 'CPU')} "
            f"({hw_gpu.get('total_memory_mb', 0)}MB) | "
            f"推荐MCTS: {self._recommended_mcts}次 | "
            f"推荐深度: {self._recommended_trad_depth}层"
        )

    def _init_ui(self):
        """初始化UI"""
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)

        # 左侧：棋盘
        self.board_widget = BoardWidget()
        self.board_widget.move_made.connect(self._on_human_move)
        self.board_widget.game_state_changed.connect(self._on_game_state_changed)
        main_layout.addWidget(self.board_widget, 3)

        # 右侧面板
        right_panel = QWidget()
        right_panel.setMaximumWidth(280)
        right_layout = QVBoxLayout(right_panel)

        # 对局信息
        info_group = QGroupBox("对局信息")
        info_layout = QVBoxLayout()
        self.lbl_current_player = QLabel("当前: ● 黑棋")
        self.lbl_current_player.setFont(QFont('Arial', 12, QFont.Weight.Bold))
        info_layout.addWidget(self.lbl_current_player)

        self.lbl_move_count = QLabel("步数: 0")
        info_layout.addWidget(self.lbl_move_count)

        self.lbl_status = QLabel("状态: 等待落子")
        info_layout.addWidget(self.lbl_status)

        gpu_tag = "GPU" if self.device.type == 'cuda' else "CPU"
        self.lbl_ai_info = QLabel(f"推理设备: [{gpu_tag}]")
        info_layout.addWidget(self.lbl_ai_info)
        info_group.setLayout(info_layout)
        right_layout.addWidget(info_group)

        # 模式选择
        mode_group = QGroupBox("对局模式")
        mode_layout = QVBoxLayout()
        self.cmb_mode = QComboBox()
        self.cmb_mode.addItems(["玩家(黑) vs AI(白)", "玩家(白) vs AI(黑)",
                                "AI vs AI", "传统AI(黑) vs NN(白)"])
        self.cmb_mode.currentIndexChanged.connect(self._on_mode_changed)
        mode_layout.addWidget(self.cmb_mode)

        self.cb_training = QCheckBox("训练数据收集模式")
        self.cb_training.toggled.connect(self._on_training_mode_toggled)
        mode_layout.addWidget(self.cb_training)
        mode_group.setLayout(mode_layout)
        right_layout.addWidget(mode_group)

        # AI设置滑动条（仅保留对弈相关的并行局数）
        ai_setting_group = QGroupBox("AI设置")
        ai_setting_layout = QVBoxLayout()

        # 并行对弈局数滑动条（AI vs AI 演示 / 训练数据收集）
        parallel_layout = QHBoxLayout()
        parallel_layout.addWidget(QLabel("并行局数:"))
        self.slider_parallel = QSlider(Qt.Orientation.Horizontal)
        self.slider_parallel.setRange(1, 10)
        self.slider_parallel.setValue(self.num_parallel_games)
        self.slider_parallel.setSingleStep(1)
        self.slider_parallel.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.slider_parallel.setTickInterval(1)
        self.slider_parallel.valueChanged.connect(self._on_parallel_changed)
        parallel_layout.addWidget(self.slider_parallel)
        self.lbl_parallel_val = QLabel(str(self.num_parallel_games))
        self.lbl_parallel_val.setMinimumWidth(35)
        self.lbl_parallel_val.setAlignment(Qt.AlignmentFlag.AlignRight)
        parallel_layout.addWidget(self.lbl_parallel_val)
        ai_setting_layout.addLayout(parallel_layout)

        # GPU加速提示
        gpu_status = "GPU (CUDA)" if self.device.type == 'cuda' else "CPU"
        gpu_mem = _HW_SUMMARY.get('gpu', {}).get('total_memory_mb', 0)
        gpu_name_short = _HW_SUMMARY.get('gpu', {}).get('name', 'CPU')
        self.lbl_gpu_status = QLabel(
            f"计算: {gpu_status} | {gpu_name_short} ({gpu_mem}MB)")
        self.lbl_gpu_status.setStyleSheet("color: #888; font-size: 10px;")
        ai_setting_layout.addWidget(self.lbl_gpu_status)

        ai_setting_group.setLayout(ai_setting_layout)
        right_layout.addWidget(ai_setting_group)

        # 控制按钮
        ctrl_group = QGroupBox("操作")
        ctrl_layout = QVBoxLayout()

        self.btn_new_game = QPushButton("新局")
        self.btn_new_game.clicked.connect(self._new_game)
        ctrl_layout.addWidget(self.btn_new_game)

        self.btn_undo = QPushButton("悔棋")
        self.btn_undo.clicked.connect(self._undo_move)
        ctrl_layout.addWidget(self.btn_undo)

        self.btn_load_model = QPushButton("加载模型...")
        self.btn_load_model.clicked.connect(self._load_model_dialog)
        ctrl_layout.addWidget(self.btn_load_model)

        self.btn_training_panel = QPushButton("训练面板")
        self.btn_training_panel.clicked.connect(self._open_training_panel)
        ctrl_layout.addWidget(self.btn_training_panel)

        ctrl_group.setLayout(ctrl_layout)
        right_layout.addWidget(ctrl_group)

        # 棋谱列表
        move_group = QGroupBox("棋谱")
        move_layout = QVBoxLayout()
        self.move_list = QListWidget()
        self.move_list.setMaximumHeight(150)
        move_layout.addWidget(self.move_list)
        move_group.setLayout(move_layout)
        right_layout.addWidget(move_group)

        right_layout.addStretch()
        main_layout.addWidget(right_panel, 1)

        # 状态栏
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("就绪")

    def _apply_dark_theme(self):
        """应用深色主题"""
        dark_palette = QPalette()
        dark_palette.setColor(QPalette.ColorRole.Window, QColor(53, 53, 53))
        dark_palette.setColor(QPalette.ColorRole.WindowText, QColor(220, 220, 220))
        dark_palette.setColor(QPalette.ColorRole.Base, QColor(35, 35, 35))
        dark_palette.setColor(QPalette.ColorRole.AlternateBase, QColor(53, 53, 53))
        dark_palette.setColor(QPalette.ColorRole.ToolTipBase, QColor(25, 25, 25))
        dark_palette.setColor(QPalette.ColorRole.ToolTipText, QColor(220, 220, 220))
        dark_palette.setColor(QPalette.ColorRole.Text, QColor(220, 220, 220))
        dark_palette.setColor(QPalette.ColorRole.Button, QColor(53, 53, 53))
        dark_palette.setColor(QPalette.ColorRole.ButtonText, QColor(220, 220, 220))
        dark_palette.setColor(QPalette.ColorRole.BrightText, QColor(255, 0, 0))
        dark_palette.setColor(QPalette.ColorRole.Link, QColor(42, 130, 218))
        dark_palette.setColor(QPalette.ColorRole.Highlight, QColor(42, 130, 218))
        dark_palette.setColor(QPalette.ColorRole.HighlightedText, QColor(35, 35, 35))
        self.setPalette(dark_palette)

    def _load_model(self):
        """加载模型（v2.0: 根据硬件配置创建自适应网络）"""
        model_path = os.path.join(PROJECT_DIR, 'models', 'model.pt')
        if os.path.exists(model_path):
            try:
                self.model, info = GomokuNet.load_checkpoint(model_path, self.device)
                model_params = self.model.count_parameters()
                self.status_bar.showMessage(
                    f"模型已加载: step={info.get('step', 0)}, "
                    f"参数={model_params:,}, "
                    f"elo={info.get('elo', '-')}"
                )
            except Exception as e:
                self.status_bar.showMessage(f"模型加载失败: {e}，创建新模型")
                # 按硬件配置创建网络
                net_cfg = _HW_CONFIG.get('network', {})
                self.model = GomokuNet(
                    num_filters=net_cfg.get('num_filters', 128),
                    num_res_blocks=net_cfg.get('num_res_blocks', 7),
                    use_se=True,
                ).to(self.device)
        else:
            # 未找到模型，按硬件配置创建新网络
            net_cfg = _HW_CONFIG.get('network', {})
            self.model = GomokuNet(
                num_filters=net_cfg.get('num_filters', 128),
                num_res_blocks=net_cfg.get('num_res_blocks', 7),
                use_se=True,
            ).to(self.device)
            model_params = self.model.count_parameters()
            self.status_bar.showMessage(
                f"新建网络: {net_cfg.get('num_filters')}通道, "
                f"{net_cfg.get('num_res_blocks')}残差块, "
                f"参数={model_params:,}"
            )

    def _load_model_dialog(self):
        """加载模型对话框"""
        default_dir = os.path.join(PROJECT_DIR, 'models')
        path, _ = QFileDialog.getOpenFileName(
            self, "加载模型", default_dir, "PyTorch模型 (*.pt)"
        )
        if path:
            try:
                self.model, info = GomokuNet.load_checkpoint(path, self.device)
                self.status_bar.showMessage(
                    f"模型已加载: {path} (step={info.get('step', 0)})"
                )
            except Exception as e:
                QMessageBox.critical(self, "错误", f"模型加载失败: {e}")

    def _new_game(self):
        """开始新局"""
        self.board_widget.reset_board()
        self.move_list.clear()
        self._update_info()
        self._start_ai_if_needed()

    def _undo_move(self):
        """悔棋"""
        if self.game_mode in ('ai_vs_ai', 'trad_vs_nn'):
            return  # AI对弈不允许悔棋

        # 悔棋两步（玩家+AI）
        if self.game_mode in ('human_black', 'human_white'):
            if len(self.board_widget.game.move_history) >= 2:
                self.board_widget.undo_move()
                self.board_widget.undo_move()
                self._update_move_list()
                self._update_info()
                self._start_ai_if_needed()

    def _on_human_move(self, row: int, col: int):
        """人类落子"""
        if self.ai_thinking:
            return

        game = self.board_widget.game
        if game.game_over:
            return

        # 纯AI对弈模式不允许人类落子
        if self.game_mode in ('ai_vs_ai', 'trad_vs_nn'):
            return

        # 检查是否轮到玩家
        if self.game_mode == 'human_black' and game.current_player != BLACK:
            return
        if self.game_mode == 'human_white' and game.current_player != WHITE:
            return

        if self.board_widget.make_move(row, col):
            self._update_move_list()
            self._update_info()

            if not game.game_over:
                self._start_ai_if_needed()

    def _on_game_state_changed(self):
        """游戏状态变化"""
        self._update_info()

    def _update_info(self):
        """更新对局信息"""
        game = self.board_widget.game
        current = game.current_player

        if game.game_over:
            if game.winner == BLACK:
                self.lbl_current_player.setText("● 黑棋获胜！")
            elif game.winner == WHITE:
                self.lbl_current_player.setText("○ 白棋获胜！")
            else:
                self.lbl_current_player.setText("平局！")
            self.lbl_status.setText("对局结束")
            self.ai_thinking = False
        else:
            player_str = "● 黑棋" if current == BLACK else "○ 白棋"
            self.lbl_current_player.setText(f"当前: {player_str}")
            if self.ai_thinking:
                self.lbl_status.setText("AI思考中...")
            else:
                self.lbl_status.setText("等待落子")

        self.lbl_move_count.setText(f"步数: {len(game.move_history)}")

    def _update_move_list(self):
        """更新棋谱列表"""
        self.move_list.clear()
        game = self.board_widget.game
        for i, (r, c) in enumerate(game.move_history):
            player = BLACK if i % 2 == 0 else WHITE
            symbol = "●" if player == BLACK else "○"
            coord = f"{chr(ord('A') + c)}{r + 1}"
            self.move_list.addItem(f"{i + 1:2d}. {symbol} {coord}")

        # 滚动到最后
        if self.move_list.count() > 0:
            self.move_list.scrollToBottom()

    def _on_mode_changed(self, index: int):
        """模式切换"""
        modes = ['human_black', 'human_white', 'ai_vs_ai', 'trad_vs_nn']
        self.game_mode = modes[index]
        self._new_game()

    def _on_parallel_changed(self, value: int):
        """并行对弈局数滑动条变化"""
        self.num_parallel_games = value
        self.lbl_parallel_val.setText(str(value))

    def _on_training_mode_toggled(self, checked: bool):
        """训练模式切换"""
        self.training_mode = checked
        if checked:
            self.status_bar.showMessage("训练数据收集模式已开启")
        else:
            self.status_bar.showMessage("训练数据收集模式已关闭")

    def _start_ai_if_needed(self):
        """如果需要，启动AI走子"""
        game = self.board_widget.game
        if game.game_over:
            return

        # 训练数据收集模式 + AI vs AI：使用多局并发对弈
        if self.training_mode and self.game_mode == 'ai_vs_ai':
            if not self.ai_thinking:
                self._start_training_multi_game()
            return

        need_ai = False
        if self.game_mode == 'human_black' and game.current_player == WHITE:
            need_ai = True
        elif self.game_mode == 'human_white' and game.current_player == BLACK:
            need_ai = True
        elif self.game_mode == 'ai_vs_ai':
            need_ai = True
        elif self.game_mode == 'trad_vs_nn':
            need_ai = True

        if need_ai and not self.ai_thinking:
            self._request_ai_move()

    def _trigger_ai_move(self):
        """计时器触发AI走子 / 自动新局"""
        self.ai_timer.stop()
        game = self.board_widget.game

        # 对局已结束 → 自动开始新局（演示模式）
        if game.game_over:
            if self.game_mode in ('ai_vs_ai', 'trad_vs_nn'):
                self._new_game()
            return

        self._start_ai_if_needed()

    def _request_ai_move(self):
        """请求AI走子（单局模式，非训练数据收集）"""
        game = self.board_widget.game
        if game.game_over:
            return

        # trad_vs_nn 模式：传统AI走子（同步，无需模型）
        if self.game_mode == 'trad_vs_nn' and game.current_player == BLACK:
            self._do_traditional_move()
            return

        if self.model is None:
            return

        self.ai_thinking = True
        self.lbl_status.setText("AI思考中...")
        self.status_bar.showMessage("AI思考中...")

        self.ai_thread = AIThread(
            model=self.model,
            game=self.board_widget.game,
            num_simulations=self.num_mcts_simulations,
            device=self.device
        )
        self.ai_thread.move_ready.connect(self._on_ai_move)
        self.ai_thread.thinking_finished.connect(self._on_ai_finished)
        self.ai_thread.start()

    def _do_traditional_move(self):
        """传统AI同步走子（用于 trad_vs_nn 模式中的黑棋）"""
        game = self.board_widget.game
        move = self.traditional_ai.get_best_move(game)
        if move is None:
            return
        self.board_widget.make_move(move[0], move[1])
        self._update_move_list()
        self._update_info()
        stats = self.traditional_ai.get_stats()
        gpu_tag = "[GPU]" if stats.get('gpu', False) else "[CPU]"
        self.status_bar.showMessage(
            f"传统AI{stats['depth']}层 {gpu_tag} → {chr(ord('A')+move[1])}{move[0]+1} "
            f"耗时 {stats['time']:.3f}s")

        # 传统AI走完后
        if not game.game_over:
            # 对局未结束：触发NN走子
            if self.game_mode == 'trad_vs_nn':
                self.ai_timer.start(300)
        else:
            # 对局已结束：演示模式自动重开
            if self.game_mode in ('trad_vs_nn',):
                self.status_bar.showMessage("对局结束，2秒后自动开始新局...")
                self.ai_timer.start(2000)

    def _on_ai_move(self, row: int, col: int):
        """AI落子"""
        self.board_widget.make_move(row, col)
        self._update_move_list()

    def _on_ai_finished(self):
        """AI思考完成"""
        self.ai_thinking = False
        self.status_bar.showMessage("就绪")
        self._update_info()

        game = self.board_widget.game
        demo_modes = ('ai_vs_ai', 'trad_vs_nn')

        if not game.game_over:
            # 对局未结束：延迟后继续走下一步
            if self.game_mode in demo_modes:
                self.ai_timer.start(500)
        else:
            # 对局已结束：自动开始新局（演示模式）
            if self.game_mode in demo_modes:
                self.status_bar.showMessage("对局结束，2秒后自动开始新局...")
                self.ai_timer.start(2000)
            elif self.game_mode in ('human_black', 'human_white'):
                self.lbl_status.setText("对局结束，点击新局开始")

    # ------------------------------------------------------------------
    # 训练数据收集：5局并发 AI vs AI
    # ------------------------------------------------------------------

    def _start_training_multi_game(self):
        """启动多局并发AI vs AI对弈（训练数据收集模式）"""
        if self.model is None:
            return

        ngames = self.num_parallel_games
        self.ai_thinking = True
        self.lbl_status.setText(f"AI训练对弈中({ngames}局并发)...")
        self.status_bar.showMessage(f"训练数据收集: {ngames}局并发对弈进行中...")

        # 重置棋盘用于参考局渲染
        self._ref_game_board = GomokuGame()

        self.multi_game_thread = MultiGameAIThread(
            model=self.model,
            num_simulations=self.num_mcts_simulations,
            device=self.device,
            num_games=ngames,
            replay_buffer=None  # 在主线程中手动添加到buffer
        )
        self.multi_game_thread.reference_move.connect(self._on_training_ref_move)
        self.multi_game_thread.all_games_completed.connect(self._on_training_games_done)
        self.multi_game_thread.game_progress.connect(self._on_training_game_progress)
        self.multi_game_thread.thinking_status.connect(self.status_bar.showMessage)
        self.multi_game_thread.start()

    def _on_training_ref_move(self, row: int, col: int):
        """参考局落子 → 更新UI棋盘渲染"""
        # 同步参考局的棋盘状态
        self._ref_game_board.make_move(row, col)

        # 刷新 BoardWidget 显示
        self.board_widget.game = self._ref_game_board.clone()
        self.board_widget.update()
        self._update_move_list()
        self._update_info()

    def _on_training_game_progress(self, completed: int, total: int):
        """更新对弈进度"""
        self.status_bar.showMessage(f"训练数据收集: {completed}/{total} 局完成...")

    def _on_training_games_done(self, all_data: list):
        """多局并发全部完成 → 保存数据并自动开始下一轮"""
        self.ai_thinking = False
        self.status_bar.showMessage(f"训练数据收集完成: 共{len(all_data)}条经验")

        # 保存训练数据到本地
        save_dir = os.path.join(PROJECT_DIR, 'training_data')
        os.makedirs(save_dir, exist_ok=True)
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        save_path = os.path.join(save_dir, f'training_data_{timestamp}.npz')

        try:
            states = np.array([d[0] for d in all_data], dtype=np.float32)
            policies = np.array([d[1] for d in all_data], dtype=np.float32)
            values = np.array([d[2] for d in all_data], dtype=np.float32)
            np.savez_compressed(save_path, states=states, policies=policies, values=values)
            self.status_bar.showMessage(
                f"训练数据已保存: {save_path} ({len(all_data)}条) → 2秒后自动开始下一轮..."
            )
        except Exception as e:
            self.status_bar.showMessage(f"保存训练数据失败: {e}")

        # 重置参考棋盘
        self._ref_game_board = GomokuGame()
        self.board_widget.reset_board()
        self.move_list.clear()
        self._update_info()
        self.lbl_status.setText(f"训练数据收集: 本轮{len(all_data)}条，2秒后继续...")

        # 自动开始下一轮收集（训练模式保持开启）
        if self.training_mode:
            self.ai_timer.start(2000)

    def _open_training_panel(self):
        """打开训练面板（非模态，允许训练时独立存在）"""
        panel = TrainingPanel(self)
        panel.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        panel.show()


# ============================================================================
# 主入口
# ============================================================================

def main():
    """启动GUI"""
    import argparse
    parser = argparse.ArgumentParser(description='五子棋神经网络AI')
    parser.add_argument('--mode', type=str, default='gui',
                        choices=['gui', 'train', 'service'],
                        help='运行模式: gui (GUI界面), train (命令行训练), service (后台服务)')
    parser.add_argument('--config', type=str, default='train_config.json',
                        help='训练配置文件路径')
    args = parser.parse_args()

    if args.mode == 'gui':
        app = QApplication(sys.argv)
        app.setApplicationName("五子棋神经网络AI")
        window = MainWindow()
        window.show()
        sys.exit(app.exec())
    elif args.mode == 'train':
        # 命令行训练模式
        from trainer import Trainer
        import json
        config_path = args.config if os.path.isabs(args.config) else os.path.join(PROJECT_DIR, args.config)
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        trainer = Trainer(config)
        trainer.train_loop()
    elif args.mode == 'service':
        print("请使用 train_service.py 运行后台服务")
        sys.exit(1)


if __name__ == '__main__':
    main()
