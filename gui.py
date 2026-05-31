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

from game import GomokuGame, BOARD_SIZE, BLACK, WHITE, EMPTY
from network import GomokuNet
from mcts import MCTS, get_best_move_from_probs
from traditional_ai import TraditionalAI
from trainer import Trainer, ReplayBuffer


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
            action_probs, _ = mcts.search(self.game, temperature=0.1)
            move = get_best_move_from_probs(action_probs, self.game, deterministic=True)
            if move and move != (-1, -1):
                self.move_ready.emit(move[0], move[1])
        except Exception as e:
            print(f"AI推理错误: {e}")
        finally:
            self.thinking_finished.emit()


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
        """运行训练"""
        try:
            self.trainer = Trainer(self.config)
            self.trainer.train_loop(games_per_iteration=5)

            # 定期更新状态
            while self.trainer.running:
                self.status_update.emit(self.trainer.get_status())
                self.msleep(2000)
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
    """训练监控面板"""

    def __init__(self, parent=None, config: dict = None):
        super().__init__(parent)
        self.setWindowTitle("训练监控面板")
        self.setMinimumSize(700, 600)
        self.config = config or {}
        self.training_thread: Optional[TrainingThread] = None

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

        # MCTS模拟次数 - 滑动条
        param_layout.addWidget(QLabel("MCTS模拟:"), 1, 0)
        self.slider_mcts = QSlider(Qt.Orientation.Horizontal)
        self.slider_mcts.setRange(50, 800)
        self.slider_mcts.setValue(400)
        self.slider_mcts.setSingleStep(50)
        self.slider_mcts.setTickInterval(100)
        self.slider_mcts.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.lbl_mcts_val = QLabel("400")
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

        # 连接滑块信号
        self.slider_self.valueChanged.connect(
            lambda v: self.lbl_self.setText(f"{v}%"))
        self.slider_trad.valueChanged.connect(
            lambda v: self.lbl_trad.setText(f"{v}%"))
        self.slider_human.valueChanged.connect(
            lambda v: self.lbl_human.setText(f"{v}%"))

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

        # --- 监控 ---
        monitor_tab = QWidget()
        monitor_layout = QVBoxLayout(monitor_tab)

        # 进度信息
        self.lbl_step = QLabel("训练步数: 0")
        self.lbl_step.setFont(QFont('Arial', 12, QFont.Weight.Bold))
        monitor_layout.addWidget(self.lbl_step)

        self.lbl_buffer = QLabel("经验池: 0")
        monitor_layout.addWidget(self.lbl_buffer)

        self.lbl_loss = QLabel("损失: -")
        monitor_layout.addWidget(self.lbl_loss)

        self.lbl_lr = QLabel("学习率: -")
        monitor_layout.addWidget(self.lbl_lr)

        self.lbl_trad_depth = QLabel("传统AI深度: -")
        monitor_layout.addWidget(self.lbl_trad_depth)

        self.lbl_trad_wr = QLabel("NN vs 传统AI胜率: -")
        monitor_layout.addWidget(self.lbl_trad_wr)

        self.lbl_elo = QLabel("ELO: -")
        monitor_layout.addWidget(self.lbl_elo)

        # 进度条
        self.progress_bar = QProgressBar()
        monitor_layout.addWidget(self.progress_bar)

        # 日志
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(200)
        monitor_layout.addWidget(QLabel("训练日志:"))
        monitor_layout.addWidget(self.log_text)

        tabs.addTab(monitor_tab, "监控")

        layout.addWidget(tabs)

    def _start_training(self):
        """开始训练"""
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

        config = {
            'training': {
                'modes': modes,
                'learning_rate': self.slider_lr.value() / 10000.0,
                'batch_size': self.slider_batch.value(),
                'num_mcts_simulations': self.slider_mcts.value(),
                'total_steps': self.slider_steps.value(),
                'use_cuda': torch.cuda.is_available(),
                'model_dir': './models',
                'log_dir': './logs',
                'save_interval_minutes': self.slider_save.value(),
                'auto_eval_games': self.slider_eval_games.value(),
                'replay_buffer': {
                    'capacity': 2000000,
                    'sampling_weights': {
                        'self': self.slider_self.value() / 100.0,
                        'trad': self.slider_trad.value() / 100.0,
                        'human': self.slider_human.value() / 100.0,
                    }
                },
                'traditional': {
                    'initial_depth': 4,
                    'depth_range': [1, 8],
                    'games_per_adjust': 10,
                    'target_win_rate': 0.20,
                    'win_rate_window': 100,
                },
                'human': {
                    'enabled': self.cb_human.isChecked(),
                    'temperature_for_human_moves': 0.1,
                }
            }
        }

        self.training_thread = TrainingThread(config)
        self.training_thread.status_update.connect(self._update_status)
        self.training_thread.log_message.connect(self._append_log)
        self.training_thread.start()

        self.btn_start.setEnabled(False)
        self.btn_pause.setEnabled(True)
        self.btn_stop.setEnabled(True)

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

    def _save_checkpoint(self):
        """保存检查点"""
        if self.training_thread and self.training_thread.trainer:
            self.training_thread.trainer._save_checkpoint(is_best=False)
            self._append_log("检查点已保存")

    def _update_status(self, status: dict):
        """更新训练状态显示"""
        self.lbl_step.setText(f"训练步数: {status.get('global_step', 0)}")
        self.lbl_buffer.setText(f"经验池: {status.get('buffer_size', 0)}")
        self.lbl_trad_depth.setText(f"传统AI深度: {status.get('trad_depth', '-')}")
        wr = status.get('trad_win_rate', 0)
        self.lbl_trad_wr.setText(f"NN vs 传统AI胜率: {wr:.1%}")
        self.lbl_elo.setText(f"ELO: current={status.get('elo_current', 0):.0f}, "
                             f"best={status.get('elo_best', 0):.0f}")

        if status.get('loss_history'):
            last = status['loss_history'][-1]
            self.lbl_loss.setText(
                f"损失: total={last.get('total_loss', 0):.4f}, "
                f"policy={last.get('policy_loss', 0):.4f}, "
                f"value={last.get('value_loss', 0):.4f}"
            )
            self.lbl_lr.setText(f"学习率: {last.get('learning_rate', 0):.6f}")

        total_steps = self.slider_steps.value()
        if total_steps > 0:
            self.progress_bar.setValue(
                int(status.get('global_step', 0) / total_steps * 100)
            )

    def _append_log(self, msg: str):
        """添加日志"""
        from datetime import datetime
        ts = datetime.now().strftime("%H:%M:%S")
        self.log_text.append(f"[{ts}] {msg}")

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
        gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
        self.setWindowTitle(f"五子棋神经网络AI — {gpu_name}")
        self.setMinimumSize(900, 650)

        # 游戏状态
        self.game_mode = 'human_black'  # 'human_black', 'human_white', 'ai_vs_ai', 'trad_vs_nn'
        self.training_mode = False  # 人机对弈训练模式
        self.traditional_is_black = True  # 传统AI vs NN 模式中，传统AI执黑先手

        # AI引擎
        self.model: Optional[GomokuNet] = None
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.ai_thread: Optional[AIThread] = None
        self.ai_thinking = False
        self.num_mcts_simulations = 400

        # 传统AI（用于纯传统AI模式）
        self.traditional_ai = TraditionalAI(search_depth=4)

        # 对弈计时器
        self.ai_timer = QTimer(self)
        self.ai_timer.timeout.connect(self._trigger_ai_move)

        self._init_ui()
        self._load_model()
        self._apply_dark_theme()

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
        self.lbl_ai_info = QLabel(f"AI: MCTS {self.num_mcts_simulations}次 [{gpu_tag}]")
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

        # AI设置滑动条
        ai_setting_group = QGroupBox("AI设置")
        ai_setting_layout = QVBoxLayout()

        # MCTS模拟次数滑动条
        mcts_layout = QHBoxLayout()
        mcts_layout.addWidget(QLabel("MCTS模拟:"))
        self.slider_mcts = QSlider(Qt.Orientation.Horizontal)
        self.slider_mcts.setRange(50, 800)
        self.slider_mcts.setValue(400)
        self.slider_mcts.setSingleStep(50)
        self.slider_mcts.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.slider_mcts.setTickInterval(100)
        self.slider_mcts.valueChanged.connect(self._on_mcts_slider_changed)
        mcts_layout.addWidget(self.slider_mcts)
        self.lbl_mcts_val = QLabel("400")
        self.lbl_mcts_val.setMinimumWidth(35)
        self.lbl_mcts_val.setAlignment(Qt.AlignmentFlag.AlignRight)
        mcts_layout.addWidget(self.lbl_mcts_val)
        ai_setting_layout.addLayout(mcts_layout)

        # 传统AI搜索深度滑动条
        trad_depth_layout = QHBoxLayout()
        trad_depth_layout.addWidget(QLabel("传统AI深度:"))
        self.slider_trad_depth = QSlider(Qt.Orientation.Horizontal)
        self.slider_trad_depth.setRange(1, 8)
        self.slider_trad_depth.setValue(4)
        self.slider_trad_depth.setSingleStep(1)
        self.slider_trad_depth.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.slider_trad_depth.setTickInterval(1)
        self.slider_trad_depth.valueChanged.connect(self._on_trad_depth_changed)
        trad_depth_layout.addWidget(self.slider_trad_depth)
        self.lbl_trad_depth_val = QLabel("4")
        self.lbl_trad_depth_val.setMinimumWidth(35)
        self.lbl_trad_depth_val.setAlignment(Qt.AlignmentFlag.AlignRight)
        trad_depth_layout.addWidget(self.lbl_trad_depth_val)
        ai_setting_layout.addLayout(trad_depth_layout)

        # GPU加速提示
        gpu_status = "GPU (CUDA)" if self.device.type == 'cuda' else "CPU"
        self.lbl_gpu_status = QLabel(f"计算设备: {gpu_status}  |  深度越大响应越慢")
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
        """加载模型"""
        model_path = os.path.join('models', 'best_model.pt')
        if os.path.exists(model_path):
            try:
                self.model, info = GomokuNet.load_checkpoint(model_path, self.device)
                self.status_bar.showMessage(
                    f"模型已加载: step={info.get('step', 0)}, "
                    f"elo={info.get('elo', '-')}"
                )
            except Exception as e:
                self.status_bar.showMessage(f"模型加载失败: {e}")
                self.model = GomokuNet().to(self.device)
        else:
            self.status_bar.showMessage("未找到预训练模型，使用随机权重")
            self.model = GomokuNet().to(self.device)

    def _load_model_dialog(self):
        """加载模型对话框"""
        path, _ = QFileDialog.getOpenFileName(
            self, "加载模型", "./models", "PyTorch模型 (*.pt)"
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

    def _on_mcts_slider_changed(self, value: int):
        """MCTS模拟次数滑动条变化"""
        self.num_mcts_simulations = value
        self.lbl_mcts_val.setText(str(value))
        gpu_tag = "GPU" if self.device.type == 'cuda' else "CPU"
        self.lbl_ai_info.setText(f"AI: MCTS {value}次 [{gpu_tag}]")

    def _on_trad_depth_changed(self, value: int):
        """传统AI搜索深度滑动条变化"""
        self.traditional_ai.set_depth(value)
        self.lbl_trad_depth_val.setText(str(value))

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
        """计时器触发AI走子"""
        self.ai_timer.stop()
        self._start_ai_if_needed()

    def _request_ai_move(self):
        """请求AI走子"""
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

        # 传统AI走完后，触发NN走子
        if not game.game_over and self.game_mode == 'trad_vs_nn':
            self.ai_timer.start(300)

    def _on_ai_move(self, row: int, col: int):
        """AI落子"""
        self.board_widget.make_move(row, col)
        self._update_move_list()

    def _on_ai_finished(self):
        """AI思考完成"""
        self.ai_thinking = False
        self.status_bar.showMessage("就绪")
        self._update_info()

        # AI vs AI 或 传统AI vs NN：延迟后继续
        if not self.board_widget.game.game_over:
            if self.game_mode in ('ai_vs_ai', 'trad_vs_nn'):
                self.ai_timer.start(500)  # 500ms延迟

    def _open_training_panel(self):
        """打开训练面板"""
        panel = TrainingPanel(self)
        panel.exec()


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
        with open(args.config, 'r', encoding='utf-8') as f:
            config = json.load(f)
        trainer = Trainer(config)
        trainer.train_loop()
    elif args.mode == 'service':
        print("请使用 train_service.py 运行后台服务")
        sys.exit(1)


if __name__ == '__main__':
    main()
