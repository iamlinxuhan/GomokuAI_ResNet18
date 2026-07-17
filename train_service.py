"""
静默训练服务
无GUI，后台执行训练循环，支持开机自启
通过 config.yaml 或 train_config.json 配置
支持优雅停机（捕获停止信号）
"""

import os
import sys
import json
import signal
import logging
import argparse
import threading
from datetime import datetime
from logging.handlers import RotatingFileHandler

from trainer import Trainer

# 项目根目录
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

# ============================================================================
# 日志配置
# ============================================================================

def setup_logging(log_dir: str = None, log_level: str = 'INFO'):
    """配置滚动日志"""
    if log_dir is None or not os.path.isabs(log_dir):
        log_dir = os.path.join(PROJECT_DIR, (log_dir or './logs').lstrip('./\\'))
    os.makedirs(log_dir, exist_ok=True)

    log_file = os.path.join(log_dir, 'training.log')
    level = getattr(logging, log_level.upper(), logging.INFO)

    # 根日志器
    logger = logging.getLogger()
    logger.setLevel(level)

    # 清除已有handler
    for h in logger.handlers[:]:
        logger.removeHandler(h)

    # 文件handler（每日滚动，保留30天）
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=10 * 1024 * 1024,  # 10MB
        backupCount=30
    )
    file_handler.setLevel(level)
    file_formatter = logging.Formatter(
        '%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    # 控制台handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_formatter = logging.Formatter(
        '%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%H:%M:%S'
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    return logger


# ============================================================================
# 配置加载
# ============================================================================

def load_config(config_path: str) -> dict:
    """加载配置文件（支持JSON和YAML）"""
    if not os.path.exists(config_path):
        logger = logging.getLogger(__name__)
        logger.warning(f"配置文件不存在: {config_path}，使用默认配置")
        return _default_config()

    with open(config_path, 'r', encoding='utf-8') as f:
        if config_path.endswith('.yaml') or config_path.endswith('.yml'):
            try:
                import yaml
                return yaml.safe_load(f)
            except ImportError:
                logger = logging.getLogger(__name__)
                logger.error("需要安装PyYAML来解析YAML配置")
                raise
        else:
            return json.load(f)


def _default_config() -> dict:
    """
    默认训练配置 v2.0 — 优先使用硬件自动检测的推荐值。

    如果 hardware_auto_config 可用，使用其推荐的参数；
    否则使用保守的通用默认值。
    """
    try:
        from hardware_auto_config import get_auto_config
        auto_config = get_auto_config()
        return auto_config
    except Exception:
        pass

    # 硬件检测失败时的保守默认值
    return {
        'training': {
            'modes': ['self', 'trad'],
            'learning_rate': 0.02,
            'batch_size': 1024,
            'num_mcts_simulations': 400,
            'total_steps': 500000,
            'use_cuda': True,
            'mixed_precision': True,
            'gradient_accumulation_steps': 1,
            'model_dir': os.path.join(PROJECT_DIR, 'models'),
            'log_dir': os.path.join(PROJECT_DIR, 'logs'),
            'save_interval_minutes': 5,
            'auto_eval_games': 100,
            'self_play': {
                'num_processes': 4,
                'mcts_simulations': 400,
            },
            'traditional': {
                'initial_depth': 4,
                'depth_range': [1, 8],
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
                'capacity': 2000000,
                'prioritized_alpha': 0.6,
                'prioritized_beta': 0.4,
                'sampling_weights': {
                    'self': 0.6,
                    'trad': 0.3,
                    'human': 0.1,
                }
            },
            'augmentation': {
                'enabled': True,
                'symmetry': True,
            }
        }
    }


# ============================================================================
# 优雅停机处理
# ============================================================================

class GracefulKiller:
    """优雅停机处理器"""

    def __init__(self, trainer: Trainer = None):
        self.trainer = trainer
        self.kill_now = False
        self._register_signals()

    def _register_signals(self):
        """注册信号处理器"""
        signal.signal(signal.SIGINT, self._exit_gracefully)
        signal.signal(signal.SIGTERM, self._exit_gracefully)
        # Windows不支持SIGUSR1/SIGUSR2
        if hasattr(signal, 'SIGBREAK'):
            signal.signal(signal.SIGBREAK, self._exit_gracefully)

    def _exit_gracefully(self, signum, frame):
        """优雅退出"""
        logger = logging.getLogger(__name__)
        logger.info(f"收到停止信号 (signal={signum})，正在优雅停机...")
        self.kill_now = True
        if self.trainer:
            logger.info("保存检查点并停止训练...")
            self.trainer.stop()
            self.trainer._save_checkpoint()
            logger.info("训练已安全停止")


# ============================================================================
# 主函数
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='五子棋AI静默训练服务',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python train_service.py --config config.yaml
  python train_service.py --config train_config.json --log-level DEBUG
        """
    )
    parser.add_argument('--config', type=str, default='train_config.json',
                        help='配置文件路径 (JSON或YAML)')
    parser.add_argument('--log-level', type=str, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                        help='日志级别')
    args = parser.parse_args()

    # 加载配置
    config = load_config(args.config)

    # 配置日志
    log_dir = config.get('training', {}).get('log_dir', './logs')
    logger = setup_logging(log_dir, args.log_level)

    logger.info("=" * 60)
    logger.info("五子棋AI静默训练服务启动 v2.0")
    logger.info(f"时间: {datetime.now().isoformat()}")
    logger.info(f"配置文件: {args.config}")
    logger.info(f"训练模式: {config.get('training', {}).get('modes', ['self'])}")

    # 打印硬件信息
    try:
        from hardware_auto_config import get_hardware_summary, format_hardware_summary
        hw = get_hardware_summary()
        for line in format_hardware_summary(hw).split('\n'):
            if line.strip():
                logger.info(line)
    except Exception:
        pass

    # 打印推荐配置概述
    train_cfg = config.get('training', {})
    net_cfg = config.get('network', {})
    logger.info(f"网络: {net_cfg.get('num_filters', 128)}通道, "
                f"{net_cfg.get('num_res_blocks', 7)}残差块")
    logger.info(f"训练: batch={train_cfg.get('batch_size')}, "
                f"MCTS={train_cfg.get('num_mcts_simulations')}, "
                f"LR={train_cfg.get('learning_rate')}, "
                f"AMP={train_cfg.get('mixed_precision', False)}")
    logger.info("=" * 60)

    # 检查人机模式（后台服务不支持）
    modes = config.get('training', {}).get('modes', [])
    if 'human' in modes:
        logger.warning("后台服务不支持人机对弈模式(human)，已自动忽略")
        modes.remove('human')
        config['training']['modes'] = modes

    if not modes:
        logger.error("没有有效的训练模式，退出")
        sys.exit(1)

    # 创建训练器
    logger.info("初始化训练器...")
    trainer = Trainer(config)

    # 注册优雅停机
    killer = GracefulKiller(trainer)

    # 启动训练
    logger.info("开始训练循环...")
    try:
        trainer.train_loop(games_per_iteration=5)
    except Exception as e:
        logger.exception(f"训练过程中发生错误: {e}")
    finally:
        logger.info("训练服务已退出")
        logger.info("=" * 60)


if __name__ == '__main__':
    main()
