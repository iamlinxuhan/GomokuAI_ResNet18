# GomokuAI_ResNet18 — 五子棋神经网络AI

基于 **AlphaZero 风格深度残差网络 + MCTS** 的五子棋对弈与训练系统。
15×15 标准棋盘，ResNet 架构约 160 万参数，支持人机对弈、自我对弈训练与 Windows 后台服务化。

## 功能特性

### 🎮 对弈功能
- **人机对弈**：通过 PyQt6 图形界面与 AI 对弈，支持执黑/执白切换
- **AI vs AI**：观看神经网络自我对弈或与传统 AI 对抗
- **实时调整**：滑动条即时调整 MCTS 模拟次数（50~800）和传统 AI 搜索深度（1~8）
- **GPU 加速**：自动检测 CUDA，MCTS 搜索与传统 AI 评估均在 GPU 上运行

### 🧠 训练系统
- **三种训练模式自由组合**：
  - **自我对弈（Self-Play）**：AlphaZero 风格强化学习
  - **传统 AI 对抗（Traditional）**：Minimax + Alpha-Beta，阶梯式课程学习
  - **人机数据收集（Human）**：模仿学习与偏好注入
- **智能自动调参**：全自动调控 MCTS 温度、噪声强度、PUCT 常数、模拟次数、每轮对弈局数等参数，基于训练进度、loss 收敛状态和胜率趋势自适应决策
- **自动难度调控**：每 100 局自动评估 NN 胜率，动态调整传统 AI 搜索深度
- **模型覆盖保存**：始终覆盖保存 `model.pt`（推理）和 `training_state.pt`（断点续训），不产生冗余文件
- **自动保存**：训练中每 5 分钟自动保存模型检查点（间隔可配）
- **经验回放池**：200 万容量循环缓冲区，多来源加权采样
- **数据增强**：8 种对称变换（旋转 + 翻转）
- **混合精度训练（AMP）**：显著加速 GPU 训练

### 📊 训练监控
- 实时损失曲线与 Win Rate 图表
- ELO 评分跟踪
- 传统 AI 难度自动调控可视化
- 经验池采样权重实时调整

### ⚙️ 部署
- **静默训练服务**：可在无 GUI 环境下后台训练
- **Windows 服务**：一键安装/卸载，支持开机自启

## 环境要求

| 项目 | 要求 |
|------|------|
| 操作系统 | Windows 10/11 64-bit（主要）；Ubuntu 20.04+（可选） |
| Python | 3.10 / 3.11 |
| GPU（推荐） | NVIDIA GPU + CUDA 11.8+（CPU 也可运行） |
| 内存 | 8GB+ |

## 快速开始

### 1. 克隆仓库

```bash
git clone https://github.com/iamlinxuhan/GomokuAI_ResNet18.git
cd GomokuAI_ResNet18
```

### 2. 创建虚拟环境

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux / macOS
source venv/bin/activate
```

### 3. 安装依赖

```bash
pip install -r requirements.txt
```

**GPU 用户**额外安装 PyTorch CUDA 版本：

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 4. 启动 GUI

```bash
python gui.py
```

## 使用指南

### 🕹️ 对弈模式

GUI 左侧为 15×15 棋盘，右侧为控制面板：

| 控件 | 说明 |
|------|------|
| **对局模式** | 选择玩家执黑/执白、AI vs AI、传统 AI vs NN |
| **训练数据收集** | 勾选后将人类走法存入经验池供后续训练 |
| **MCTS 模拟** | 滑动条 50~800，越大 AI 棋力越强但响应越慢 |
| **传统 AI 深度** | 滑动条 1~8，控制 Minimax 搜索深度 |

### 🏋️ 训练模式

点击顶部 "训练" 标签进入训练面板：

1. 勾选需要的训练模式（自我对弈 / 传统 AI 对抗 / 人机收集）
2. 通过滑动条调整超参数：
   - 学习率：0.0001 ~ 0.1
   - 批量大小：64 ~ 4096
   - MCTS 模拟次数：50 ~ 800
   - 总训练步数：10000 ~ 1000000
   - 自动保存间隔：1 ~ 60 分钟（默认 5 分钟）
   - 自动调参评估局数：10 ~ 200 局（默认 100 局）
3. 调整经验池采样权重
4. 点击 "开始训练"

### 🖥️ 命令行训练

```bash
python gui.py --mode train --config train_config.json
```

### 🔧 静默训练服务

```bash
python train_service.py --config config.yaml
```

### 🪟 Windows 服务（开机自启）

以**管理员**身份运行：

```bash
# 安装服务（开机自启静默训练）
install_service.bat

# 卸载服务
uninstall_service.bat
```

也可以使用 VBS 脚本实现开机自启（无需管理员）：

```bash
# 将 autostart.vbs 放入 Windows 启动文件夹即可
# Win+R → shell:startup → 复制 autostart.vbs
```

## 模型管理

采用**覆盖保存策略**，仅保留 2 个文件：

| 文件 | 用途 |
|------|------|
| `models/model.pt` | 模型权重 + 训练步数 + ELO，供 GUI 推理加载 |
| `models/training_state.pt` | 优化器状态 + 自动调参参数，供断点续训 |

每次保存和定时自动保存均覆盖旧文件，不会产生冗余检查点。

## 项目结构

```
GomokuAI_ResNet18/
├── game.py               # 游戏核心：15×15 棋盘逻辑、规则判定
├── network.py            # 神经网络：ResNet + 策略头 + 价值头 (~160万参数)
├── mcts.py               # MCTS 搜索：PUCT 选择 + GPU 批量评估
├── traditional_ai.py     # 传统 AI：Minimax + Alpha-Beta + 模式识别
├── trainer.py            # 训练系统：混合训练 + 经验池 + 智能自动调参
├── gui.py                # 图形界面：PyQt6 棋盘 + 训练面板 + 实时图表
├── train_service.py      # 静默训练服务（无 GUI）
├── requirements.txt       # Python 依赖
├── train_config.json      # 训练配置（JSON 格式）
├── config.yaml            # 训练配置（YAML 格式）
├── autostart.vbs          # 开机自启 VBS 脚本
├── install_service.bat    # Windows 服务安装脚本
├── uninstall_service.bat  # Windows 服务卸载脚本
├── models/                # 模型存储目录
│   ├── model.pt           # 模型权重（覆盖保存，供推理使用）
│   └── training_state.pt  # 训练状态（覆盖保存，断点续训）
└── README.md
```

## 训练配置说明

配置文件 `train_config.json` / `config.yaml` 主要参数：

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `training.modes` | 训练模式列表 | `["self", "trad"]` |
| `training.learning_rate` | 学习率 | `0.02` |
| `training.batch_size` | 批次大小 | `1024` |
| `training.num_mcts_simulations` | MCTS 模拟次数 | `400` |
| `training.total_steps` | 总训练步数 | `500000` |
| `training.use_cuda` | 是否使用 GPU | `true` |
| `training.save_interval_minutes` | 自动保存间隔（分钟） | `5` |
| `training.auto_eval_games` | 自动评估局数 | `100` |
| `training.traditional.initial_depth` | 传统 AI 初始搜索深度 | `4` |
| `training.traditional.depth_range` | 深度调整范围 | `[1, 8]` |
| `training.traditional.target_win_rate` | 目标 NN 胜率 | `0.20` |
| `training.traditional.win_rate_window` | 胜率统计窗口 | `100` |
| `training.replay_buffer.capacity` | 经验池容量 | `2000000` |
| `training.replay_buffer.sampling_weights` | 各来源采样权重 | `{"self":0.6,"trad":0.3,"human":0.1}` |

## 技术架构

### 神经网络架构

```
输入: (4, 15, 15)
  │ 4通道: [黑棋位, 白棋位, 上一步标记, 颜色常数]
  ▼
初始卷积层 (128 filters, 3×3) + BN + ReLU
  │
  ▼
残差塔 × 7
  │ 每块: Conv(3×3)→BN→ReLU→Conv(3×3)→BN → ⊕ → ReLU
  ▼
  ├── 策略头 ──→ Conv(1×1, 32)→FC(225) → 落子概率分布
  └── 价值头 ──→ Conv(1×1, 32)→FC(128)→FC(1)→tanh → 胜率 [-1,1]
```

- **参数量**：约 160 万
- **权重初始化**：Kaiming 正态初始化
- **策略输出**：初始化为零（均匀分布）

### MCTS 搜索

使用 PUCT 公式选择节点：

$$a^* = \arg\max_a \left[ Q(s,a) + c_{puct} \cdot P(s,a) \cdot \frac{\sqrt{\sum_b N(s,b)}}{1 + N(s,a)} \right]$$

- Dirichlet 噪声增强根节点探索
- GPU 批量评估叶节点
- 搜索完成后按访问次数比例采样

### 传统 AI

- **Minimax + Alpha-Beta 剪枝**
- 模式匹配评估：活四 > 冲四 > 活三 > 活二
- 支持深度 1~8，深度越深棋力越强

### 训练流程

```
生成数据 ──→ 经验回放池 ──→ 采样批次 ──→ 神经网络训练
   ↑                                          │
   │    Self-Play / vs Traditional / Human    │
   └──────────────────────────────────────────┘

损失 = 策略交叉熵 + 价值MSE + L2正则化
优化器: SGD + Momentum + Cosine退火
```

### 自动调参机制

#### 传统 AI 对手难度调控
1. 定期运行自动评估赛 NN vs 传统 AI
2. 统计 NN 胜率
3. 胜率高 → 加深传统 AI 搜索深度（更强的对手）
4. 胜率低 → 降低传统 AI 搜索深度（更容易的对手）
5. 目标维持 NN 胜率在目标区间，实现阶梯式课程学习

#### 神经网络超参数智能调控
训练期间 **AutoParameterController** 自动调整以下参数，无需人工干预：

| 参数 | 调控策略 | 范围 |
|------|---------|------|
| **MCTS 探索温度** | Early→Mid→Late 三阶段衰减 | 1.5 → 0.1 |
| **Dirichlet 噪声 ε** | 随训练进度线性衰减 | 0.25 → 0.05 |
| **MCTS 模拟次数** | 根据 vs 传统AI 胜率动态调整 | 100～800 |
| **PUCT 探索常数** | 监控 loss 标准差自动微调 | 1.2～4.0 |
| **每轮对弈局数** | 根据经验池健康度动态调整 | 3～10 |
| **温度截止步数** | 三阶段切换（Early=12, Mid=18, Late=10） | 10～18 |

## 依赖清单

```
torch >= 2.0.0        # 深度学习框架
PyQt6 >= 6.5.0        # 图形界面
numpy >= 1.24.0       # 数值计算
matplotlib >= 3.7.0   # 图表绘制
PyYAML >= 6.0         # YAML 配置解析
```

## 许可

MIT License

---

**作者**: [@iamlinxuhan](https://github.com/iamlinxuhan)
