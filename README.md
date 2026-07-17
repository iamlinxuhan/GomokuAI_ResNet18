# GomokuAI_ResNet18 — 五子棋神经网络AI v2.0

基于 **AlphaZero 风格深度残差网络 + MCTS** 的五子棋对弈与训练系统。
15×15 标准棋盘，支持硬件自适应配置、人机对弈、自我对弈训练与 Windows 后台服务化。

## v2.0 新特性

### 🚀 传统 AI 全面重写（训练速度提升 60 倍）
- **迭代加深 + PVS 零窗口试探**：大幅提升搜索效率
- **置换表 v2.0**：带年龄字段 + 深度优先替换策略
- **杀手走法 + 历史启发**：更好的走法排序，剪枝更高效
- **训练模式**：纯 CPU 快速搜索，避免 GPU 同步瓶颈（每步仅 150ms）
- **对弈模式**：GPU 全力搜索，棋力更强

### 🔧 硬件自适应配置
- **自动 GPU 检测**：识别型号、显存、Tensor Cores
- **5 级硬件分级**：从纯 CPU 到旗舰 GPU 自动适配
- **智能参数推荐**：根据硬件自动选择网络大小、Batch Size、MCTS 模拟次数
- **一键应用**：训练面板点"应用推荐配置"即可

### 📊 训练监控图表
- **损失曲线**：总损失、策略损失、价值损失实时绘制
- **胜率曲线**：NN vs 传统AI 胜率趋势
- **自动参数趋势**：MCTS 模拟数、温度变化可视化
- **学习率曲线**：调度过程一目了然

### 🎯 优先经验回放 (PER)
- 按 TD-error 优先级采样，加速收敛
- 新样本自动获得最高优先级
- 重要性采样权重修正偏差

### 🔄 自动调参与 UI 同步
- 自动调控的 MCTS、温度、深度等参数实时反映到面板滑条
- 训练中修改的参数持久化保存

## 功能特性

### 🎮 对弈功能
- **人机对弈**：通过 PyQt6 图形界面与 AI 对弈，支持执黑/执白切换
- **AI vs AI**：观看神经网络自我对弈或与传统 AI 对抗，**对局结束后自动开始新局**
- **GPU 加速**：自动检测 CUDA，MCTS 搜索在 GPU 上运行

### 🧠 训练系统
- **三种训练模式自由组合**：
  - **自我对弈（Self-Play）**：AlphaZero 风格强化学习
  - **传统 AI 对抗（Traditional）**：Minimax + Alpha-Beta，阶梯式课程学习
  - **人机数据收集（Human）**：模仿学习与偏好注入
- **智能自动调参**：全自动调控 MCTS 温度、噪声强度、PUCT 常数、模拟次数、每轮对弈局数等参数
- **自动难度调控**：自动评估 NN 胜率，动态调整传统 AI 搜索深度
- **硬件自适应**：根据 GPU 显存自动选择网络规模（32ch~256ch）
- **优先经验回放 (PER)**：基于 TD-error 的优先采样
- **数据增强**：8 种对称变换，训练时自动应用
- **混合精度训练 (AMP)**：Tensor Cores 自动加速
- **梯度累积**：支持更大有效 Batch Size
- **热重启学习率调度**：Cosine Annealing with Warm Restarts

### 📊 训练监控
- 实时损失曲线与 Win Rate 图表（matplotlib 嵌入）
- 自动参数趋势图表（MCTS/温度/LR）
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

### 硬件自适应等级

| 等级 | 适用硬件 | 网络规模 | Batch | MCTS |
|------|---------|---------|-------|------|
| tier_0 | 无 GPU / 仅 CPU | 32ch/5block | 256 | 100 |
| tier_1 | 低端 GPU (< 4GB) | 64ch/7block | 256 | 200 |
| tier_2 | 中端 GPU (4-6GB) | 128ch/7block | 512 | 520 |
| tier_3 | 中高端 GPU (6-8GB) | 128ch/10block | 1024 | 600 |
| tier_4 | 高端 GPU (8-12GB) | 192ch/10block | 2048 | 800 |
| tier_5 | 旗舰 GPU (> 12GB) | 256ch/15block | 4096 | 1200 |

启动时自动检测硬件并应用推荐配置。

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

启动时自动检测硬件并显示推荐配置。也可运行硬件检测脚本查看详细信息：

```bash
python hardware_auto_config.py
```

## 使用指南

### 🕹️ 对弈模式

GUI 左侧为 15×15 棋盘，右侧为控制面板：

| 控件 | 说明 |
|------|------|
| **对局模式** | 选择玩家执黑/执白、AI vs AI、传统 AI vs NN |
| **训练数据收集** | 勾选后并发多局 AI 对弈，数据自动保存，**循环持续运行** |
| **并行局数** | 控制 AI vs AI 演示模式下并发对弈局数 |

**演示模式自动循环**：AI vs AI 和 传统AI vs NN 模式下，每局结束后自动 2 秒后开始新局，无需手动点击。

### 🏋️ 训练模式

点击底部 "训练面板" 按钮进入训练面板，包含 3 个标签页：

**训练控制** — 设置训练参数：
1. 勾选训练模式（自我对弈 / 传统AI对抗 / 人机收集）
2. 调整超参数：学习率、Batch Size、MCTS 模拟次数、总步数等
3. 调整数据采样权重（Self/Trad/Human 联动绑定，总和恒为 100%）
4. 点击 "开始训练"

**硬件信息** — 查看硬件检测报告和推荐配置：
- 点击 **"应用推荐配置"** 一键设置最优参数
- 点击 **"运行 GPU 基准测试"** 获得精确的 Batch 推荐

**监控** — 训练过程可视化：
- 4 张实时图表：损失曲线、胜率曲线、自动参数趋势、学习率
- 紧凑信息栏：步数、缓冲、损失、LR、胜率、深度、ELO
- 训练日志输出

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
├── network.py            # 神经网络 v2.0：ResNet + SE注意力 + 硬件自适应
├── mcts.py               # MCTS 搜索：PUCT 选择 + GPU 批量评估
├── traditional_ai.py     # 传统 AI v2.0：迭代加深 + 置换表 + CPU/GPU双模式
├── trainer.py            # 训练系统 v2.0：PER + 数据增强 + 硬件感知
├── gui.py                # 图形界面 v2.0：棋盘 + 训练面板 + 实时图表
├── hardware_auto_config.py # 硬件自动检测与智能配置推荐（新增）
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

### 手动配置 vs 自动配置

系统支持两种模式：
1. **自动配置（推荐）**：不修改配置文件，`hardware_auto_config.py` 启动时自动检测硬件并推荐参数
2. **手动配置**：修改 `config.yaml` 或 `train_config.json`，手动指定的参数会覆盖自动检测值

### 关键配置参数

| 参数 | 说明 | 自动推荐 |
|------|------|---------|
| `training.modes` | 训练模式列表 | `["self", "trad"]` |
| `network.num_filters` | 卷积通道数（根据显存自动选择） | 32~256 |
| `network.num_res_blocks` | 残差块数 | 5~15 |
| `training.learning_rate` | 学习率（按 batch 自动缩放） | 0.005~0.02 |
| `training.batch_size` | 批次大小（按显存自动选择） | 256~4096 |
| `training.num_mcts_simulations` | MCTS 模拟次数 | 100~1200 |
| `training.mixed_precision` | 混合精度 AMP | 有 GPU 时自动开启 |
| `training.replay_buffer.prioritized_alpha` | PER 优先级强度 | 0.6 |
| `training.augmentation.enabled` | 数据增强 | True |
| `training.traditional.initial_depth` | 传统 AI 初始深度 | 2~8 |

## 技术架构

### 神经网络架构 v2.0

```
输入: (4, 15, 15)
  │ 4通道: [黑棋位, 白棋位, 上一步标记, 颜色常数]
  ▼
初始卷积层 (N filters, 3×3) + BN + ReLU
  │ N = 硬件自动选择 (32/64/128/192/256)
  ▼
残差塔 × M  (含 SE 注意力模块)
  │ M = 硬件自动选择 (5/7/10/15)
  │ 每块: Conv→BN→ReLU→Conv→BN→SE→ ⊕ → ReLU
  ▼
  ├── 策略头 ──→ Conv(1×1)→FC(225) → 落子概率分布
  └── 价值头 ──→ Conv(1×1)→FC(128)→FC(1)→tanh → 胜率 [-1,1]
```

- **参数量**：3 万 ~ 1200 万（硬件自动选择）
- **SE 注意力**：Squeeze-and-Excitation 通道注意力模块
- **权重初始化**：Kaiming 正态初始化 + 策略头零初始化

### MCTS 搜索

使用 PUCT 公式选择节点，Dirichlet 噪声增强根节点探索，GPU 批量评估叶节点。

### 传统 AI v2.0

**对弈模式（GPU 全力搜索）：**
- 迭代加深 + PVS 零窗口试探
- 置换表：带年龄字段 + 深度优先替换
- 杀手走法 + 历史启发
- 主变例表 (PV-Table) 引导走法排序

**训练模式（CPU 快速搜索）：**
- 纯 numpy 操作，零 GPU 同步
- 深度 2-3 固定搜索
- 每步仅 150ms，比 GPU 模式快 50 倍

### 训练流程 v2.0

```
生成数据 ──→ 经验回放池(PER) ──→ 数据增强 ──→ 神经网络训练
   ↑                                              │
   │  Self-Play / vs Traditional / Human          │
   └──────────────────────────────────────────────┘

损失 = 策略交叉熵 + 价值MSE + L2正则化
优化器: SGD + Momentum + Cosine Warm Restarts
精度: 混合精度 AMP (Tensor Cores 自动加速)
```

### 自动调参机制 v2.0

#### 5 阶段精细调控
训练过程分为 `warmup → early → mid → late → converge` 五个阶段，参数曲线更平滑。

#### 传统 AI 对手难度调控
每 N 局自动评估 NN vs 传统 AI，连续 3 次胜率偏离目标才调整深度，防止震荡。

#### 训练停滞检测
监控 loss 下降趋势，连续停滞时自动增加探索（提高温度 + PUCT）。

#### 神经网络超参数智能调控

| 参数 | 调控策略 | 范围 |
|------|---------|------|
| **MCTS 探索温度** | 5 阶段衰减 | 1.5 → 0.08 |
| **Dirichlet 噪声 ε** | 随训练进度线性衰减 | 0.25 → 0.05 |
| **MCTS 模拟次数** | 根据胜率趋势动态调整 | 100～800+ |
| **PUCT 探索常数** | 监控 loss 收敛自动微调 | 1.2～4.0 |
| **每轮对弈局数** | 根据经验池健康度动态调整 | 2～10 |
| **温度截止步数** | 阶段自适应 | 8～20 |

## 依赖清单

```
torch >= 2.0.0        # 深度学习框架
PyQt6 >= 6.5.0        # 图形界面
numpy >= 1.24.0       # 数值计算
matplotlib >= 3.7.0   # 图表绘制（监控面板）
PyYAML >= 6.0         # YAML 配置解析
```

## 常见问题

### Q: 训练很慢怎么办？
A: v2.0 已优化传统 AI 搜索。如果仍慢，检查：
- 训练面板中传统 AI 深度是否 ≤ 3（训练模式自动使用 CPU 快速搜索）
- MCTS 模拟次数是否合理（50~100 次足够训练）
- GPU 驱动和 CUDA 是否正确安装

### Q: 如何让 AI 更强？
A: 对弈模式下增加 MCTS 模拟次数（200~800）和传统 AI 深度（4~8）。
训练时确保 "传统AI对抗" 模式开启，自动难度调控会逐步提升对手强度。

### Q: 如何适配其他电脑？
A: 直接运行即可。`hardware_auto_config.py` 启动时自动检测 GPU/CPU 并应用最佳配置。
也可在训练面板的"硬件信息"标签页查看检测结果并手动调整。

## 许可

MIT License

---

**作者**: [@iamlinxuhan](https://github.com/iamlinxuhan)
