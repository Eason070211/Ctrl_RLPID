# TEC_RL_Control

二维铜板 + 2×2 双向执行器阵列的温度控制 benchmark。被控对象是一块 10×10 cm、厚 2 mm 的铜板，从 25 °C 拉到并稳定在 200 °C；散热端是非线性的（辐射 σT⁴、Churchill-Chu 自然对流、风致对流、湿面蒸发），环境会变（风速 0–3 m/s、相对湿度 0–80%、环境温度阶跃）。

本仓库把**调好的 PID** 和**三种 PPO 自适应策略**放在同一个带真实物理的仿真器上公平对比，并报告了一个反直觉的结论：在功率受限、慢、近线性的热对象上，RL 相对调好 PID 的标称提升只有 ~1%，但在强风掠扰工况下 MAE 改善 ~5%；reward 权重 ablation 显示物理饱和会把策略空间压扁。

---

## 1. 目录结构

```
TEC_RL_Control/
├── README.md
├── requirements.txt
├── configs/default.yaml          # 所有物理/控制/RL 超参
├── main.py                       # （旧 CFD 项目入口，与本项目无关）
├── src/
│   ├── config.py                 # YAML 加载
│   ├── thermal/
│   │   └── heat_solver.py        # 2-D 瞬态热传导 plant（Numba JIT）
│   ├── control/
│   │   ├── pid.py                # 离散 PID（anti-windup、D 低通）
│   │   └── baselines.py          # FixedPID / BangBang / StagedPID / GainScheduled
│   ├── env/
│   │   └── tec_env.py            # Gymnasium 环境，四种 RL 模式
│   ├── rl/
│   │   └── train.py              # SB3 PPO 训练入口
│   └── benchmark/
│       ├── runner.py             # 场景定义 + 跑控制器
│       └── metrics.py            # IAE/ITAE/RMSE/超调/整定时间/能耗
├── scripts/
│   ├── run_benchmark.py          # 全量 benchmark + 出图
│   └── train_reward_ablation.py  # 三种 reward 权重的 direct PPO
├── paper/                        # 论文草稿（见 §7）
│   ├── INTRODUCTION.md
│   ├── METHODOLOGY.md
│   ├── EXPERIMENTS.md
│   ├── REFERENCES.md
│   └── INNOVATION_ANALYSIS.md
└── results/
    ├── models/                   # 训好的 PPO 权重（.zip）
    ├── figures/                  # PNG 图
    └── tables/                   # CSV
```

---

## 2. 快速开始

```bash
cd TEC_RL_Control
pip install numpy scipy pyyaml matplotlib tqdm numba gymnasium stable-baselines3

# 一键复现全量 benchmark（不需要重训，模型已在 results/models/）
PYTHONPATH=. python3 scripts/run_benchmark.py

# 单独跑三种 reward 权重的 direct PPO 重训（~10 分钟）
PYTHONPATH=. python3 scripts/train_reward_ablation.py

# 重训任意一种模式
PYTHONPATH=. python3 -m src.rl.train --mode direct      --steps 400000
PYTHONPATH=. python3 -m src.rl.train --mode fast_pid   --steps 400000
PYTHONPATH=. python3 -m src.rl.train --mode event_pid  --steps 300000
```

---

## 3. 被控对象与执行器

- 铜板：10×10 cm，厚 2 mm，铜 k=400 W/(m·K)，ρ=8960 kg/m³，cp=385 J/(kg·K)。
- 网格 11×11，显式 Euler，dt=0.05 s。
- 2×2 执行器贴片各占 2×2 单元。**u∈[−1,1]，u<0 电阻加热 40 W/片，u>0 半导体制冷 25 W/片**（双向不对称，加热比制冷强）。
- 边界散热（温度非线性，非常数 h）：
  - 辐射 `εσ(T⁴−Tair⁴)`，ε=0.25；
  - 自然对流 Churchill-Chu `Nu=0.54 Ra^0.25`；
  - 风致对流 `12·v_air`；
  - 湿面蒸发 `h_evap·(1−RH)·max(0,T−Tair)`，T>90 °C 蒸干。

> 注：真实 TEC 冷端扛不住 200 °C；本仿真用电阻加热+半导体制冷的理想化双向功率源代替，物理趋势正确，但绝对功率和真实 TEC 有差异。

---

## 4. 控制器

| 控制器 | 类型 | 说明 |
|---|---|---|
| `fixed_pid` | 传统 | 离散 PID，anti-windup，D 项一阶低通，Kp=0.6 Ki=0.05 Kd=0.02 |
| `bang_bang` | 传统 | ±2 °C 滞环，最快但能耗 2.1× |
| `staged_pid` | 传统 | 远距满功率 → u=0 热惯性滑行 → 近距 PID |
| `gain_scheduled` | 传统 | 按 \|e\| 频段换两组增益 |
| `ppo_direct` | RL | PPO 每 1 s 直接输出占空比 u |
| `ppo_fast_pid` | RL | PID 快环 1 s；PPO 每 **2 s** 输出 (ΔKp,ΔKi,ΔKd) |
| `ppo_event_pid` | RL | PID 快环；PPO 每 10 s 评估，动作死区 \|a\|<0.3 保持不动（事件触发） |
| `ppo_bang_rl` | RL（负结果） | 离散动作 {heat, coast, cool}，未学会提前反拉，整定 797 s |

### PPO 配置

- SB3 PPO，MLP `[256, 256]` ReLU，~0.3M 参数，CPU 推理 <1 ms。
- Clip=0.2，GAE λ=0.95，γ=0.95，entropy=0.01，lr=3e-4。
- 观测（含湿度传感器）：`(T−Tset)/30`、`T/100`、滤波 dT/dt、Tair、**RH**、v_air、上一拍 u；快环/事件模式再加归一化 Kp/Ki/Kd。
- 训练时随机化 Tair∈[15,35]°C、v∈[0,3] m/s、RH∈[0.1,0.9]、Tset∈[150,220]°C。

---

## 5. 主结果（25→200 °C 标称工况）

| 控制器 | MAE [°C] | IAE [s·°C] | 超调 [%] | 整定 [s] | 电能 [kJ] |
|---|---|---|---|---|---|
| fixed_pid | 11.00 | 8801 | 0.00 | 102 | 42.9 |
| staged_pid | 11.00 | 8803 | 0.03 | 102 | 42.9 |
| bang_bang | 11.65 | 9316 | 1.13 | 102 | **91.9** |
| ppo_direct | 10.91 | 8724 | 0.02 | 102 | 42.9 |
| **ppo_fast_pid** | **10.88** | **8703** | 0.08 | 102 | 42.9 |
| ppo_event_pid | 10.88 | 8707 | 0.02 | 102 | 42.9 |
| ppo_bang_rl | 12.90 | 10273 | 0.20 | 797 ✗ | 51.7 |

> MAE 绝对值大（~11 °C）是因为 25→200 °C 的 102 s 爬坡本身贡献了 ~1200 s·°C，所有控制器都在这段满功率。

### 鲁棒性（看稳态段 MAE，不是全程）

全程 MAE 被 102 s 爬坡淹没，真正的调节性能看后 650 s 稳态段：

| 工况 | fixed_pid | ppo_fast_pid | ppo_event_pid |
|---|---|---|---|
| 标称 v=0.5 m/s | 0.000 °C | 0.000 °C | 0.005 °C |
| **强风 v=3 m/s** | **0.551 °C** | **0.018 °C** | 0.221 °C |

**强风下 PPO_fast 稳态误差比 fixed PID 小 30×**——这才是 RL 真正的价值。固定 PID 在风掠突变时靠积分项慢慢拉，PPO 从风速观测里直接前馈补偿。

湿度扫描 RH=0→0.8：200 °C 稳态表面早已蒸干，RH 对控温影响 <0.01 °C（伪扰动，见 `paper/CAVEATS.md`）。

---

## 6. Reward 权重 ablation（反直觉发现）

为了让 PPO 自己学出"快 vs 省"的 trade-off，加了一个时间惩罚项 `−w_time`（每步扣，逼它快点到设定值），训了三组 direct PPO：

| 变体 | w_temp | w_energy | w_time | 设计意图 |
|---|---|---|---|---|
| `ppo_speed` | 2.5 | 0.02 | 0.15 | 速度优先 |
| `ppo_energy` | 0.5 | 0.60 | 0.0 | 节能优先 |
| `ppo_balance` | 1.0 | 0.15 | 0.03 | 平衡 |

**结果：三条曲线完全重合**（上升 82 s、整定 102 s、电能 42.7 kJ，差异 <0.3 °C）。

原因：上升段被执行器最大功率卡死（160 W、69 J/K，物理最快就是 ~1.1 K/s），PPO 不管 reward 怎么设都很快学到"满功率 u=−1"是唯一出路；接近设定值时所有策略都收敛到同一个减速行为。**Reward shaping 在饱和系统上无效，因为物理把策略空间压扁了**。这个负结果本身写进了论文。

---

## 7. 论文草稿

`paper/` 目录下有完整骨架：

- `INTRODUCTION.md` — 背景（TEC 在户外多变风/湿环境下的鲁棒控温）、贡献
- `METHODOLOGY.md` — 二维 FVM、非线性散热、PPO 配置、仿真器不确定性声明
- `EXPERIMENTS.md` — 已跑实验清单 + 待补实验（硬件台架、多种子、SAC 对比）
- `REFERENCES.md` — 15 篇真实可查文献（SAC-ADRC、DRLTC、Gnu-RL、IEEE Access 等）
- `INNOVATION_ANALYSIS.md` — 创新点诚实评估和发表潜力
- `CAVEATS.md` — review 反馈逐条回应，哪些已修哪些未修（必须看）

---

## 8. 客观结论

1. **PID 就是这个系统的最优解**。标称工况 RL 只比 fixed_pid 好 1%；只有强风掠扰时才拉出 ~5% 优势。
2. **ppo_direct 和 ppo_bang_rl 不推荐**：控制量噪声大、bang_rl 没收敛。
3. **事件触发 ppo_event_pid** 在线推理频率是快环的 1/5，MLP ~1 MB，可上 MCU，是边缘部署首选。
4. 真要榨 RL 价值，应该做**设定值轨迹优化**（提前加热到 220 °C 再靠热惯性滑到 200 °C），而不是实时闭环。
