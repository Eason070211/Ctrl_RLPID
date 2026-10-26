# Experiments（实验记录草稿）

## 已跑实验

| # | 场景 | 扰动 | 控制器 | 输出 |
|---|---|---|---|---|
| E1 | 标称 25→200°C | Tair=25, v=0.5, RH=0.4 | 全部 8 个 | `results/figures/nominal_traj.png` |
| E2 | 热空气阶跃 | t=300s Tair 25→35 | 全部 | benchmark.csv 列 hot_air_step |
| E3 | 设定值阶跃 | t=300s Tset 200→220 | 全部 | benchmark.csv 列 setpoint_step |
| E4 | 湿度扫描 | RH ∈ {0,0.2,0.4,0.6,0.8} | 全部 | `results/figures/humidity_sweep.png` |
| E5 | 风速扫描 | v ∈ {0,1,2,3} m/s | 全部 | `results/figures/wind_sweep.png` |
| E6 | 二维温度场 | 固定 PID 启动过程 t=10,30,60,200s | — | `results/figures/temperature_contour.png` |

## 关键数字（标称 25→200°C）

| 控制器 | MAE | IAE | 超调 | 整定 | 电能 |
|---|---|---|---|---|---|
| fixed_pid | 11.00 | 8801 | 0% | 102s | 42.9 kJ |
| staged_pid | 11.00 | 8803 | 0.03% | 102s | 42.9 kJ |
| bang_bang | 11.65 | 9316 | 1.13% | 102s | 91.9 kJ |
| ppo_direct | 10.91 | 8724 | 0.02% | 102s | 42.9 kJ |
| ppo_fast_pid | 10.88 | **8703** | 0.08% | 102s | 42.9 kJ |
| ppo_event_pid | 10.88 | 8707 | 0.02% | 102s | 42.9 kJ |
| ppo_bang_rl | 12.90 | 10273 | 0.20% | **797s** ✗ | 51.7 kJ |

**相对 fixed_pid 的 IAE 改善**：ppo_fast = 1.1%，ppo_event = 1.1%，ppo_direct = 0.9%。

## 强风工况 v=3 m/s 下

| 控制器 | MAE |
|---|---|
| fixed_pid | 13.85 |
| ppo_fast_pid | **13.21**（改善 4.6%） |
| ppo_event_pid | 13.24 |
| bang_bang | 13.87 |

## 待补实验

- [ ] 3 个随机种子 × 8 控制器，报 mean±std。
- [ ] SAC/TD3 对照。
- [ ] 奖励权重扫描：w_energy ∈ {0.05, 0.15, 0.3, 0.6}，画 IAE vs kJ Pareto。
- [ ] 模型压缩：256×256 → 32×32 + int8 量化，测 STM32H7 推理延迟。
- [ ] 硬件台架：1× TEC + 铜板 + 热敏电阻，跑 E1/E5 两个工况。
- [ ] 更长训练（2M 步）重训 ppo_bang_rl，确认它是真不行还是欠训练。
