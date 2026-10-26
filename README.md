# TEC_RL_Control

教学/研究型 2×2 帕尔贴（TEC）阵列温度控制 benchmark：二维瞬态热传导 + 对流 + 蒸发散热的铜板上，比较四类传统控制器与三种强化学习（PPO）策略，并用湿度 0→80% 做鲁棒性扫描。

## 快速开始

```bash
cd TEC_RL_Control
pip install numpy scipy pyyaml matplotlib tqdm numba gymnasium stable-baselines3
PYTHONPATH=. python3 scripts/run_benchmark.py        # 跑全量 benchmark + 出图
PYTHONPATH=. python3 -m src.rl.train --mode direct   # 重新训练
```

## 动作约定

`u ∈ [-1, 1]`，**u>0 制冷（正向通电），u<0 加热（反向通电）**。TEC 双向可用，不是纯加热执行器。

## 控制器

| 控制器 | 说明 |
|---|---|
| `fixed_pid` | 离散 PID，anti-windup，D 项低通 |
| `bang_bang` | ±0.5°C 滞环最快控制 |
| `staged_pid` | 远距满功率 → 热惯性滑行（u=0）→ 近距 PID |
| `gain_scheduled` | 人工按 \|e\| 频段换增益 |
| `ppo_direct` | PPO 直接输出占空比 u |
| `ppo_fast_pid` | PID 快环闭环，PPO 每 5s 增量调 Kp/Ki/Kd |
| `ppo_event_pid` | 长周期（20s）评估 PID 健康度，动作死区内保持不动 |

## 目录

```
src/thermal/    2-D 热传导 + Peltier + 对流蒸发 plant（Numba JIT）
src/control/    PID 与四类传统控制器
src/env/        Gymnasium 环境（三种 RL 模式）
src/rl/         SB3 PPO 训练入口
src/benchmark/  指标 + 场景 + runner
configs/        default.yaml
results/        models / figures / tables
```

## 物理模型

- 铜板 10×10 cm，厚 2 mm，铜 k=400 W/mK；2×2 TEC 贴片各占 2×2 单元。
- 显式 Euler 子步 0.05 s；TEC 准静态 Peltier 方程（含热电/焦耳/热漏），热沉对环境电导 5 W/K。
- 顶部自然+风致对流 `h=5+12·v_air`；湿面蒸发 `h_evap·(1-RH)·max(0,T-T_air)`，RH 升高则蒸发散热减弱。
