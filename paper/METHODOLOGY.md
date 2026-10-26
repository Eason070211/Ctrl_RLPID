# Methodology（方法学草稿）

## 1. 被控对象

薄铜板 Lx=Ly=0.1 m，Lz=0.002 m，铜 k=400 W/(m·K)，ρ=8960，cp=385。二维能量方程：

ρ cp ∂T/∂t = k ∇²T + q_act/Lz − (q_conv + q_rad + q_evap)/Lz

- 扩散项：5 点二阶中心差分，边界 Neumann（绝缘）。
- 时间积分：显式 Euler，dt=0.05 s（满足 CFL）。
- 2×2 TEC 贴片各占 2×2 单元；u∈[−1,1]，u<0 电阻加热 40 W/片，u>0 半导体制冷 25 W/片（双向不对称）。

## 2. 边界散热（温度非线性）

- **辐射**：q_rad = εσ(T⁴−Tair⁴)，ε=0.25。
- **自然对流**（Churchill-Chu，水平板层流）：Nu=0.54·Ra_L^0.25，Ra=gβΔTL³/(νa)。
- **风致对流**：h_wind = 12·v_air。
- **蒸发**：q_evap = h_evap·(1−RH)·max(0,T−Tair)，仅当 T<90°C（湿面蒸干）。

## 3. 控制器族

| 控制器 | 动作 | 频率 |
|---|---|---|
| FixedPID | 离散 PID，anti-windup，D 低通 | 1 s |
| BangBang | ±2°C 滞环 | 1 s |
| StagedPID | 远距满功率→u=0 滑行→近距 PID | 1 s |
| GainScheduled | 按 \|e\| 频段换两组增益 | 1 s |
| PPO-direct | MLP 直接输出 u | 1 s |
| PPO-fast-PID | PID 快环 1 s；PPO 每 2 s 输出 (ΔKp,ΔKi,ΔKd) | 1 s / 2 s |
| PPO-event-PID | PID 快环；PPO 每 10 s 评估，动作死区 \|a\|<0.3 保持不动 | 1 s / 10 s |
| PPO-bang-rl | 离散动作 {heat, coast, cool} | 1 s |

## 4. PPO 配置

- SB3 PPO，MLP [256,256] ReLU，高斯/分类头。
- Clip=0.2，GAE λ=0.95，γ=0.95，entropy=0.01，n_epochs=10，lr=3e-4。
- 训练步数：direct/fast/bang 400k，event 300k；CPU 4 envs。
- 观测：(T−Tset)/30、T/100、滤波 dT/dt、Tair、RH、v_air、u_prev；快环/事件模式加归一化 Kp/Ki/Kd；事件模式加"距上次调参步数"和 PID 健康度。
- 奖励：−wT·e² − wE·E² − wdu·Δu² − 越界重罚；事件模式加 −wdK·|ΔK|。
- 训练扰动随机化：Tair∈[15,35]°C，v∈[0,3] m/s，RH∈[0.1,0.9]，Tset∈[150,220]°C。

## 5. 指标

IAE、ITAE、RMSE、MAE、超调%、上升时间、整定时间（2% 带）、总电能 kJ、控制 effort Σ|Δu|、越界时间。

## 6. 仿真器不确定性声明

- 真实 TEC 的热阻随温度非线性、接触热阻、热沉热容未建模。
- 辐射率 ε 取 0.25（抛光铜），实际氧化后可达 0.6–0.8。
- 蒸发模型是线性近似，未考虑水滴脱落/毛细效应。
- 没有三维边缘效应（板边散热）。
