# References（已检索到的真实文献）

## RL 调 PID / 热控制

1. **Jeon S., Kwak H., Lee W.** (2024). *A Study of Advancing Ultralow-Power 3D Integrated Circuits with TEI-LP Technology and AI-Enhanced PID Autotuning.* Mathematics, 12(4):543. MDPI. https://www.mdpi.com/2227-7390/12/4/543
   — 深度学习 PID 自整定用于 3D-IC 热管理，仿真验证。

2. **Li D., Gu Y., Sun L.** (2025). *SAC-Optimized Active Disturbance Rejection Controller for Thermal Regulation in Aerospace Applications.* Energy Proceedings, CUE 2025. https://www.energy-proceedings.org/wp-content/uploads/cue2025/5_Paper_1025025439.pdf
   — SAC 调 ADRC 参数，IAE 降 14.6%，整定时间降 36–58%；±5% 参数扰动鲁棒性。**对标对象**。

3. **Xiong Y., Guo L., Wang H., Huang Y., Liu C.** (2019). *Intelligent Thermal Control Algorithm Based on Deep Deterministic Policy Gradient for Spacecraft.* Journal of Thermophysics and Heat Transfer. http://159.226.165.120/bitstream/181722/64625/1/Intelligent%20Thermal%20Control%20Algorithm%20Based%20on.pdf
   — DDPG 热控制器（DRLTC），稳态误差降 40–90%，航天器主镜支撑。**对标对象**。

4. **Abadalcaraz V., Castilla M., Alvarez J.d.** (2024). *A Comparison of Classical and Reinforcement Learning-based Tuning Techniques for PI controllers.* IFAC PID 2024. https://skoge.folk.ntnu.no/prost/proceedings/PID-2024/0095.pdf
   — HVAC 一阶模型，RL-PI vs 极点配置 PI，能耗优化。

5. **Wu Y., Xing L., Liu X., Guo F.** (2022). *A New Solution to the PID18 Challenge: Reinforcement-Learning-based PI Control.* 34th CCDC.
   — 制冷系统多变量耦合，RL-PI。

## RL 在建筑/HVAC/数据中心热控

6. **Manjavacas A. et al.** (2024). *An experimental evaluation of deep reinforcement learning algorithms for HVAC control.* Artificial Intelligence Review. Springer. https://link.springer.com/content/pdf/10.1007/s10462-024-10819-x.pdf
   — SAC/TD3/PPO 在 Sinergym 上的可复现对比，指出泛化和增量学习挑战。

7. **Naug A., Quiñones-Grueiro M., Biswas G.** (2020). *A Relearning Approach to RL for control of Smart Buildings.* arXiv:2008.01879.
   — PPO 在非稳态建筑工况上的持续再学习。

8. **Chen B., Cai Z., Bergés M.** (2020). *Gnu-RL: A Practical and Scalable RL Solution for Building HVAC Control Using a Differentiable MPC Policy.* Frontiers in Built Environment. https://www.frontiersin.org/journals/built-environment/articles/10.3389/fbuil.2020.562239/pdf
   — 真实会议室部署，制冷需求降 16.7%。

9. **Liu J., Jiang C., Li X., Zhang W.** (2022). *Deep RL for the control of a desiccant wheel-based AC system.* 41st CCC, IEEE. https://ieeexplore.ieee.org/document/9901521
   — PPO 控湿度+温度，对比 DDPG。

10. **Adluri A., Karumanchi R., Reddy S.R.** (2025). *Discretizing Action Space to Improve Data Center Thermal Control with RL.* Stanford CS224R. https://cs224r.stanford.edu/spring_2025/projects/pdfs/CS_224R_Final_Report%20(3).pdf
    — Stick-Breaking PPO，极端工况成本降 62%。

11. **Al-Harbi A.F.N., Elhagh S.O., Alghamdi H.S.** (2025). *Simulation-Based Benchmarking of RL Algorithms for Adaptive Thermal Control in IoT-Enabled Smart Umbrella Systems.* IEEE Access. https://ieeexplore.ieee.org/document/11142253
    — SAC/DQN/PPO 在极端湿热（朝觐）工况对比，SAC 最优，提到 MCU 部署。**对标对象**。

## 边缘/嵌入式部署

12. **Ahmadi A., Abdelhafez H.A., Pattabiraman K., Ripeanu M.** (2023). *EdgeEngine: A Thermal-Aware Optimization Framework for Edge Inference.* ACM. https://dl.acm.org/doi/pdf/10.1145/3583740.3626616
    — RL 调 DVFS，能耗降 29%。

13. **Gong Y. et al.** (2024). *LOTUS: learning-based online thermal and latency variation management for two-stage detectors on edge devices.* DAC 2024, arXiv:2410.10847.
    — Jetson Orin Nano 上 DRL 温控。

## 热传导/Peltier 基础

14. 陶文铨. 《计算流体力学》. 西安交通大学出版社. — FVM 教材规范。
15. Incropera F.P. et al. *Fundamentals of Heat and Mass Transfer.* — Churchill-Chu 自然对流、辐射、蒸发关联式来源。
