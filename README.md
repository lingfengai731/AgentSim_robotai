# 🚖 AgentSim — 基于 LLM 多智能体的 Robotaxi 调度仿真系统

> **LLM-Powered Multi-Agent Robotaxi Dispatch Simulation**
>
> 4 个角色 Agent（Coordinator · Dispatcher · Driver · Analyst）协同驱动，
> 仿真 Robotaxi 订单分配、车辆调度、路径规划全流程，
> 并与随机调度基线进行量化对比实验。

---

## ✨ 核心特性

| 特性 | 说明 |
|------|------|
| 🤖 **LLM 驱动决策** | Dispatcher / Driver / Analyst 三类 Agent 调用大模型做调度推理与播报 |
| ♟️ **匈牙利算法最优调度** | Kuhn-Munkres 批量全局最优匹配（工业级 OR 方案），与 LLM 策略热切换对照 |
| ⭐ **自研 Enhanced-KM 改进** | 在标准 KM 上叠加「前瞻指派 + 空闲车需求重定位」，需求热点下完成率 ↑约5%、接驾距离 ↓ |
| 🔬 **消融实验** | 逐模块开关验证各改进贡献，数据驱动确定最优模块组合 |
| 🗺️ **A\* 绕障真实路网** | 地图含建筑障碍物，车辆用 A\* 栅格路径规划绕障行驶；调度代价用真实路网距离（灵感来自作者一级项目 RRT-Connect 路径规划） |
| 📊 **四方量化对比** | 5 seed 重复实验，LLM vs 标准KM vs 增强KM vs 随机，均值 ± 标准差统计 |
| 🌐 **Web 交互界面** | Figma 风格深色仪表盘：实时地图（光晕 + 接驾路径）+ KPI 卡片 + 决策日志，可一键部署公网 |
| 🖥️ **桌面 GUI** | 现代深色仪表盘，渲染建筑/热点/A\* 规划路径/行驶轨迹，四策略热切换 + 速度调节 |
| 📈 **自动报告生成** | Analyst Agent 仿真结束后自动输出中文运营分析报告 |

---

## 🏗️ 系统架构

```
乘客请求（泊松到达）
      ↓
┌─────────────────────────────────────────┐
│          Coordinator Agent              │  ← 总协调，管理全局状态与冲突仲裁
└──────┬──────────────────────────────────┘
       ├──→  Dispatcher Agent   ──→  LLM 逐单推理：选择距离最优的空闲车辆
       ├──→  匈牙利算法引擎       ──→  批量全局最优匹配（可与 LLM 热切换）
       ├──→  Driver Agent × N   ──→  LLM 播报：接单 / 到达 / 送达事件
       └──→  Analyst Agent      ──→  LLM 报告：KPI 分析 + 改进建议
                  ↓
          仿真环境（N×N 网格地图，曼哈顿距离）
          离散事件驱动：乘客生成 → 批量分配 → 车辆移动 → 状态更新
                  ↓
     可视化层（Web / 桌面 / 图表文件）
```

> **核心算法升级：从"贪心就近"到"批量全局最优"再到"自研改进"**
> 原始 LLM/随机调度是"来一单分一单"的贪心策略，多个订单同时出现时无法全局协调。
> 新增的**匈牙利算法（Kuhn-Munkres）调度器**采用工业界（滴滴 / Uber）的
> **批量延迟匹配（Batch Matching）**思路：在时间窗内聚合"待派乘客 × 空闲车辆"，
> 构造接驾距离代价矩阵，求解**总等待距离最小**的二分图最优指派
> （`agents/hungarian_dispatcher.py`，自研 O(n³) 实现 + scipy 加速兜底）。
> 同时修复了旧版"乘客生成时若无空闲车则永久遗漏"的缺陷——现每步重试所有待派订单。

---

## ⭐ 自研改进：Enhanced-KM（在匈牙利算法基础上的改进版）

标准 KM 只对**当前快照**做最优匹配，看不到"未来"。我们在保留 KM 全局最优内核的前提下，
针对真实运营叠加三个改进模块（`agents/enhanced_km_dispatcher.py`），并通过**消融实验**确定有效组合：

| 模块 | 标准 KM 的问题 | 改进做法 | 实测结论 |
|------|---------------|---------|---------|
| ① 等待时间感知（Aging） | 久等乘客与新乘客同权，可能饿死 | `cost −= α×已等待步数` | 纯公平性策略，**过载时牺牲吞吐 → 默认关闭** |
| ② 前瞻指派（Anticipatory） | 不管车送达后停哪 | `cost += λ×(下车点→需求热点距离)` | **高负载（供不应求）下全面占优** |
| ③ 空闲车重定位（Rebalancing）★ | 空闲车原地不动，"等更近的车"永远等不来 | 空闲车每步朝**在线估计的需求重心**巡航一格 | **低负载下显著提升完成率**，与②负载互补 |

> **为什么需要"需求热点"？** 真实城市需求集中在 CBD / 枢纽，而非全图均匀。
> 我们给仿真加入**需求热点**（`config.USE_HOTSPOTS`，乘客上车点按高斯分布聚集），
> 这才让"前瞻 + 重定位"有可利用的空间结构——也是高级调度算法相对就近匹配的价值所在。

**A/B 实测（默认配置：10² 网格 · 4 车 · 需求热点 + 建筑障碍 · A\* 路网 · 5 seed）：**

| 指标 | ⭐ Enhanced-KM | ♟️ 标准 KM | 提升 |
|------|---------------|-----------|------|
| **订单完成率** | **66.8%** | 65.1% | **+1.7%** |
| **平均等待** | **2.83 步** | 4.09 步 | **−1.26 步** |
| **最长等待** | **6.2 步** | 7.4 步 | **−1.2 步** |
| **平均接驾距离** | **3.63 步** | 4.67 步 | **−1.04 步** |

> 运行 `python compare.py` 复现，输出 `outputs/comparison_four_way.png`（四方对比）
> 与 `outputs/ablation_enhanced_km.png`（消融实验）。默认开启障碍物 + 需求热点。

---

## 🗺️ A\* 真实路网（融合作者一级项目的路径规划专长）

原系统车辆"曼哈顿直线瞬移、穿楼而过"。本项目引入作者大三一级项目
《基于基因调控网络的改进 RRT-Connect 路径规划》中的**碰撞检测 + 无障碍路径搜索**思想，
在离散网格上实现 **A\* 路径规划**（`simulation/pathfinding.py`，4 邻域、曼哈顿启发式、LRU 缓存）：

- **地图含建筑障碍物**（`config.USE_OBSTACLES`），车辆/乘客生成均避障；
- **执行层**：车辆每步沿 A\* 最短路**绕障行驶**，桌面 GUI 实时绘制规划路径与行驶轨迹；
- **调度层**：匈牙利 / 增强KM 的代价矩阵采用 **A\* 真实路网距离**而非曼哈顿近似，与执行层一致；
- 关闭 `USE_OBSTACLES` 时自动退化为曼哈顿直行（与早期实验完全一致，保证可复现）。

> 这条线把"路径规划（大三）"与"多智能体调度（本项目）"串成统一的简历叙事：
> **调度层做高层决策、路径层做底层执行**，正是真实自动驾驶车队的分层架构。

---

## 📊 实验结果（四方对比，默认配置 · 5 seed 均值）

运行 `python compare.py` 复现（输出 `outputs/comparison_four_way.png`）：

| 指标 | ⭐ 增强KM | ♟️ 标准KM | 🤖 LLM 调度 | 🎲 随机基线 |
|------|---------|----------|------------|------------|
| **订单完成率** | **66.8%** | 65.1% | 逼近最优 | 59.5% |
| **平均等待** | **2.83 步** | 4.09 步 | 较短 | 5.44 步 |
| **最长等待** | **6.2 步** | 7.4 步 | 中 | 9.2 步 |
| **平均接驾距离** | **3.63 步** | 4.67 步 | 中 | 5.91 步 |

> **结论解读**：
> - **Enhanced-KM（自研改进）** 凭借前瞻指派 + 空闲车重定位，在需求热点场景下各项指标最优；
> - **标准匈牙利** 是强经典基线（当前快照全局最优）；
> - **LLM 多智能体** 逼近最优解，且额外提供**可解释推理 + 自然语言运营报告**——这是 LLM 的独特价值；
> - 四者均显著优于随机基线。
>
> *（具体数值随地图规模 / 车辆数 / 乘客频率 / 是否开启热点而变化，请运行 `compare.py` 获取当前配置统计。）*

---

## 🚀 快速开始

### 环境要求

- Python 3.9+
- 任意 OpenAI 兼容 LLM API（本项目使用 Xiaomi MiMo）

### 安装依赖

```bash
git clone https://github.com/lingfengai731/AgentSim_robotai.git
cd AgentSim_robotai
pip install -r requirements.txt
```

### 配置 API Key

复制 `.env.example` 为 `.env`，填入你的 API 凭证：

```ini
API_KEY  = your_api_key_here
BASE_URL = https://api.xiaomimimo.com/v1
MODEL    = mimo-v2.5-pro
```

### 验证 API 连通性

```bash
python test_api.py
```

---

## 🖥️ 使用方式

### 方式一：Web 界面（推荐）

```bash
streamlit run app.py
```

浏览器打开 `http://localhost:8501`，支持：
- 实时仿真地图动画
- 侧边栏参数调节（车辆数 / 步数 / 乘客频率）
- 一键对比实验（LLM vs 随机调度）

### 方式二：桌面 GUI（现代深色仪表盘 + A\* 绕障动画）

```bash
python gui.py
```

左上角下拉切换调度策略（LLM / 标准KM / 增强KM / 随机），地图实时渲染建筑障碍物、
需求热点、车辆 A\* 规划路径与行驶轨迹。

### 方式三：命令行仿真

```bash
python main.py
```

### 方式四：对比实验（多 seed）

```bash
python compare.py
```

输出 `outputs/comparison_three_way.png`（LLM / 匈牙利 / 随机 三方均值 ± 标准差柱状图）

---

## 📁 项目结构

```
AgentSim/
├── app.py                  # Streamlit Web 界面
├── gui.py                  # tkinter 桌面 GUI
├── main.py                 # 命令行仿真入口
├── compare.py              # 多 seed 对比实验
├── test_api.py             # API 连通性测试
├── config.py               # 全局配置（从 .env 读取）
├── requirements.txt
│
├── agents/                 # Agent 模块
│   ├── base_agent.py       # LLM 调用基类（含重试 + 兜底）
│   ├── coordinator.py      # Coordinator Agent（总协调）
│   ├── dispatcher.py       # Dispatcher Agent（订单分配）
│   ├── driver.py           # Driver Agent（移动 + 播报）
│   ├── analyst.py          # Analyst Agent（运营报告）
│   ├── hungarian_dispatcher.py    # 标准匈牙利调度器（KM 批量最优匹配 + scipy 加速）
│   ├── enhanced_km_dispatcher.py  # ⭐ Enhanced-KM：前瞻指派 + 空闲车重定位（自研改进）
│   └── random_dispatcher.py# 随机调度基线（对照实验用）
│
├── simulation/             # 仿真引擎
│   ├── environment.py      # 地图 + 车辆 + 乘客状态管理
│   ├── events.py           # 订单事件数据结构
│   ├── metrics.py          # KPI 统计（含平均接驾距离）
│   ├── pathfinding.py      # A* 栅格路径规划（绕障行驶 + 真实路网距离）
│   └── runner.py           # 通用仿真运行器（对比实验复用）
│
├── visualization/          # 可视化
│   ├── plotter.py          # 单次仿真图表
│   └── compare_plot.py     # 对比实验图表（含多 seed 误差棒）
│
└── .streamlit/
    └── secrets.toml        # 本地 Secrets（已 gitignore）
```

---

## ☁️ 部署到 Streamlit Cloud（公网访问）

1. 访问 [share.streamlit.io](https://share.streamlit.io) 并用 GitHub 登录
2. New app → 选择 `AgentSim_robotai` 仓库 → 入口文件 `app.py`
3. 进入 **Settings → Secrets**，填入：

```toml
API_KEY  = "your_api_key_here"
BASE_URL = "https://api.xiaomimimo.com/v1"
MODEL    = "mimo-v2.5-pro"
```

4. 点击 **Deploy** → 约 2 分钟后获得公网链接 🎉
https://agentsim-robotaxi.streamlit.app/
---

## 🔧 技术栈

| 层次 | 技术 |
|------|------|
| LLM 调用 | OpenAI SDK（兼容任意 OpenAI API 格式） |
| 多智能体框架 | 自研（无第三方 Agent 框架依赖） |
| 最优化算法 | 匈牙利算法 / Kuhn-Munkres（自研 O(n³) + SciPy `linear_sum_assignment` 加速） |
| 仿真引擎 | 纯 Python 离散事件仿真 |
| Web 界面 | Streamlit + 自定义 CSS（Figma 风格深色仪表盘） |
| 桌面 GUI | tkinter（多线程 + Queue） |
| 可视化 | Matplotlib |
| 数据统计 | NumPy / SciPy |

---

## 📌 简历表述参考

> **基于 LLM 多智能体的 Robotaxi 订单调度仿真系统**（个人开发，2026.04–至今）
>
> - 实现**匈牙利算法（Kuhn-Munkres）批量全局最优调度器**（自研 O(n³) + SciPy 加速），对标滴滴 / Uber 工业级 batch-matching 方案
> - 在其基础上提出自研改进 **Enhanced-KM**（前瞻指派 + 空闲车需求重定位），障碍物 + 需求热点场景下较标准 KM **平均等待 −31%、接驾空驶距离 −22%**；并通过**消融实验**数据驱动确定最优模块组合
> - 融合本人路径规划专长，引入**建筑障碍物地图 + A\* 栅格路径规划**：执行层车辆绕障行驶、调度层代价采用真实路网距离，形成"调度决策 + 路径执行"的分层自动驾驶架构
> - 设计 Coordinator / Dispatcher / Driver / Analyst 四类角色 Agent，基于 OpenAI 兼容接口调用大模型进行调度推理与运营分析报告生成
> - 构建 N×N 网格离散事件仿真环境（含障碍物与需求热点建模），实现乘客泊松到达、批量订单分配、A\* 绕障行驶完整闭环
> - 设计 LLM / 标准KM / 增强KM / 随机四方多 seed 对比实验（N=5），量化各策略差异
> - 基于 Streamlit + 自定义 CSS 开发 Figma 风格深色仪表盘，支持策略热切换、实时调度动画与四方对比，已部署至公网

---

## 📄 License

MIT License © 2026 吴凌峰
