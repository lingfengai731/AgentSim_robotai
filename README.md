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
| 📊 **量化对比实验** | 5 seed 重复实验，LLM 调度 vs 随机调度，均值 ± 标准差统计 |
| 🌐 **Web 交互界面** | Streamlit 实时仿真地图 + KPI 面板 + 决策日志，可一键部署到公网 |
| 🖥️ **桌面 GUI** | tkinter 多线程动态动画，支持暂停 / 继续 / 重置 + 速度调节 |
| 📈 **自动报告生成** | Analyst Agent 仿真结束后自动输出中文运营分析报告 |

---

## 🏗️ 系统架构

```
乘客请求（随机生成）
      ↓
┌─────────────────────────────────────────┐
│          Coordinator Agent              │  ← 总协调，管理全局状态与冲突仲裁
└──────┬──────────────────────────────────┘
       ├──→  Dispatcher Agent  ──→  LLM 推理：选择距离最优的空闲车辆
       ├──→  Driver Agent × N  ──→  LLM 播报：接单 / 到达 / 送达事件
       └──→  Analyst Agent     ──→  LLM 报告：KPI 分析 + 改进建议
                  ↓
          仿真环境（10×10 网格地图）
          离散事件驱动：乘客生成 → 订单分配 → 车辆移动 → 状态更新
                  ↓
     可视化层（Web / 桌面 / 图表文件）
```

---

## 📊 实验结果（5 seed 均值 ± 标准差）

| 指标 | LLM 调度 | 随机调度 | 提升 |
|------|---------|---------|------|
| **订单完成率** | **75.0% ± 18.3%** | 57.1% ± 4.6% | **+17.9%** |
| **平均等待时间** | **3.8 ± 0.5 步** | 5.4 ± 1.6 步 | **-1.6 步** |
| 车辆利用率 | 50.1% ± 12.1% | 57.2% ± 10.2% | -7.1% |

> 利用率降低是 **效率提升的体现**：LLM 优先派遣距离最近的车辆，
> 每单行程更短，车辆更快恢复空闲，从而承接更多订单，完成率提升 17.9%。

---

## 🚀 快速开始

### 环境要求

- Python 3.9+
- 任意 OpenAI 兼容 LLM API（本项目使用 Xiaomi MiMo）

### 安装依赖

```bash
git clone https://github.com/YOUR_USERNAME/AgentSim.git
cd AgentSim
pip install -r requirements.txt
```

### 配置 API Key

复制 `.env.example` 为 `.env`，填入你的 API 凭证：

```bash
cp .env.example .env
```

```ini
# .env
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

### 方式二：桌面 GUI

```bash
python gui.py
```

### 方式三：命令行仿真

```bash
python main.py
```

### 方式四：对比实验（多 seed）

```bash
python compare.py
```

输出 `outputs/comparison_multi.png`（均值 ± 标准差柱状图）

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
│   └── random_dispatcher.py# 随机调度基线（对照实验用）
│
├── simulation/             # 仿真引擎
│   ├── environment.py      # 地图 + 车辆 + 乘客状态管理
│   ├── events.py           # 订单事件数据结构
│   ├── metrics.py          # KPI 统计
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

1. 将代码推送到 GitHub（见下方）
2. 访问 [share.streamlit.io](https://share.streamlit.io) 并用 GitHub 登录
3. New app → 选择 `AgentSim` 仓库 → 入口文件 `app.py`
4. 进入 **Settings → Secrets**，填入：

```toml
API_KEY  = "your_api_key_here"
BASE_URL = "https://api.xiaomimimo.com/v1"
MODEL    = "mimo-v2.5-pro"
```

5. 点击 **Deploy** → 约 2 分钟后获得公网链接 🎉

---

## 🔧 技术栈

| 层次 | 技术 |
|------|------|
| LLM 调用 | OpenAI SDK（兼容任意 OpenAI API 格式） |
| 多智能体框架 | 自研（无第三方 Agent 框架依赖） |
| 仿真引擎 | 纯 Python 离散事件仿真 |
| Web 界面 | Streamlit |
| 桌面 GUI | tkinter（多线程 + Queue） |
| 可视化 | Matplotlib |
| 数据统计 | NumPy |

---

## 📌 简历表述参考

> **基于 LLM 多智能体的 Robotaxi 订单调度仿真系统**（个人开发，2026.04–至今）
>
> - 设计 Coordinator / Dispatcher / Driver / Analyst 四类角色 Agent，基于 OpenAI 兼容接口调用大模型进行订单分配推理与运营分析报告生成
> - 构建 10×10 网格离散事件仿真环境，实现乘客生成、订单分配、动态路径规划完整闭环
> - 设计多 seed 对比实验（N=5），LLM 调度在订单完成率上平均领先随机基线 **17.9%**，等待时间缩短 **1.6 步**
> - 基于 Streamlit 开发可交互 Web 界面，支持参数实时调节与仿真动画可视化，已部署至公网

---

## 📄 License

MIT License © 2026 吴凌峰
