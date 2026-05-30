"""
app.py — AgentSim Streamlit Web 版
运行方式：streamlit run app.py
"""

import os, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Streamlit Cloud 兼容：把 secrets 注入为环境变量 ──────
# 本地用 .env，云上用 Streamlit Secrets，config.py 无需改动
try:
    import streamlit as _st
    for _k, _v in _st.secrets.items():
        os.environ.setdefault(_k, str(_v))
except Exception:
    pass  # 本地无 secrets.toml 时忽略
import matplotlib.patches as mpatches
import streamlit as st
from dotenv import load_dotenv

load_dotenv()
import config

# ── 页面配置（必须第一行）───────────────────────────────
st.set_page_config(
    page_title="AgentSim — Robotaxi 多智能体调度",
    page_icon="🚖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── 全局字体（matplotlib 中文）──────────────────────────
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# ══════════════════════════════════════════════════════
#  辅助：绘制地图 → matplotlib Figure
# ══════════════════════════════════════════════════════

STATUS_COLOR = {
    "idle":       "#A6E3A1",
    "to_pickup":  "#FAB387",
    "to_dropoff": "#89DCEB",
}

def draw_map(snap: dict, map_size: int) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(5, 5))
    fig.patch.set_facecolor("#1E1E2E")
    ax.set_facecolor("#1E1E2E")

    # 网格
    for i in range(map_size + 1):
        ax.axhline(i, color="#2E2E3E", linewidth=0.8)
        ax.axvline(i, color="#2E2E3E", linewidth=0.8)

    # 车辆
    for d in snap["drivers"]:
        x, y = d["pos"][0] + 0.5, d["pos"][1] + 0.5
        color = STATUS_COLOR.get(d["status"], "#6C7086")
        circle = plt.Circle((x, y), 0.35, color=color, zorder=3)
        ax.add_patch(circle)
        ax.text(x, y, str(d["id"]), ha="center", va="center",
                fontsize=11, fontweight="bold", color="#1E1E2E", zorder=4)

    ax.set_xlim(0, map_size)
    ax.set_ylim(0, map_size)
    ax.set_xticks(range(map_size + 1))
    ax.set_yticks(range(map_size + 1))
    ax.tick_params(colors="#6C7086", labelsize=7)
    for spine in ax.spines.values():
        spine.set_edgecolor("#2E2E3E")

    # 图例
    legend = [
        mpatches.Patch(color="#A6E3A1", label="空闲"),
        mpatches.Patch(color="#FAB387", label="前往接客"),
        mpatches.Patch(color="#89DCEB", label="送客中"),
    ]
    ax.legend(handles=legend, loc="upper right", fontsize=7,
              facecolor="#28283A", labelcolor="#CDD6F4", edgecolor="#45475A")

    fig.tight_layout(pad=0.3)
    return fig


def draw_curve(completed: list, active: list) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(5, 2.5))
    fig.patch.set_facecolor("#1E1E2E")
    ax.set_facecolor("#1E1E2E")
    steps = list(range(len(completed)))
    ax.plot(steps, completed, color="#A6E3A1", linewidth=2, label="累计完成")
    ax.plot(steps, active,    color="#FAB387", linewidth=2,
            linestyle="--", label="进行中")
    ax.set_xlabel("步数", color="#6C7086", fontsize=8)
    ax.set_ylabel("订单数", color="#6C7086", fontsize=8)
    ax.tick_params(colors="#6C7086", labelsize=7)
    ax.legend(fontsize=7, facecolor="#28283A", labelcolor="#CDD6F4",
              edgecolor="#45475A")
    ax.grid(True, alpha=0.2, color="#6C7086")
    for spine in ax.spines.values():
        spine.set_edgecolor("#2E2E3E")
    fig.tight_layout(pad=0.3)
    return fig


# ══════════════════════════════════════════════════════
#  仿真驱动（带 yield，逐步推送快照）
# ══════════════════════════════════════════════════════

def run_simulation_stream(dispatcher, seed, map_size, num_drivers,
                           sim_steps, passenger_rate):
    """Generator：每步 yield (snap, log_line, metrics_curve)"""
    import random
    from simulation.environment import SimEnvironment
    from simulation.metrics     import MetricsTracker
    from agents.driver import DriverAgent as _DA

    # 临时覆盖 config（Streamlit 滑块值）
    config.MAP_SIZE       = map_size
    config.NUM_DRIVERS    = num_drivers
    config.SIM_STEPS      = sim_steps
    config.PASSENGER_RATE = passenger_rate

    env     = SimEnvironment(seed=seed)
    metrics = MetricsTracker()

    completed_curve, active_curve = [], []

    for step in range(sim_steps):
        logs = []

        req = env.maybe_generate_passenger()
        if req:
            logs.append(f"🙋 P{req.passenger_id} {req.pickup}→{req.dropoff}")
            result = dispatcher.assign_order(
                passenger={"id": req.passenger_id,
                           "pickup": req.pickup,
                           "dropoff": req.dropoff},
                drivers=env.drivers,
            )
            did = result["assigned_driver_id"]
            if did is not None:
                env.assign(req.passenger_id, did)
                tag = "LLM" if "兜底" not in result["reason"] else "兜底"
                logs.append(f"  📋 [{tag}] P{req.passenger_id} → 车辆{did}")
            else:
                logs.append("  ⚠ 无空闲车辆")

        for driver in env.drivers:
            if driver["status"] == "idle":
                continue
            target = env.get_driver_target(driver["id"])
            if target is None:
                continue

            event = None
            if driver["status"] == "to_pickup" and driver["pos"] == target:
                event = _DA.EVENT_ARRIVED_PICKUP
            elif driver["status"] == "to_dropoff" and driver["pos"] == target:
                event = _DA.EVENT_ARRIVED_DROPOFF

            move = dispatcher.move_driver(
                driver=driver, target=target,
                map_size=config.MAP_SIZE, event=event,
            )
            env.move_driver(driver["id"], move["next_pos"])

            if event == _DA.EVENT_ARRIVED_PICKUP:
                logs.append(f"  🚗 车辆{driver['id']} 接到乘客")
            elif event == _DA.EVENT_ARRIVED_DROPOFF:
                logs.append(f"  ✅ 车辆{driver['id']} 送达，订单完成")

        env.advance_step()
        snap = env.snapshot()
        metrics.record(snap)
        completed_curve.append(snap["completed"])
        active_curve.append(snap["active"])

        yield snap, logs, completed_curve[:], active_curve[:]

    # 最终 KPI
    kpi = metrics.summary(
        completed=env.completed_requests,
        pending=env.pending_requests + env.active_requests,
        total_steps=sim_steps,
        num_drivers=num_drivers,
    )
    kpi["_completed_curve"] = completed_curve
    kpi["_active_curve"]    = active_curve
    yield None, ["── 仿真结束 ──"], completed_curve, active_curve, kpi


# ══════════════════════════════════════════════════════
#  Streamlit 页面
# ══════════════════════════════════════════════════════

# ── 侧边栏 ────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ 仿真参数")
    map_size      = st.slider("地图大小",       6,  15, config.MAP_SIZE)
    num_drivers   = st.slider("车辆数量",        2,   8, config.NUM_DRIVERS)
    sim_steps     = st.slider("仿真步数",       10,  60, config.SIM_STEPS)
    passenger_rate = st.slider("乘客频率",     0.1, 0.6, config.PASSENGER_RATE, 0.05)
    seed          = st.number_input("随机种子", value=42, step=1)
    speed         = st.select_slider("动画速度",
                                      options=["极慢(2s)", "慢(1s)", "中(0.5s)",
                                               "快(0.2s)", "极快(0s)"],
                                      value="中(0.5s)")
    SPEED_MAP = {"极慢(2s)": 2.0, "慢(1s)": 1.0, "中(0.5s)": 0.5,
                 "快(0.2s)": 0.2, "极快(0s)": 0.0}
    step_delay = SPEED_MAP[speed]

    st.divider()
    st.markdown("## 🔬 对比实验")
    run_compare = st.checkbox("同步运行随机调度基线", value=False,
                               help="勾选后将同时运行 LLM 调度和随机调度，\n仿真结束后显示对比图")
    multi_seed  = st.checkbox("多Seed重复实验（5组）", value=False,
                               help="运行5个不同随机种子，统计均值±标准差")

    st.divider()
    st.caption(f"Model: `{config.MODEL}`")
    st.caption(f"Base URL: `{config.BASE_URL[:30]}...`")

# ── 主页面标题 ────────────────────────────────────────
st.markdown(
    "<h1 style='color:#89B4FA;'>🚖 AgentSim — Robotaxi 多智能体调度仿真</h1>",
    unsafe_allow_html=True
)
st.caption("基于 LLM 多智能体的 Robotaxi 订单调度仿真系统 | "
           "Coordinator · Dispatcher · Driver · Analyst")

st.divider()

# ── 开始按钮 ──────────────────────────────────────────
col_btn1, col_btn2, _ = st.columns([1, 1, 4])
start_btn   = col_btn1.button("▶ 开始仿真", type="primary", use_container_width=True)
compare_btn = col_btn2.button("📊 仅对比实验", use_container_width=True)

# ── 实时仿真区域 ──────────────────────────────────────
if start_btn:
    from agents.coordinator import CoordinatorAgent

    st.markdown("### 🗺️ 实时仿真")
    col_map, col_right = st.columns([1.1, 1])

    with col_map:
        map_ph   = st.empty()
        curve_ph = st.empty()

    with col_right:
        st.markdown("**📊 KPI 面板**")
        kpi_ph = st.empty()
        st.markdown("**🤖 Agent 决策日志**")
        log_ph  = st.empty()

    all_logs = []
    dispatcher = CoordinatorAgent(num_drivers=num_drivers)
    gen = run_simulation_stream(dispatcher, int(seed), map_size, num_drivers,
                                 sim_steps, passenger_rate)

    final_kpi = None
    for result in gen:
        if len(result) == 5:          # 最后一步返回 kpi
            _, logs, cc, ac, final_kpi = result
            all_logs.extend(logs)
            break

        snap, logs, cc, ac = result
        all_logs.extend(logs)

        # 更新地图
        with map_ph.container():
            fig = draw_map(snap, map_size)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

        # 更新曲线
        with curve_ph.container():
            fig2 = draw_curve(cc, ac)
            st.pyplot(fig2, use_container_width=True)
            plt.close(fig2)

        # 更新 KPI
        total = snap["completed"] + snap["active"] + snap["pending"]
        with kpi_ph.container():
            k1, k2 = st.columns(2)
            k1.metric("步数",   f"{snap['step']} / {sim_steps}")
            k2.metric("✅ 完成", snap["completed"])
            k3, k4 = st.columns(2)
            k3.metric("🚗 进行中", snap["active"])
            k4.metric("⏳ 等待",   snap["pending"])

        # 更新日志（最近20条）
        with log_ph.container():
            st.code("\n".join(all_logs[-20:]), language=None)

        time.sleep(step_delay)

    # ── 仿真结束：最终报告 ────────────────────────────
    if final_kpi:
        total = final_kpi["completed_orders"] + final_kpi["pending_orders"]
        rate  = final_kpi["completed_orders"] / total * 100 if total else 0

        st.success(f"✅ 仿真完成！完成率 **{rate:.1f}%** | "
                   f"平均等待 **{final_kpi['avg_wait_steps']:.1f} 步** | "
                   f"利用率 **{final_kpi['utilization']*100:.1f}%**")

        # Analyst 报告
        with st.spinner("📝 Analyst Agent 生成分析报告…"):
            from agents import AnalystAgent
            analyst = AnalystAgent()
            report  = analyst.run({"metrics": final_kpi})["report"]

        with st.expander("📋 Analyst Agent 运营分析报告", expanded=True):
            st.markdown(f"> {report}")

        # 如果勾选了对比实验
        if run_compare:
            st.markdown("---")
            st.markdown("### 📊 LLM 调度 vs 随机调度 对比")
            with st.spinner("运行随机调度基线…"):
                from agents.random_dispatcher import RandomDispatcher
                from simulation.runner import run_once
                rnd   = RandomDispatcher()
                r_rnd = run_once(rnd, seed=int(seed))
                total_r = r_rnd["completed_orders"] + r_rnd["pending_orders"]
                rate_r  = r_rnd["completed_orders"] / total_r * 100 if total_r else 0

            c1, c2, c3 = st.columns(3)
            c1.metric("完成率", f"{rate:.1f}%",
                       f"{rate - rate_r:+.1f}% vs 随机")
            c2.metric("平均等待", f"{final_kpi['avg_wait_steps']:.1f} 步",
                       f"{r_rnd['avg_wait_steps'] - final_kpi['avg_wait_steps']:+.1f} 步")
            c3.metric("利用率", f"{final_kpi['utilization']*100:.1f}%",
                       f"{(final_kpi['utilization'] - r_rnd['utilization'])*100:+.1f}%")

# ── 仅对比实验模式 ────────────────────────────────────
elif compare_btn:
    from agents.coordinator       import CoordinatorAgent
    from agents.random_dispatcher import RandomDispatcher
    from simulation.runner        import run_once

    seeds_to_run = [42, 123, 456, 789, 1024] if multi_seed else [int(seed)]
    n = len(seeds_to_run)

    st.markdown(f"### 📊 对比实验（{'多Seed × ' + str(n) if multi_seed else 'Seed=' + str(seed)}）")
    prog = st.progress(0, text="准备中…")

    llm_all, rnd_all = [], []
    llm_disp = CoordinatorAgent(num_drivers=num_drivers)
    rnd_disp = RandomDispatcher()

    for i, s in enumerate(seeds_to_run):
        prog.progress((2*i)   / (2*n), text=f"Seed {s} — LLM 调度 ({i+1}/{n})")
        config.MAP_SIZE = map_size
        config.NUM_DRIVERS = num_drivers
        config.SIM_STEPS = sim_steps
        config.PASSENGER_RATE = passenger_rate
        llm_all.append(run_once(llm_disp, seed=s))

        prog.progress((2*i+1) / (2*n), text=f"Seed {s} — 随机调度 ({i+1}/{n})")
        rnd_all.append(run_once(rnd_disp, seed=s))

    prog.progress(1.0, text="计算中…")

    def _rate(m):
        t = m["completed_orders"] + m["pending_orders"]
        return m["completed_orders"] / t * 100 if t else 0

    llm_rates = [_rate(m) for m in llm_all]
    rnd_rates = [_rate(m) for m in rnd_all]
    llm_waits = [m["avg_wait_steps"]    for m in llm_all]
    rnd_waits = [m["avg_wait_steps"]    for m in rnd_all]
    llm_utils = [m["utilization"] * 100 for m in llm_all]
    rnd_utils = [m["utilization"] * 100 for m in rnd_all]

    prog.empty()

    # 指标卡
    c1, c2, c3 = st.columns(3)
    dr = np.mean(llm_rates) - np.mean(rnd_rates)
    dw = np.mean(rnd_waits) - np.mean(llm_waits)
    c1.metric("完成率 LLM vs 随机",
               f"{np.mean(llm_rates):.1f}% vs {np.mean(rnd_rates):.1f}%",
               f"LLM 领先 {dr:+.1f}%")
    c2.metric("平均等待 LLM vs 随机",
               f"{np.mean(llm_waits):.1f}步 vs {np.mean(rnd_waits):.1f}步",
               f"LLM 短 {dw:+.1f} 步")
    c3.metric("利用率 LLM vs 随机",
               f"{np.mean(llm_utils):.1f}% vs {np.mean(rnd_utils):.1f}%",
               f"{np.mean(llm_utils)-np.mean(rnd_utils):+.1f}%")

    # 对比柱状图
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    fig.patch.set_facecolor("#1E1E2E")
    titles  = ["订单完成率 (%)", "平均等待 (步)", "车辆利用率 (%)"]
    llm_vs  = [np.mean(llm_rates), np.mean(llm_waits), np.mean(llm_utils)]
    rnd_vs  = [np.mean(rnd_rates), np.mean(rnd_waits), np.mean(rnd_utils)]
    llm_std = [np.std(llm_rates),  np.std(llm_waits),  np.std(llm_utils)]
    rnd_std = [np.std(rnd_rates),  np.std(rnd_waits),  np.std(rnd_utils)]

    for ax, title, lv, rv, ls, rs in zip(axes, titles, llm_vs, rnd_vs, llm_std, rnd_std):
        ax.set_facecolor("#1E1E2E")
        bars = ax.bar(["LLM调度", "随机调度"], [lv, rv],
                       yerr=[ls, rs], capsize=6,
                       color=["#89B4FA", "#F38BA8"], alpha=0.85,
                       error_kw={"ecolor": "#CDD6F4", "elinewidth": 1.5})
        ax.set_title(title, color="#CDD6F4", fontsize=10)
        ax.tick_params(colors="#6C7086", labelsize=8)
        ax.set_facecolor("#1E1E2E")
        for spine in ax.spines.values():
            spine.set_edgecolor("#2E2E3E")
        ax.grid(axis="y", alpha=0.2, color="#6C7086")
        for bar, val in zip(bars, [lv, rv]):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + max(lv, rv) * 0.05,
                    f"{val:.1f}", ha="center", color="#CDD6F4",
                    fontsize=9, fontweight="bold")

    fig.suptitle(f"LLM 调度 vs 随机调度（N={n} seeds）",
                  color="#89B4FA", fontsize=13, fontweight="bold")
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

# ── 未点击时显示介绍 ──────────────────────────────────
else:
    st.markdown("""
    ### 🏗️ 系统架构

    ```
    用户请求（仿真参数）
         ↓
    ┌─────────────────────────────────────┐
    │       Coordinator Agent             │  总协调，管理全局状态
    └──────┬──────────────────────────────┘
           ├──→  Dispatcher Agent  ──→ LLM 决策：把订单分配给最近车辆
           ├──→  Driver Agent × N  ──→ LLM 播报：接单/到达/送达事件
           └──→  Analyst Agent     ──→ LLM 报告：仿真结束后生成运营分析
    ```

    ### 🚀 快速开始
    1. 在左侧调整参数（车辆数、步数、乘客频率）
    2. 点击 **▶ 开始仿真** 查看实时动画
    3. 勾选"同步运行随机调度基线"可在仿真结束后自动对比
    4. 点击 **📊 仅对比实验** 直接运行多 Seed 统计对比

    ### 📊 核心指标
    | 指标 | 说明 |
    |------|------|
    | 订单完成率 | 在仿真步数内成功送达的订单占比 |
    | 平均等待时间 | 从乘客请求到上车的平均步数 |
    | 车辆利用率 | 车辆处于"执行任务"状态的时间比例 |
    """)
