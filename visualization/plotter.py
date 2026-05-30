"""
plotter.py — 仿真可视化
左图：实时地图（车辆 + 乘客位置）
右图：折线图（完成订单数 & 进行中订单数 随步数变化）
"""

import os
import matplotlib
matplotlib.use("Agg")          # 无 GUI 环境也能保存图片
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
plt.rcParams['font.sans-serif'] = ['SimHei']  # 或 ['Microsoft YaHei'] 微软雅黑 等
plt.rcParams['axes.unicode_minus'] = False   # 解决负号 '-' 显示为方块的问题
import config


# 车辆状态颜色
STATUS_COLOR = {
    "idle":       "green",
    "to_pickup":  "orange",
    "to_dropoff": "dodgerblue",
}


class SimPlotter:

    def __init__(self):
        os.makedirs(config.OUTPUT_DIR, exist_ok=True)
        self.fig, (self.ax_map, self.ax_kpi) = plt.subplots(
            1, 2, figsize=(14, 6)
        )
        self.fig.suptitle("AgentSim — Robotaxi 多智能体调度仿真", fontsize=14)

    def render_step(self, snapshot: dict,
                    completed_curve: list,
                    active_curve:    list):
        """渲染当前步的地图 + KPI 折线（覆盖上次）。"""
        self._draw_map(snapshot)
        self._draw_kpi(completed_curve, active_curve, snapshot["step"])
        plt.tight_layout()
        # 保存每步截图
        path = os.path.join(config.OUTPUT_DIR, f"step_{snapshot['step']:03d}.png")
        self.fig.savefig(path, dpi=80)

    def save_final(self, metrics: dict, report: str):
        """仿真结束，保存最终总结图。"""
        self.fig.suptitle(
            f"仿真结束 | 完成订单:{metrics['completed_orders']} "
            f"| 平均等待:{metrics['avg_wait_steps']:.1f}步 "
            f"| 利用率:{metrics['utilization']*100:.0f}%",
            fontsize=12
        )
        path = os.path.join(config.OUTPUT_DIR, "final_summary.png")
        self.fig.savefig(path, dpi=120)
        print(f"\n[可视化] 最终图表已保存至 {path}")

        # 把分析报告写成 txt
        report_path = os.path.join(config.OUTPUT_DIR, "analyst_report.txt")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("=== Analyst Agent 分析报告 ===\n\n")
            f.write(report)
        print(f"[可视化] 分析报告已保存至 {report_path}")

    # ── 内部绘图方法 ──────────────────────────────────────

    def _draw_map(self, snapshot: dict):
        ax = self.ax_map
        ax.cla()
        ax.set_xlim(-0.5, config.MAP_SIZE - 0.5)
        ax.set_ylim(-0.5, config.MAP_SIZE - 0.5)
        ax.set_xticks(range(config.MAP_SIZE))
        ax.set_yticks(range(config.MAP_SIZE))
        ax.grid(True, alpha=0.3)
        ax.set_title(f"Step {snapshot['step']}  |  "
                     f"等待:{snapshot['pending']}  进行中:{snapshot['active']}  "
                     f"完成:{snapshot['completed']}")
        ax.set_aspect("equal")

        # 画车辆
        for d in snapshot["drivers"]:
            color = STATUS_COLOR.get(d["status"], "gray")
            ax.plot(*d["pos"], marker="s", markersize=16,
                    color=color, alpha=0.85)
            ax.text(d["pos"][0], d["pos"][1],
                    str(d["id"]), ha="center", va="center",
                    fontsize=8, color="white", fontweight="bold")

        # 图例
        legend = [
            mpatches.Patch(color="green",      label="空闲"),
            mpatches.Patch(color="orange",     label="前往接客"),
            mpatches.Patch(color="dodgerblue", label="送客途中"),
        ]
        ax.legend(handles=legend, loc="upper right", fontsize=8)

    def _draw_kpi(self, completed: list, active: list, step: int):
        ax = self.ax_kpi
        ax.cla()
        steps = list(range(len(completed)))
        ax.plot(steps, completed, label="累计完成订单", color="green",  linewidth=2)
        ax.plot(steps, active,    label="进行中订单",   color="orange", linewidth=2,
                linestyle="--")
        ax.set_xlabel("仿真步数")
        ax.set_ylabel("订单数")
        ax.set_title("订单完成趋势")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
