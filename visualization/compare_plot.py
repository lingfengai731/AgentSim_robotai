"""
compare_plot.py — 对比实验可视化
生成 LLM 调度 vs 随机调度 的量化对比图（3 组指标 + 折线趋势）。
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

import config


def plot_comparison(llm_metrics: dict, rnd_metrics: dict,
                    save_path: str = None) -> str:
    """
    llm_metrics / rnd_metrics: run_once() 返回的 summary dict
    save_path: 保存路径，默认 outputs/comparison.png
    """
    if save_path is None:
        os.makedirs(config.OUTPUT_DIR, exist_ok=True)
        save_path = os.path.join(config.OUTPUT_DIR, "comparison.png")

    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(
        "AgentSim — LLM多智能体调度 vs 随机调度 对比实验",
        fontsize=15, fontweight="bold", y=0.98
    )

    # ── 上半部分：3 个指标柱状图 ────────────────────────────
    ax1 = fig.add_subplot(2, 3, 1)
    ax2 = fig.add_subplot(2, 3, 2)
    ax3 = fig.add_subplot(2, 3, 3)

    colors = {"LLM调度": "#2196F3", "随机调度": "#FF7043"}

    def bar_pair(ax, llm_val, rnd_val, title, ylabel, fmt=".1f", higher_better=True):
        bars = ax.bar(["LLM调度", "随机调度"],
                      [llm_val, rnd_val],
                      color=[colors["LLM调度"], colors["随机调度"]],
                      width=0.5, edgecolor="white", linewidth=1.5)
        ax.set_title(title, fontsize=12, pad=8)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.grid(axis="y", alpha=0.3)
        ax.set_ylim(0, max(llm_val, rnd_val) * 1.3 + 0.1)

        # 标数值
        for bar, val in zip(bars, [llm_val, rnd_val]):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + max(llm_val, rnd_val)*0.03,
                    f"{val:{fmt}}", ha="center", va="bottom",
                    fontsize=11, fontweight="bold")

        # 高亮胜者
        best_idx = 0 if (llm_val >= rnd_val) == higher_better else 1
        bars[best_idx].set_edgecolor("#FFD600")
        bars[best_idx].set_linewidth(3)

    total_llm = llm_metrics["completed_orders"] + llm_metrics["pending_orders"]
    total_rnd = rnd_metrics["completed_orders"] + rnd_metrics["pending_orders"]
    llm_rate  = llm_metrics["completed_orders"] / total_llm * 100 if total_llm else 0
    rnd_rate  = rnd_metrics["completed_orders"] / total_rnd * 100 if total_rnd else 0

    bar_pair(ax1, llm_rate, rnd_rate,
             "订单完成率 (%)", "%", fmt=".1f", higher_better=True)
    bar_pair(ax2, llm_metrics["avg_wait_steps"], rnd_metrics["avg_wait_steps"],
             "平均等待时间 (步)", "步", fmt=".1f", higher_better=False)
    bar_pair(ax3, llm_metrics["utilization"]*100, rnd_metrics["utilization"]*100,
             "车辆利用率 (%)", "%", fmt=".1f", higher_better=True)

    # ── 下半部分：订单完成趋势折线 ──────────────────────────
    ax4 = fig.add_subplot(2, 1, 2)
    steps_llm = range(len(llm_metrics["_completed_curve"]))
    steps_rnd = range(len(rnd_metrics["_completed_curve"]))

    ax4.plot(steps_llm, llm_metrics["_completed_curve"],
             color=colors["LLM调度"], linewidth=2.5, label="LLM调度 — 累计完成")
    ax4.plot(steps_rnd, rnd_metrics["_completed_curve"],
             color=colors["随机调度"], linewidth=2.5, label="随机调度 — 累计完成",
             linestyle="--")
    ax4.fill_between(steps_llm, llm_metrics["_completed_curve"],
                     rnd_metrics["_completed_curve"],
                     alpha=0.12, color=colors["LLM调度"],
                     label="LLM 优势区间")

    ax4.set_xlabel("仿真步数", fontsize=11)
    ax4.set_ylabel("累计完成订单数", fontsize=11)
    ax4.set_title("订单完成趋势对比", fontsize=12, pad=8)
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)

    # ── 右下角文字摘要 ──────────────────────────────────────
    delta_rate = llm_rate - rnd_rate
    delta_wait = rnd_metrics["avg_wait_steps"] - llm_metrics["avg_wait_steps"]
    sign_r = "+" if delta_rate >= 0 else ""
    sign_w = "+" if delta_wait >= 0 else ""
    summary_text = (
        f"LLM调度 vs 随机调度\n"
        f"完成率提升: {sign_r}{delta_rate:.1f}%\n"
        f"等待缩短:   {sign_w}{delta_wait:.1f} 步\n"
        f"利用率差:   {(llm_metrics['utilization']-rnd_metrics['utilization'])*100:+.1f}%"
    )
    fig.text(0.92, 0.28, summary_text,
             fontsize=11, va="top", ha="right",
             bbox=dict(boxstyle="round,pad=0.6", facecolor="#E3F2FD",
                       edgecolor="#1565C0", linewidth=1.5))

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(save_path, dpi=130, bbox_inches="tight")
    print(f"[对比图] 已保存至 {save_path}")
    return save_path


def plot_multi_seed(llm_stats: dict, rnd_stats: dict,
                    seeds: list, llm_all: list, rnd_all: list,
                    save_path: str = None) -> str:
    """
    多 seed 统计图：均值柱状图 + 误差棒（±1 std）+ 每个 seed 的散点
    """
    if save_path is None:
        os.makedirs(config.OUTPUT_DIR, exist_ok=True)
        save_path = os.path.join(config.OUTPUT_DIR, "comparison_multi.png")

    fig, axes = plt.subplots(1, 3, figsize=(15, 6))
    fig.suptitle(
        f"AgentSim — 多Seed统计对比（N={len(seeds)} seeds）\n"
        f"Seeds: {seeds}",
        fontsize=13, fontweight="bold"
    )

    colors = {"LLM调度": "#2196F3", "随机调度": "#FF7043"}
    labels = ["LLM调度", "随机调度"]

    metrics_cfg = [
        # (title, llm_mean, llm_std, rnd_mean, rnd_std, ylabel, higher_better, llm_vals, rnd_vals)
        (
            "订单完成率 (%)",
            llm_stats["rate_mean"], llm_stats["rate_std"],
            rnd_stats["rate_mean"], rnd_stats["rate_std"],
            "%", True,
            [r["completed_orders"]/(r["completed_orders"]+r["pending_orders"])*100
             for r in llm_all],
            [r["completed_orders"]/(r["completed_orders"]+r["pending_orders"])*100
             for r in rnd_all],
        ),
        (
            "平均等待时间 (步)",
            llm_stats["wait_mean"], llm_stats["wait_std"],
            rnd_stats["wait_mean"], rnd_stats["wait_std"],
            "步", False,
            [r["avg_wait_steps"] for r in llm_all],
            [r["avg_wait_steps"] for r in rnd_all],
        ),
        (
            "车辆利用率 (%)",
            llm_stats["util_mean"], llm_stats["util_std"],
            rnd_stats["util_mean"], rnd_stats["util_std"],
            "%", True,
            [r["utilization"]*100 for r in llm_all],
            [r["utilization"]*100 for r in rnd_all],
        ),
    ]

    for ax, (title, lm, ls, rm, rs, ylabel, hb, lvals, rvals) in zip(axes, metrics_cfg):
        means  = [lm, rm]
        stds   = [ls, rs]
        x      = np.arange(2)
        clrs   = [colors["LLM调度"], colors["随机调度"]]

        bars = ax.bar(x, means, yerr=stds, capsize=8, width=0.5,
                      color=clrs, alpha=0.85,
                      error_kw={"elinewidth": 2, "ecolor": "#333"})

        # 每个 seed 的散点（增加透明度区分）
        jitter = 0.08
        for xi, vals in zip([0, 1], [lvals, rvals]):
            xs = [xi + np.random.uniform(-jitter, jitter) for _ in vals]
            ax.scatter(xs, vals, color="black", s=30, zorder=5, alpha=0.6)

        # 数值标注
        for bar, mean, std in zip(bars, means, stds):
            ax.text(bar.get_x() + bar.get_width()/2,
                    mean + std + max(means)*0.04,
                    f"{mean:.1f}±{std:.1f}",
                    ha="center", va="bottom", fontsize=10, fontweight="bold")

        # 高亮胜者
        best_idx = 0 if (lm >= rm) == hb else 1
        bars[best_idx].set_edgecolor("#FFD600")
        bars[best_idx].set_linewidth(3)

        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=11)
        ax.set_title(title, fontsize=12, pad=8)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.grid(axis="y", alpha=0.3)
        ax.set_ylim(0, max(means) * 1.45 + max(stds))

    # 底部结论文字
    dr = llm_stats['rate_mean'] - rnd_stats['rate_mean']
    dw = rnd_stats['wait_mean'] - llm_stats['wait_mean']
    conclusion = (
        f"结论：LLM调度在订单完成率上平均{'领先' if dr>=0 else '落后'} {abs(dr):.1f}%，"
        f"等待时间{'缩短' if dw>=0 else '延长'} {abs(dw):.1f} 步（均值，N={len(seeds)}）"
    )
    fig.text(0.5, 0.01, conclusion, ha="center", fontsize=11,
             color="#1565C0", fontweight="bold")

    plt.tight_layout(rect=[0, 0.05, 1, 0.93])
    fig.savefig(save_path, dpi=130, bbox_inches="tight")
    print(f"[多seed图] 已保存至 {save_path}")
    return save_path
