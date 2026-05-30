"""
compare.py — 对比实验入口（多 seed 版）
用 5 个不同随机种子重复实验，统计均值±标准差，
结论才具有统计可靠性。
运行方式：python compare.py
"""

import numpy as np
import config
from agents.coordinator       import CoordinatorAgent
from agents.random_dispatcher import RandomDispatcher
from simulation.runner        import run_once
from visualization.compare_plot import plot_comparison, plot_multi_seed


# ── 实验配置 ──────────────────────────────────────────────
SEEDS = [42, 123, 456, 789, 1024]   # 5 个不同随机种子


def _completion_rate(m: dict) -> float:
    total = m["completed_orders"] + m["pending_orders"]
    return m["completed_orders"] / total * 100 if total else 0.0


def _aggregate(results: list) -> dict:
    """把多次运行结果聚合成均值±标准差。"""
    rates = [_completion_rate(r) for r in results]
    waits = [r["avg_wait_steps"]    for r in results]
    utils = [r["utilization"] * 100 for r in results]
    return {
        "rate_mean": float(np.mean(rates)), "rate_std": float(np.std(rates)),
        "wait_mean": float(np.mean(waits)), "wait_std": float(np.std(waits)),
        "util_mean": float(np.mean(utils)), "util_std": float(np.std(utils)),
        "_all_results": results,   # 保留原始数据供折线图使用
    }


def main():
    print("=" * 62)
    print("  AgentSim — 多seed对比实验：LLM调度 vs 随机调度")
    print("=" * 62)
    print(f"  Seeds: {SEEDS}")
    print(f"  地图:{config.MAP_SIZE}×{config.MAP_SIZE}  "
          f"车辆:{config.NUM_DRIVERS}  步数:{config.SIM_STEPS}\n")

    llm_dispatcher = CoordinatorAgent(num_drivers=config.NUM_DRIVERS)
    rnd_dispatcher = RandomDispatcher()

    llm_all, rnd_all = [], []

    for i, seed in enumerate(SEEDS):
        print(f"── Seed {seed} ({i+1}/{len(SEEDS)}) " + "─" * 35)

        # LLM 调度
        print(f"  [LLM]  运行中...", end="", flush=True)
        m_llm = run_once(llm_dispatcher, seed=seed, verbose=False)
        llm_all.append(m_llm)
        print(f"  完成率:{_completion_rate(m_llm):.1f}%  "
              f"等待:{m_llm['avg_wait_steps']:.1f}步  "
              f"利用率:{m_llm['utilization']*100:.1f}%")

        # 随机调度
        print(f"  [随机] 运行中...", end="", flush=True)
        m_rnd = run_once(rnd_dispatcher, seed=seed, verbose=False)
        rnd_all.append(m_rnd)
        print(f"  完成率:{_completion_rate(m_rnd):.1f}%  "
              f"等待:{m_rnd['avg_wait_steps']:.1f}步  "
              f"利用率:{m_rnd['utilization']*100:.1f}%")

    # ── 聚合统计 ─────────────────────────────────────────
    llm_stats = _aggregate(llm_all)
    rnd_stats = _aggregate(rnd_all)

    print("\n" + "=" * 62)
    print("  多seed统计结果（均值 ± 标准差）：")
    print(f"  {'指标':<12} {'LLM调度':>18} {'随机调度':>18} {'差值':>12}")
    print("  " + "─" * 58)

    dr = llm_stats['rate_mean'] - rnd_stats['rate_mean']
    dw = rnd_stats['wait_mean'] - llm_stats['wait_mean']
    du = llm_stats['util_mean'] - rnd_stats['util_mean']

    print(f"  {'订单完成率':<10} "
          f"{llm_stats['rate_mean']:>6.1f}%±{llm_stats['rate_std']:.1f}%  "
          f"{rnd_stats['rate_mean']:>6.1f}%±{rnd_stats['rate_std']:.1f}%  "
          f"LLM {'领先' if dr>=0 else '落后'}{abs(dr):.1f}%")
    print(f"  {'平均等待':<10} "
          f"{llm_stats['wait_mean']:>6.1f}步±{llm_stats['wait_std']:.1f}   "
          f"{rnd_stats['wait_mean']:>6.1f}步±{rnd_stats['wait_std']:.1f}   "
          f"LLM {'更短' if dw>=0 else '更长'}{abs(dw):.1f}步")
    print(f"  {'车辆利用率':<10} "
          f"{llm_stats['util_mean']:>6.1f}%±{llm_stats['util_std']:.1f}%  "
          f"{rnd_stats['util_mean']:>6.1f}%±{rnd_stats['util_std']:.1f}%  "
          f"{du:+.1f}%")

    # ── 生成图表 ─────────────────────────────────────────
    print("\n▶ 生成对比图表（单seed + 多seed误差棒）...")

    # 图1：最后一个 seed 的单次对比（保留原图）
    plot_comparison(llm_all[-1], rnd_all[-1],
                    save_path="outputs/comparison_single.png")

    # 图2：多 seed 均值±std 误差棒图（新增）
    plot_multi_seed(llm_stats, rnd_stats,
                    seeds=SEEDS,
                    llm_all=llm_all,
                    rnd_all=rnd_all)

    print("\n✅ 完成！输出文件：")
    print("   outputs/comparison_single.png  ← 单次对比（可视化效果好）")
    print("   outputs/comparison_multi.png   ← 多seed统计（科学性强）")
    print("=" * 62)


if __name__ == "__main__":
    main()
