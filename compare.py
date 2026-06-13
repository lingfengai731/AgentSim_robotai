"""
compare.py — 多策略对比实验 + 消融实验入口（多 seed 版）

主对比（四方）：
  · LLM 多智能体调度（Coordinator + Dispatcher，逐单 LLM 推理）
  · 标准匈牙利算法（Kuhn-Munkres 批量全局最优匹配）
  · 增强匈牙利 Enhanced-KM（自研改进：前瞻指派 + 空闲车需求重定位）
  · 随机调度（无策略基线）

消融实验（仅 KM 系）：逐个开关 Enhanced-KM 的三个模块，验证各自贡献。

运行方式：python compare.py
"""

import numpy as np
import config
from agents.coordinator          import CoordinatorAgent
from agents.hungarian_dispatcher import HungarianDispatcher
from agents.enhanced_km_dispatcher import EnhancedKMDispatcher
from agents.random_dispatcher     import RandomDispatcher
from simulation.runner            import run_once
from visualization.compare_plot   import plot_strategies


SEEDS = [42, 123, 456, 789, 1024]


def _rate(m):
    t = m["completed_orders"] + m["pending_orders"]
    return m["completed_orders"] / t * 100 if t else 0.0


def _aggregate(results):
    rates = [_rate(r)               for r in results]
    waits = [r["avg_wait_steps"]    for r in results]
    maxws = [r["max_wait_steps"]    for r in results]
    picks = [r["avg_pickup_dist"]   for r in results]
    utils = [r["utilization"] * 100 for r in results]
    return {
        "rate_mean": np.mean(rates), "rate_std": np.std(rates), "rates": rates,
        "wait_mean": np.mean(waits), "wait_std": np.std(waits), "waits": waits,
        "maxw_mean": np.mean(maxws), "maxw_std": np.std(maxws), "maxws": maxws,
        "pick_mean": np.mean(picks), "pick_std": np.std(picks), "picks": picks,
        "util_mean": np.mean(utils), "util_std": np.std(utils), "utils": utils,
    }


def _run_all(strategies):
    """strategies: [(name, factory)] → {name: agg}"""
    out = {}
    for name, factory in strategies:
        runs = []
        for seed in SEEDS:
            runs.append(run_once(factory(), seed=seed))
        out[name] = _aggregate(runs)
        a = out[name]
        print(f"  {name:<12} 完成率 {a['rate_mean']:5.1f}%  "
              f"平均等待 {a['wait_mean']:.2f}  最长等待 {a['maxw_mean']:.1f}  "
              f"接驾距离 {a['pick_mean']:.2f}")
    return out


def main():
    print("=" * 70)
    print("  AgentSim — 多策略对比 + 消融实验")
    print(f"  Seeds:{SEEDS}  地图:{config.MAP_SIZE}²  车辆:{config.NUM_DRIVERS}  "
          f"步数:{config.SIM_STEPS}  需求热点:{getattr(config,'USE_HOTSPOTS',False)}")
    print("=" * 70)

    # ── 主对比：四方 ─────────────────────────────────────
    print("\n[1] 四方主对比：LLM / 标准KM / 增强KM / 随机")
    main_strats = [
        ("LLM调度",   lambda: CoordinatorAgent(num_drivers=config.NUM_DRIVERS)),
        ("标准KM",    lambda: HungarianDispatcher()),
        ("增强KM",    lambda: EnhancedKMDispatcher()),
        ("随机",      lambda: RandomDispatcher()),
    ]
    main_stats = _run_all(main_strats)

    colors = {"LLM调度": "#89B4FA", "标准KM": "#A6E3A1",
              "增强KM": "#F9E2AF", "随机": "#F38BA8"}
    plot_strategies(main_stats, ["LLM调度", "标准KM", "增强KM", "随机"],
                    colors, SEEDS,
                    save_path="outputs/comparison_four_way.png",
                    title=f"四方调度对比（N={len(SEEDS)} seeds）"
                          f" · LLM vs 标准KM vs 增强KM vs 随机")

    # ── 消融实验：Enhanced-KM 三模块 ─────────────────────
    print("\n[2] 消融实验：标准KM + 逐模块开关（验证各模块贡献）")
    abl_strats = [
        ("标准KM",       lambda: HungarianDispatcher()),
        ("+前瞻",        lambda: EnhancedKMDispatcher(use_aging=False, use_lookahead=True,  use_rebalance=False)),
        ("+重定位",      lambda: EnhancedKMDispatcher(use_aging=False, use_lookahead=False, use_rebalance=True)),
        ("+前瞻+重定位", lambda: EnhancedKMDispatcher(use_aging=False, use_lookahead=True,  use_rebalance=True)),
        ("+全部(含aging)", lambda: EnhancedKMDispatcher(use_aging=True, use_lookahead=True, use_rebalance=True)),
    ]
    abl_stats = _run_all(abl_strats)
    abl_colors = {n: c for n, c in zip(
        [s[0] for s in abl_strats],
        ["#A6E3A1", "#94E2D5", "#89DCEB", "#F9E2AF", "#FAB387"])}
    plot_strategies(abl_stats, [s[0] for s in abl_strats], abl_colors, SEEDS,
                    save_path="outputs/ablation_enhanced_km.png",
                    title=f"Enhanced-KM 消融实验（N={len(SEEDS)} seeds）")

    # ── 关键结论 ─────────────────────────────────────────
    km, ek = main_stats["标准KM"], main_stats["增强KM"]
    print("\n" + "=" * 70)
    print("  关键结论（增强KM vs 标准KM）：")
    print(f"    订单完成率   {km['rate_mean']:.1f}% → {ek['rate_mean']:.1f}%  "
          f"({ek['rate_mean']-km['rate_mean']:+.1f}%)")
    print(f"    平均等待     {km['wait_mean']:.2f} → {ek['wait_mean']:.2f} 步  "
          f"({ek['wait_mean']-km['wait_mean']:+.2f})")
    print(f"    平均接驾距离 {km['pick_mean']:.2f} → {ek['pick_mean']:.2f} 步  "
          f"({ek['pick_mean']-km['pick_mean']:+.2f})")
    print("\n✅ 输出：outputs/comparison_four_way.png · outputs/ablation_enhanced_km.png")
    print("=" * 70)


if __name__ == "__main__":
    main()
