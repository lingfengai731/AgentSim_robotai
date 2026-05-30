"""
main.py — AgentSim 仿真入口
运行方式：python main.py
"""

import config
from agents      import CoordinatorAgent, AnalystAgent
from simulation  import SimEnvironment, MetricsTracker

from visualization import SimPlotter


def run_simulation():
    print("=" * 55)
    print("  AgentSim — 基于LLM多智能体的Robotaxi调度仿真")
    print("=" * 55)
    print(f"地图：{config.MAP_SIZE}×{config.MAP_SIZE}  "
          f"车辆：{config.NUM_DRIVERS}  "
          f"步数：{config.SIM_STEPS}\n")

    # 初始化各模块
    env         = SimEnvironment(seed=42)
    metrics     = MetricsTracker()
    coordinator = CoordinatorAgent(num_drivers=config.NUM_DRIVERS)
    analyst     = AnalystAgent()
    plotter     = SimPlotter()

    # ── 主仿真循环 ─────────────────────────────────────
    for step in range(config.SIM_STEPS):
        print(f"── Step {step+1}/{config.SIM_STEPS} " + "─" * 30)

        # 1. 随机产生乘客请求
        req = env.maybe_generate_passenger()
        if req:
            print(f"  [新乘客] P{req.passenger_id}  "
                  f"上车:{req.pickup} → 目的地:{req.dropoff}")

            # 2. Coordinator 调用 Dispatcher 分配订单
            result = coordinator.assign_order(
                passenger={"id": req.passenger_id,
                           "pickup": req.pickup,
                           "dropoff": req.dropoff},
                drivers=env.drivers,
            )
            did = result["assigned_driver_id"]
            if did is not None:
                env.assign(req.passenger_id, did)
                print(f"  [调度] → 分配给车辆 {did}  ({result['reason']})")
            else:
                print(f"  [调度] 无法分配：{result['reason']}")

        # 3. 所有有任务的车辆移动一步
        from agents.driver import DriverAgent as _DA
        for driver in env.drivers:
            if driver["status"] == "idle":
                continue
            target = env.get_driver_target(driver["id"])
            if target is None:
                continue

            # 判断是否刚接单（本步才变成 to_pickup）
            event = None
            if driver["status"] == "to_pickup" and driver["pos"] == target:
                event = _DA.EVENT_ARRIVED_PICKUP
            elif driver["status"] == "to_dropoff" and driver["pos"] == target:
                event = _DA.EVENT_ARRIVED_DROPOFF

            move_result = coordinator.move_driver(
                driver=driver,
                target=target,
                map_size=config.MAP_SIZE,
                event=event,
            )
            next_pos = move_result["next_pos"]
            msg      = move_result.get("message", "")
            env.move_driver(driver["id"], next_pos)
            print(f"  [车辆{driver['id']}] {driver['pos']} → {next_pos}  {msg[:40]}")

        # 4. 记录快照 & 可视化
        env.advance_step()
        snap = env.snapshot()
        metrics.record(snap)
        plotter.render_step(
            snapshot=snap,
            completed_curve=metrics.completed_per_step(),
            active_curve=metrics.active_per_step(),
        )

        print(f"  状态：等待{snap['pending']} / 进行中{snap['active']} / 完成{snap['completed']}")

    # ── 仿真结束：统计 + 分析报告 ──────────────────────
    print("\n" + "=" * 55)
    print("  仿真结束，生成分析报告...")

    kpi = metrics.summary(
        completed=env.completed_requests,
        pending=env.pending_requests + env.active_requests,
        total_steps=config.SIM_STEPS,
        num_drivers=config.NUM_DRIVERS,
    )

    print(f"\n  KPI 汇总：")
    print(f"    完成订单：{kpi['completed_orders']}")
    print(f"    未完成：  {kpi['pending_orders']}")
    print(f"    平均等待：{kpi['avg_wait_steps']:.1f} 步")
    print(f"    利用率：  {kpi['utilization']*100:.1f}%")

    # Analyst Agent 生成 LLM 分析报告
    report_result = analyst.run({"metrics": kpi})
    report = report_result["report"]
    print(f"\n  [Analyst Agent 报告]\n{report}")

    plotter.save_final(kpi, report)
    print("\n  输出文件保存在 outputs/ 目录下。")
    print("=" * 55)


if __name__ == "__main__":
    run_simulation()
