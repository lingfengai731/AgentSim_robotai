"""
runner.py — 通用仿真运行器
把"跑一次仿真"的逻辑抽出来，让 compare.py 可以复用，
支持传入任意调度策略（LLM Coordinator 或 RandomDispatcher）。
"""

import config
from simulation.environment import SimEnvironment
from simulation.metrics     import MetricsTracker


def run_once(dispatcher, seed: int = 42, verbose: bool = False) -> dict:
    """
    运行一次完整仿真。
    dispatcher: 实现了 assign_order(passenger, drivers) 和
                move_driver(driver, target, map_size, event) 接口的对象。
    返回: metrics summary dict
    """
    env     = SimEnvironment(seed=seed)
    metrics = MetricsTracker()

    for step in range(config.SIM_STEPS):
        # 1. 随机产生乘客
        req = env.maybe_generate_passenger()
        if req:
            result = dispatcher.assign_order(
                passenger={"id": req.passenger_id,
                           "pickup": req.pickup,
                           "dropoff": req.dropoff},
                drivers=env.drivers,
            )
            did = result["assigned_driver_id"]
            if did is not None:
                env.assign(req.passenger_id, did)
                if verbose:
                    print(f"  Step{step+1} P{req.passenger_id}"
                          f" → 车辆{did}  ({result['reason']})")

        # 2. 移动车辆
        for driver in env.drivers:
            if driver["status"] == "idle":
                continue
            target = env.get_driver_target(driver["id"])
            if target is None:
                continue

            # 判断事件
            from agents.driver import DriverAgent as _DA
            event = None
            if driver["status"] == "to_pickup" and driver["pos"] == target:
                event = _DA.EVENT_ARRIVED_PICKUP
            elif driver["status"] == "to_dropoff" and driver["pos"] == target:
                event = _DA.EVENT_ARRIVED_DROPOFF

            move_result = dispatcher.move_driver(
                driver=driver, target=target,
                map_size=config.MAP_SIZE, event=event,
            )
            env.move_driver(driver["id"], move_result["next_pos"])

        env.advance_step()
        metrics.record(env.snapshot())

    summary = metrics.summary(
        completed=env.completed_requests,
        pending=env.pending_requests + env.active_requests,
        total_steps=config.SIM_STEPS,
        num_drivers=config.NUM_DRIVERS,
    )
    # 额外附上逐步数据，供可视化使用
    summary["_completed_curve"] = metrics.completed_per_step()
    summary["_active_curve"]    = metrics.active_per_step()
    return summary
