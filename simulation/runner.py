"""
runner.py — 通用仿真运行器（批量匹配版）

把"跑一次仿真"的逻辑抽出来，让 compare.py / app.py 复用，
支持传入任意调度策略（LLM Coordinator / RandomDispatcher / HungarianDispatcher）。

v2 关键改进
-----------
1. **批量匹配**：每一步把"当前所有待分配乘客 × 所有空闲车"一起交给调度器
   的 assign_batch()，而非来一单分一单。这让匈牙利算法能做全局最优指派。
2. **修复遗漏乘客 Bug**：旧版若乘客生成那一步没有空闲车，该乘客就永远不会被
   再次分配。新版每一步都会重试所有 pending 乘客，更贴近真实运营。
3. **向后兼容**：调度器若未实现 assign_batch，自动回退到逐单 assign_order。
"""

import config
from simulation.environment import SimEnvironment
from simulation.metrics     import MetricsTracker


def _dispatch_pending(dispatcher, env, verbose=False, step=0, logs=None):
    """对当前所有 pending 乘客做一次（批量）分配。"""
    pending = list(env.pending_requests)
    idle = [d for d in env.drivers if d["status"] == "idle"]
    if not pending or not idle:
        return

    passengers = [{"id": r.passenger_id, "pickup": r.pickup, "dropoff": r.dropoff,
                   "wait": env.step - r.created_step}   # 已等待步数（供增强KM使用）
                  for r in pending]

    if hasattr(dispatcher, "assign_batch"):
        assignments = dispatcher.assign_batch(passengers, env.drivers,
                                              blocked=env.obstacle_cells)
    else:
        # 回退：逐单分配（边分边从候选里去掉已占用的车）
        assignments = []
        taken = set()
        for p in passengers:
            cands = [d for d in env.drivers
                     if d["status"] == "idle" and d["id"] not in taken]
            if not cands:
                break
            res = dispatcher.assign_order(p, cands)
            did = res["assigned_driver_id"]
            if did is not None:
                taken.add(did)
                assignments.append({"passenger_id": p["id"],
                                    "assigned_driver_id": did,
                                    "reason": res["reason"]})

    for a in assignments:
        did = a["assigned_driver_id"]
        if did is not None:
            env.assign(a["passenger_id"], did)
            if verbose:
                print(f"  Step{step+1} P{a['passenger_id']} → 车辆{did}"
                      f"  ({a['reason']})")
            if logs is not None:
                logs.append(f"  📋 P{a['passenger_id']} → 车辆{did}")


def run_once(dispatcher, seed: int = 42, verbose: bool = False) -> dict:
    """
    运行一次完整仿真。
    dispatcher: 实现 assign_batch()（或 assign_order()）+ move_driver() 的对象。
    返回: metrics summary dict
    """
    env     = SimEnvironment(seed=seed)
    metrics = MetricsTracker()

    for step in range(config.SIM_STEPS):
        # 1. 随机产生乘客（进入 pending 队列）
        env.maybe_generate_passenger()

        # 2. 批量分配：所有 pending 乘客 × 所有空闲车
        _dispatch_pending(dispatcher, env, verbose=verbose, step=step)

        # 3. 移动车辆
        blocked = env.obstacle_cells
        for driver in env.drivers:
            if driver["status"] == "idle":
                # 空闲车重定位：仅当调度器支持（如 Enhanced-KM）才移动，
                # 其它调度器无此方法 → 空闲车原地不动（行为不变）
                if hasattr(dispatcher, "reposition_idle"):
                    new_pos = dispatcher.reposition_idle(driver, config.MAP_SIZE, blocked)
                    env.move_driver(driver["id"], new_pos)
                continue
            target = env.get_driver_target(driver["id"])
            if target is None:
                continue

            from agents.driver import DriverAgent as _DA
            event = None
            if driver["status"] == "to_pickup" and driver["pos"] == target:
                event = _DA.EVENT_ARRIVED_PICKUP
            elif driver["status"] == "to_dropoff" and driver["pos"] == target:
                event = _DA.EVENT_ARRIVED_DROPOFF

            move_result = dispatcher.move_driver(
                driver=driver, target=target,
                map_size=config.MAP_SIZE, event=event, blocked=blocked,
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
    summary["_completed_curve"] = metrics.completed_per_step()
    summary["_active_curve"]    = metrics.active_per_step()
    return summary
