"""
random_dispatcher.py — 随机调度基线（不调用 LLM）
职责：从空闲车辆中随机选一辆分配订单，作为对照实验的 baseline。
"""

import random
import math
from simulation.pathfinding import next_step as _next_step


class RandomDispatcher:
    """纯随机调度，不消耗任何 API token，用于与 LLM 调度做对比。"""

    name = "Random"

    def assign_order(self, passenger: dict, drivers: list) -> dict:
        idle = [d for d in drivers if d["status"] == "idle"]
        if not idle:
            return {"assigned_driver_id": None, "reason": "无空闲车辆"}
        chosen = random.choice(idle)
        return {
            "assigned_driver_id": chosen["id"],
            "reason": f"随机选择车辆{chosen['id']}",
        }

    def assign_batch(self, passengers: list, drivers: list, blocked=None) -> list:
        """批量随机匹配：把待分配乘客随机指派给互不重复的空闲车（基线对照）。"""
        idle = [d for d in drivers if d["status"] == "idle"]
        if not idle or not passengers:
            return []
        pool = idle[:]
        random.shuffle(pool)
        assignments = []
        for p, d in zip(passengers, pool):   # 配对数 = min(乘客, 空闲车)
            assignments.append({
                "passenger_id": p["id"],
                "assigned_driver_id": d["id"],
                "reason": f"随机选择车辆{d['id']}",
            })
        return assignments

    def move_driver(self, driver: dict, target: list,
                    map_size: int, event=None, blocked=None) -> dict:
        """确定性移动（A* 绕障），与其它调度器保持一致（控制变量）。"""
        pos = driver["pos"]
        next_pos = _next_step(pos, target, map_size, blocked)
        return {
            "next_pos": next_pos,
            "message":  f"[随机] {pos}→{next_pos}",
        }
