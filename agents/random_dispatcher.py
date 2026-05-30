"""
random_dispatcher.py — 随机调度基线（不调用 LLM）
职责：从空闲车辆中随机选一辆分配订单，作为对照实验的 baseline。
"""

import random
import math


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

    def move_driver(self, driver: dict, target: list,
                    map_size: int, event=None) -> dict:
        """确定性移动，与 LLM 版保持一致（控制变量）。"""
        pos = driver["pos"]
        next_pos = self._move_toward(pos, target, map_size)
        return {
            "next_pos": next_pos,
            "message":  f"[随机] {pos}→{next_pos}",
        }

    @staticmethod
    def _move_toward(pos, target, map_size):
        x, y   = pos
        tx, ty = target
        dx, dy = tx - x, ty - y
        if abs(dx) >= abs(dy):
            x += (1 if dx > 0 else -1) if dx != 0 else 0
        else:
            y += (1 if dy > 0 else -1) if dy != 0 else 0
        return [max(0, min(map_size-1, x)),
                max(0, min(map_size-1, y))]
