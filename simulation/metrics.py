"""
metrics.py — KPI 统计
记录每步快照，仿真结束后计算汇总指标。
"""

from typing import List
from .events import PassengerRequest


class MetricsTracker:

    def __init__(self):
        self.history: list = []   # 每步快照

    def record(self, snapshot: dict):
        self.history.append(snapshot)

    def summary(self, completed: List[PassengerRequest],
                pending:   List[PassengerRequest],
                total_steps: int,
                num_drivers: int) -> dict:
        """计算最终 KPI 汇总。"""
        wait_times = [r.wait_steps for r in completed if r.wait_steps >= 0]
        pickup_dists = [r.pickup_distance for r in completed
                        if r.pickup_distance is not None]

        # 里程利用率：车辆处于 to_pickup/to_dropoff 的步数 / 总步数
        busy_counts = [
            sum(1 for snap in self.history
                if any(d["id"] == i and d["status"] != "idle"
                       for d in snap["drivers"]))
            for i in range(num_drivers)
        ]
        utilization = (sum(busy_counts) / (num_drivers * total_steps)
                       if total_steps > 0 else 0)

        return {
            "completed_orders": len(completed),
            "pending_orders":   len(pending),
            "avg_wait_steps":   sum(wait_times)/len(wait_times) if wait_times else 0,
            "max_wait_steps":   max(wait_times) if wait_times else 0,
            "avg_pickup_dist":  sum(pickup_dists)/len(pickup_dists) if pickup_dists else 0,
            "utilization":      utilization,
        }

    # 供可视化使用的逐步数据
    def completed_per_step(self) -> list:
        return [snap["completed"] for snap in self.history]

    def active_per_step(self) -> list:
        return [snap["active"] for snap in self.history]
