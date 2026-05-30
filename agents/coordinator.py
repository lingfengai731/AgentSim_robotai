"""
coordinator.py — 协调 Agent
职责：全局决策协调，解决分配冲突，管理仿真主循环中的 Agent 调用顺序。
"""

import math
from .base_agent    import BaseAgent
from .dispatcher    import DispatcherAgent
from .driver        import DriverAgent


class CoordinatorAgent(BaseAgent):

    system_prompt = """你是 Robotaxi 运营系统的总协调 AI。
你负责监控整个车队的运行状态，在调度出现冲突时给出裁决意见。
输出一句简短的协调指令即可，不超过 30 字。"""

    def __init__(self, num_drivers: int):
        super().__init__("Coordinator")
        self.dispatcher = DispatcherAgent()
        # 为每辆车创建独立的 DriverAgent
        self.driver_agents = {
            i: DriverAgent(i) for i in range(num_drivers)
        }

    # ── 公开接口 ──────────────────────────────────────────

    def assign_order(self, passenger: dict, drivers: list) -> dict:
        """
        调用 Dispatcher 分配订单；
        若 LLM 分配失败则兜底用距离最近算法。
        返回 {"assigned_driver_id": int or None, "reason": str}
        """
        result = self.dispatcher.run({
            "passenger": passenger,
            "drivers":   drivers,
        })
        # 兜底逻辑
        if result["assigned_driver_id"] is None:
            idle = [d for d in drivers if d["status"] == "idle"]
            if idle:
                best = min(idle, key=lambda d: self._dist(d["pos"], passenger["pickup"]))
                result["assigned_driver_id"] = best["id"]
                result["reason"] = "兜底：距离最近"
        return result

    def move_driver(self, driver: dict, target: list,
                    map_size: int, event: str = None) -> dict:
        """
        调用对应 DriverAgent 计算下一步位置。
        event: DriverAgent.EVENT_* 常量，None 表示普通移动步。
        返回 {"next_pos": [x,y], "message": str}
        """
        agent = self.driver_agents[driver["id"]]
        return agent.run({
            "driver":   driver,
            "target":   target,
            "map_size": map_size,
            "event":    event,
        })

    def arbitrate(self, situation: str) -> str:
        """遇到冲突时请 LLM 给出仲裁指令（可选调用）。"""
        return self.chat(f"当前情况：{situation}，请给出协调指令。")

    # ── 内部工具 ──────────────────────────────────────────

    @staticmethod
    def _dist(a, b) -> float:
        return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)
