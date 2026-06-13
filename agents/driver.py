"""
driver.py — 司机 Agent
职责：收到订单后决定接受或拒绝，移动时仅在状态变化事件（接单/上车/下车）
      才调用 LLM 播报，普通移动步使用确定性算法，节省 token。
"""

import config
from .base_agent import BaseAgent
from simulation.pathfinding import next_step as _next_step


class DriverAgent(BaseAgent):

    system_prompt = """你是一名 Robotaxi 自动驾驶车辆的 AI 驾驶员。
在关键事件节点（接单、到达上车点、到达下车点），用一句简短的中文播报你的状态，
不超过20字，语气专业简洁。直接说播报内容，不要加任何格式标记。"""

    # 事件类型
    EVENT_ASSIGNED  = "assigned"    # 刚接到订单
    EVENT_ARRIVED_PICKUP  = "arrived_pickup"   # 到达上车点
    EVENT_ARRIVED_DROPOFF = "arrived_dropoff"  # 完成送达

    def __init__(self, driver_id: int):
        super().__init__(f"Driver-{driver_id}")
        self.driver_id = driver_id

    def run(self, state: dict) -> dict:
        """
        state:
          - driver:    {"id", "pos": [x,y], "status"}
          - target:    [x, y]
          - map_size:  int
          - event:     str or None  （EVENT_* 常量，None 表示普通移动步）
        返回:
          {"accept": bool, "next_pos": [x,y], "message": str}
        """
        driver   = state["driver"]
        target   = state["target"]
        pos      = state["driver"]["pos"]
        event    = state.get("event")          # 有事件才调 LLM
        map_size = state["map_size"]
        blocked  = state.get("blocked")        # 障碍物集合（A* 绕障）

        next_pos = _next_step(pos, target, map_size, blocked)

        # 只在关键事件 或 未开启节流时才调 LLM
        if event or not config.DRIVER_LLM_ON_EVENT_ONLY:
            event_desc = {
                self.EVENT_ASSIGNED:        f"我接到新订单，前往{target}接客",
                self.EVENT_ARRIVED_PICKUP:  f"已到达上车点{pos}，乘客上车",
                self.EVENT_ARRIVED_DROPOFF: f"已到达目的地{pos}，订单完成",
            }.get(event, f"行驶中，当前{pos}→目标{target}")

            raw    = self.chat(event_desc)
            result = self._parse(raw)
        else:
            # 普通移动步：跳过 LLM，直接返回
            result = {"accept": True, "message": f"行驶中 {pos}→{next_pos}"}

        result["next_pos"] = next_pos   # 强制使用确定性坐标
        return result

    @staticmethod
    def _move_toward(pos: list, target: list, map_size: int) -> list:
        """每步沿曼哈顿距离最大轴移动一格。"""
        x, y = pos
        tx, ty = target
        dx, dy = tx - x, ty - y
        if abs(dx) >= abs(dy):
            x += (1 if dx > 0 else -1) if dx != 0 else 0
        else:
            y += (1 if dy > 0 else -1) if dy != 0 else 0
        x = max(0, min(map_size - 1, x))
        y = max(0, min(map_size - 1, y))
        return [x, y]

    @staticmethod
    def _parse(raw: str) -> dict:
        # 模型现在直接返回纯文本播报，不再是 JSON
        msg = raw.strip().replace('\n', ' ')[:40] if raw else "行驶中"
        return {"accept": True, "next_pos": None, "message": msg}
