"""
dispatcher.py — 调度 Agent
职责：接收新乘客请求 + 当前所有车辆状态，决定把订单分配给哪辆车。

Prompt 策略：让模型自然回答"选择车辆X"，用正则提取，
避免"严格JSON"指令让推理型模型返回空内容。
"""

import re
import math
from .base_agent import BaseAgent


class DispatcherAgent(BaseAgent):

    system_prompt = """你是 Robotaxi 调度中心的 AI 调度员。
我会给你乘客的位置和所有空闲车辆的位置，
请计算出距离乘客最近的车辆，并用以下固定句式回答：

选择车辆X（X为车辆ID数字），理由：……

例如：选择车辆2，理由：距离乘客最近，仅需3步。"""

    def __init__(self):
        super().__init__("Dispatcher")

    def run(self, state: dict) -> dict:
        """
        state:
          - passenger: {"id": int, "pickup": [x,y], "dropoff": [x,y]}
          - drivers:   [{"id": int, "pos": [x,y], "status": str}, ...]
        返回:
          {"assigned_driver_id": int or None, "reason": str}
        """
        passenger    = state["passenger"]
        idle_drivers = [d for d in state["drivers"] if d["status"] == "idle"]

        if not idle_drivers:
            return {"assigned_driver_id": None, "reason": "当前无空闲车辆"}

        # 预计算距离，加入 prompt 让模型推理更容易
        pickup = passenger["pickup"]
        dist_info = "\n".join(
            f"  车辆{d['id']}：位置{d['pos']}，"
            f"与乘客曼哈顿距离={abs(d['pos'][0]-pickup[0])+abs(d['pos'][1]-pickup[1])}"
            for d in idle_drivers
        )
        user_msg = (
            f"乘客{passenger['id']} 在位置{pickup}，"
            f"目的地{passenger['dropoff']}。\n"
            f"当前空闲车辆距离信息：\n{dist_info}\n"
            f"请选择最合适的车辆。"
        )

        # max_retries=1：只调用一次，失败由 _parse 内部的距离兜底处理
        # 减少重试噪音，Dispatcher 有确定性兜底，不需要多次重试
        raw = self.chat(user_msg, max_retries=1)
        return self._parse(raw, idle_drivers, pickup)

    @staticmethod
    def _parse(raw: str, idle_drivers: list, pickup: list) -> dict:
        """
        从回答中提取车辆ID。
        策略1：匹配"车辆X"中的数字
        策略2：匹配独立数字
        策略3：距离最近兜底
        """
        if raw:
            # 策略1：匹配 "车辆X" 或 "vehicle X"
            m = re.search(r'车辆\s*(\d+)', raw)
            if not m:
                m = re.search(r'vehicle\s*(\d+)', raw, re.IGNORECASE)
            if m:
                did = int(m.group(1))
                # 验证 ID 在空闲车辆中
                valid_ids = {d["id"] for d in idle_drivers}
                if did in valid_ids:
                    reason = raw.strip().replace('\n', ' ')[:60]
                    return {"assigned_driver_id": did, "reason": reason}

        # 策略3：LLM 失败兜底，取距离最近的车辆
        best = min(idle_drivers,
                   key=lambda d: abs(d["pos"][0]-pickup[0]) + abs(d["pos"][1]-pickup[1]))
        return {
            "assigned_driver_id": best["id"],
            "reason": f"兜底(LLM空): 距离最近车辆{best['id']}",
        }
