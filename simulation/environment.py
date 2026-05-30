"""
environment.py — 仿真环境主体
管理地图、车辆状态、乘客队列，驱动每一步的状态转移。
"""

import random
from typing import List, Optional
from .events import PassengerRequest
import config


class SimEnvironment:

    def __init__(self, seed: int = 42):
        random.seed(seed)
        self.map_size   = config.MAP_SIZE
        self.step       = 0

        # 初始化车辆：随机分布在地图上
        self.drivers = [
            {
                "id":     i,
                "pos":    [random.randint(0, self.map_size-1),
                           random.randint(0, self.map_size-1)],
                "status": "idle",         # idle / to_pickup / to_dropoff
                "order":  None,           # 当前订单 PassengerRequest
            }
            for i in range(config.NUM_DRIVERS)
        ]

        self.pending_requests:   List[PassengerRequest] = []  # 等待分配
        self.active_requests:    List[PassengerRequest] = []  # 已分配，进行中
        self.completed_requests: List[PassengerRequest] = []  # 已完成

        self._next_pid = 0  # 乘客 ID 自增

    # ── 状态快照（供 Agent 使用）─────────────────────────

    def snapshot(self) -> dict:
        """返回当前仿真状态的副本（浅拷贝，安全传给 Agent）。"""
        return {
            "step":    self.step,
            "drivers": [
                {**d, "pos": list(d["pos"])}
                for d in self.drivers
            ],
            "pending":   len(self.pending_requests),
            "active":    len(self.active_requests),
            "completed": len(self.completed_requests),
        }

    # ── 乘客生成 ─────────────────────────────────────────

    def maybe_generate_passenger(self) -> Optional[PassengerRequest]:
        """按概率随机生成一个乘客请求。"""
        if random.random() < config.PASSENGER_RATE:
            pickup  = [random.randint(0, self.map_size-1),
                       random.randint(0, self.map_size-1)]
            dropoff = [random.randint(0, self.map_size-1),
                       random.randint(0, self.map_size-1)]
            req = PassengerRequest(
                passenger_id=self._next_pid,
                pickup=pickup,
                dropoff=dropoff,
                created_step=self.step,
            )
            self._next_pid += 1
            self.pending_requests.append(req)
            return req
        return None

    # ── 订单分配 ─────────────────────────────────────────

    def assign(self, passenger_id: int, driver_id: int):
        """将 pending 订单分配给指定车辆。"""
        req = next((r for r in self.pending_requests
                    if r.passenger_id == passenger_id), None)
        driver = next((d for d in self.drivers if d["id"] == driver_id), None)
        if req is None or driver is None:
            return
        req.assigned_driver_id = driver_id
        driver["status"] = "to_pickup"
        driver["order"]  = req
        self.pending_requests.remove(req)
        self.active_requests.append(req)

    # ── 车辆移动 ─────────────────────────────────────────

    def move_driver(self, driver_id: int, next_pos: list):
        """更新车辆位置，并检查是否到达 pickup/dropoff。"""
        driver = next(d for d in self.drivers if d["id"] == driver_id)
        driver["pos"] = next_pos

        if driver["status"] == "to_pickup" and driver["order"]:
            req = driver["order"]
            if driver["pos"] == req.pickup:
                # 到达上车点
                req.pickup_step  = self.step
                driver["status"] = "to_dropoff"

        elif driver["status"] == "to_dropoff" and driver["order"]:
            req = driver["order"]
            if driver["pos"] == req.dropoff:
                # 到达下车点 → 完成
                req.dropoff_step = self.step
                driver["status"] = "idle"
                driver["order"]  = None
                self.active_requests.remove(req)
                self.completed_requests.append(req)

    def advance_step(self):
        self.step += 1

    # ── 车辆目标位置（供 Coordinator 使用）──────────────

    def get_driver_target(self, driver_id: int) -> Optional[list]:
        driver = next((d for d in self.drivers if d["id"] == driver_id), None)
        if driver is None or driver["status"] == "idle":
            return None
        req = driver["order"]
        if driver["status"] == "to_pickup":
            return req.pickup
        return req.dropoff
