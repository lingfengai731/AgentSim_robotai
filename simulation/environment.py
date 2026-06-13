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

        # 障碍物（建筑）：占据若干网格，车辆需 A* 绕行
        self.obstacle_cells = self._generate_obstacles()

        # 初始化车辆：随机分布在空地（避开障碍物）
        self.drivers = [
            {
                "id":     i,
                "pos":    self._free_cell(),
                "status": "idle",         # idle / to_pickup / to_dropoff
                "order":  None,           # 当前订单 PassengerRequest
            }
            for i in range(config.NUM_DRIVERS)
        ]

        self.pending_requests:   List[PassengerRequest] = []  # 等待分配
        self.active_requests:    List[PassengerRequest] = []  # 已分配，进行中
        self.completed_requests: List[PassengerRequest] = []  # 已完成

        self._next_pid = 0  # 乘客 ID 自增

        # 需求热点：乘客上车点围绕这些中心聚集（更贴近真实城市需求分布）
        if getattr(config, "USE_HOTSPOTS", False):
            self.hotspots = []
            for _ in range(getattr(config, "NUM_HOTSPOTS", 2)):
                c = self._free_cell()
                self.hotspots.append(c)
        else:
            self.hotspots = []

    # ── 障碍物 / 空地工具 ────────────────────────────────
    def _generate_obstacles(self) -> set:
        """随机放置若干矩形"建筑"，返回被占据的网格集合 set[(x,y)]。"""
        if not getattr(config, "USE_OBSTACLES", False):
            return set()
        cells = set()
        n = getattr(config, "NUM_OBSTACLES", 6)
        max_sz = getattr(config, "OBSTACLE_MAX_SIZE", 2)
        budget = int(self.map_size * self.map_size * 0.22)   # 占用上限 ~22%
        for _ in range(n):
            w = random.randint(1, max_sz)
            h = random.randint(1, max_sz)
            ox = random.randint(0, self.map_size - w)
            oy = random.randint(0, self.map_size - h)
            for x in range(ox, ox + w):
                for y in range(oy, oy + h):
                    cells.add((x, y))
            if len(cells) >= budget:
                break
        return cells

    def _free_cell(self) -> list:
        """随机返回一个非障碍空地坐标。"""
        for _ in range(200):
            c = (random.randint(0, self.map_size - 1),
                 random.randint(0, self.map_size - 1))
            if c not in self.obstacle_cells:
                return [c[0], c[1]]
        return [0, 0]

    # ── 状态快照（供 Agent 使用）─────────────────────────

    def snapshot(self) -> dict:
        """返回当前仿真状态的副本（浅拷贝，安全传给 Agent）。"""
        return {
            "step":    self.step,
            "drivers": [
                {**d, "pos": list(d["pos"])}
                for d in self.drivers
            ],
            "obstacles": [list(c) for c in self.obstacle_cells],
            "hotspots":  [list(h) for h in self.hotspots],
            "pending_pts": [list(r.pickup) for r in self.pending_requests],
            "pending":   len(self.pending_requests),
            "active":    len(self.active_requests),
            "completed": len(self.completed_requests),
        }

    # ── 乘客生成 ─────────────────────────────────────────

    def _clamp(self, v):
        return max(0, min(self.map_size - 1, int(round(v))))

    def _sample_pickup(self) -> list:
        """上车点采样：有热点则围绕随机热点高斯散布，否则全图均匀；均避开障碍。"""
        for _ in range(50):
            if self.hotspots:
                cx, cy = random.choice(self.hotspots)
                sigma  = getattr(config, "HOTSPOT_SIGMA", 1.4)
                c = [self._clamp(random.gauss(cx, sigma)),
                     self._clamp(random.gauss(cy, sigma))]
            else:
                c = [random.randint(0, self.map_size - 1),
                     random.randint(0, self.map_size - 1)]
            if tuple(c) not in self.obstacle_cells:
                return c
        return self._free_cell()

    def maybe_generate_passenger(self) -> Optional[PassengerRequest]:
        """按概率随机生成一个乘客请求。"""
        if random.random() < config.PASSENGER_RATE:
            pickup  = self._sample_pickup()   # 上车点聚集于需求热点（避障）
            dropoff = self._free_cell()       # 下车点全图均匀（避障）
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
        # 记录接驾距离（派单时车辆 → 上车点的真实路网距离 = 空驶里程）
        from .pathfinding import path_distance
        req.pickup_distance = path_distance(driver["pos"], req.pickup,
                                            self.map_size, self.obstacle_cells)
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
