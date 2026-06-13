"""
enhanced_km_dispatcher.py — 增强版匈牙利调度器（Enhanced-KM，自研改进）

在标准 Kuhn-Munkres 批量最优匹配（agents/hungarian_dispatcher.py）的基础上，
针对 Robotaxi 真实运营场景叠加三个改进模块，并保留"全局最优指派"的内核：

  ① 等待时间感知（Aging / Fairness）
     标准 KM 只看接驾距离，久等的乘客和刚到的乘客同等对待，可能"饿死"个别乘客。
     改进：匹配代价 cost -= ALPHA × 已等待步数 —— 等得越久优先级越高，
     在多乘客争抢车辆时让久等者胜出，降低**最长等待时间**。

  ② 前瞻性指派（Anticipatory Assignment）
     标准 KM 不关心车辆送达后停在哪，车可能停到偏远角落拖累后续接单。
     改进：cost += LAMBDA × (下车点到需求热点的距离) —— 在线估计需求重心，
     在代价相近时优先选"送完单后离热点更近"的派遣。

  ③ 空闲车需求重定位（Idle Rebalancing）★ 关键
     标准 KM（及原系统）里空闲车**原地不动**，导致"等更近的车"永远等不来。
     改进：空闲车每步朝在线估计的需求重心巡航一格，使车队整体分布贴近需求，
     缩短未来接驾距离 —— 这是 Anticipatory Rebalancing of RoboTaxi (TR-C, 2023)
     的核心思想，也是让 ① ② 真正发挥作用的前提。

设计要点：本类只重写"代价矩阵构造"(assign_batch) 与"空闲车移动"(reposition_idle)，
匈牙利求解内核仍复用父类 solve_assignment（自研 O(n³) + scipy 加速）。
三个模块均可独立开关，便于做消融实验（ablation study）。
"""

import config
from .hungarian_dispatcher import HungarianDispatcher, solve_assignment
from simulation.pathfinding import path_distance as _path_distance


class EnhancedKMDispatcher(HungarianDispatcher):

    name = "Enhanced-KM"

    # ── 可调超参数 ────────────────────────────────────────
    ALPHA_WAIT       = 0.8    # ① 每等待 1 步，代价降低多少（优先级提升强度）
    LAMBDA_LOOKAHEAD = 0.25   # ② 前瞻指派权重
    REBALANCE_STEP   = True   # ③ 是否启用空闲车重定位

    def __init__(self, use_aging=False, use_lookahead=True, use_rebalance=True):
        # 模块开关（消融实验用）
        self.use_aging     = use_aging
        self.use_lookahead = use_lookahead
        self.use_rebalance = use_rebalance
        # 在线需求热点估计
        self._seen_pids = set()
        self._sum_x = 0.0
        self._sum_y = 0.0
        self._cnt   = 0

    # ── 需求热点（在线重心估计）──────────────────────────
    def _update_demand(self, passengers):
        for p in passengers:
            pid = p["id"]
            if pid not in self._seen_pids:
                self._seen_pids.add(pid)
                self._sum_x += p["pickup"][0]
                self._sum_y += p["pickup"][1]
                self._cnt   += 1

    def _demand_centroid(self):
        if self._cnt == 0:
            c = (config.MAP_SIZE - 1) / 2.0   # 冷启动：地图中心
            return [c, c]
        return [self._sum_x / self._cnt, self._sum_y / self._cnt]

    # ── ① ② 核心：构造增强代价矩阵 + 匈牙利求解 ───────────
    def assign_batch(self, passengers: list, drivers: list, blocked=None) -> list:
        idle = [d for d in drivers if d["status"] == "idle"]
        if not idle or not passengers:
            return []

        self._update_demand(passengers)
        centroid = self._demand_centroid()
        ms = config.MAP_SIZE

        # 代价矩阵：行=乘客，列=空闲车（接驾距离用 A* 真实路网距离）
        cost = [[0.0] * len(idle) for _ in range(len(passengers))]
        raw  = [[0] * len(idle) for _ in range(len(passengers))]
        for i, p in enumerate(passengers):
            wait = p.get("wait", 0) if self.use_aging else 0
            look = (self._manhattan(p["dropoff"], centroid)
                    if self.use_lookahead else 0)
            for j, d in enumerate(idle):
                dist = _path_distance(d["pos"], p["pickup"], ms, blocked)
                raw[i][j] = dist
                cost[i][j] = (dist
                              - self.ALPHA_WAIT * wait           # ① 等待感知
                              + self.LAMBDA_LOOKAHEAD * look)    # ② 前瞻指派

        pairs = solve_assignment(cost)

        assignments = []
        for r, c in pairs:
            passenger, driver = passengers[r], idle[c]
            w = passenger.get("wait", 0)
            assignments.append({
                "passenger_id": passenger["id"],
                "assigned_driver_id": driver["id"],
                "reason": f"增强KM：车{driver['id']}接驾{raw[r][c]}步·等待{w}步",
            })
        return assignments

    # ── ③ 空闲车需求重定位 ────────────────────────────────
    def reposition_idle(self, driver: dict, map_size: int, blocked=None):
        """
        空闲车朝需求重心巡航一格（A* 绕障）。返回新坐标；已在重心/未启用则原位。
        runner 会对所有 idle 车调用本方法（其它调度器无此方法 → 车辆原地不动）。
        """
        if not self.use_rebalance:
            return driver["pos"]
        cx, cy = self._demand_centroid()
        target = [int(round(cx)), int(round(cy))]
        if driver["pos"] == target:
            return driver["pos"]
        return self._move_toward(driver["pos"], target, map_size, blocked)
