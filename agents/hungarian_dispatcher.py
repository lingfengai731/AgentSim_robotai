"""
hungarian_dispatcher.py — 匈牙利算法（Kuhn-Munkres）批量最优调度器

背景与动机
----------
现有 LLM / 随机调度都是"乘客来一个、就地分一个"的**贪心**策略：
每个订单只看当下、只分给当前最近的车，多个订单同时出现时无法全局协调，
容易出现"两单抢同一辆车、另一辆车空跑"的次优匹配。

工业界（滴滴、Uber 的 batch-matching）采用的做法是
**批量延迟匹配（Batch Matching）**：在一个很短的时间窗内攒下若干订单与若干空闲车，
构造"车×乘客"代价矩阵，用**匈牙利算法**求解**全局总等待距离最小**的一对一最优指派。
这是经典的二分图最优匹配问题（Kuhn-Munkres / Hungarian Algorithm）。

参考
----
- Batch-delay matching for online car-hailing（PLTS-Hungarian），
  Journal of Systems Science and Systems Engineering, 2025.
- "Timing the Match", arXiv:2503.13200 —— 批量匹配 vs 即时匹配的系统性分析。

实现
----
- 默认走自研 O(n³) 匈牙利算法（无第三方依赖，离线可用）。
- 若环境装有 scipy，则用 scipy.optimize.linear_sum_assignment 加速（结果一致）。
"""

import math
import config
from simulation.pathfinding import next_step as _next_step
from simulation.pathfinding import path_distance as _path_distance

# scipy 可选加速：装了就用，没装就回退自研实现
try:
    from scipy.optimize import linear_sum_assignment as _scipy_lsa  # type: ignore
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False


# ════════════════════════════════════════════════════════════════
#  匈牙利算法核心（最小化指派，自研 O(n³) 实现）
# ════════════════════════════════════════════════════════════════

def _hungarian_min(cost: list) -> list:
    """
    经典 Kuhn-Munkres（potential / augmenting-path 版本），最小化总代价。
    要求行数 n <= 列数 m。返回 row->col 的指派列表（长度 n）。
    """
    n = len(cost)
    m = len(cost[0])
    INF = float("inf")

    u = [0.0] * (n + 1)     # 行势
    v = [0.0] * (m + 1)     # 列势
    p = [0] * (m + 1)       # p[j] = 指派到列 j 的行（1-indexed），0 表示未指派
    way = [0] * (m + 1)     # 增广路径回溯

    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = [INF] * (m + 1)
        used = [False] * (m + 1)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = INF
            j1 = -1
            for j in range(1, m + 1):
                if not used[j]:
                    cur = cost[i0 - 1][j - 1] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j] = cur
                        way[j] = j0
                    if minv[j] < delta:
                        delta = minv[j]
                        j1 = j
            for j in range(m + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        # 沿增广路回溯
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break

    ans = [-1] * n
    for j in range(1, m + 1):
        if p[j] > 0:
            ans[p[j] - 1] = j - 1
    return ans


def solve_assignment(cost_matrix: list) -> list:
    """
    求解矩形代价矩阵的最小化一对一指派。
    cost_matrix: rows × cols 的二维列表。
    返回: [(row_idx, col_idx), ...]，长度 = min(rows, cols)。
    """
    rows = len(cost_matrix)
    if rows == 0:
        return []
    cols = len(cost_matrix[0])
    if cols == 0:
        return []

    # scipy 快速路径
    if _HAS_SCIPY:
        r_idx, c_idx = _scipy_lsa(cost_matrix)
        return list(zip(r_idx.tolist(), c_idx.tolist()))

    # 自研实现要求 n <= m，必要时转置
    transposed = False
    if rows > cols:
        cost_matrix = [[cost_matrix[r][c] for r in range(rows)]
                       for c in range(cols)]
        rows, cols = cols, rows
        transposed = True

    assignment = _hungarian_min(cost_matrix)  # row -> col
    pairs = []
    for r, c in enumerate(assignment):
        if c >= 0:
            pairs.append((c, r) if transposed else (r, c))
    return pairs


# ════════════════════════════════════════════════════════════════
#  调度器封装（与 LLM / Random 调度器同接口，可热插拔）
# ════════════════════════════════════════════════════════════════

class HungarianDispatcher:
    """匈牙利算法批量最优调度器，作为强经典基线 / 生产级对照。"""

    name = "Hungarian"

    @staticmethod
    def _manhattan(a, b) -> int:
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    # ── 批量接口（核心）──────────────────────────────────────
    def assign_batch(self, passengers: list, drivers: list, blocked=None) -> list:
        """
        passengers: [{"id", "pickup", "dropoff"}, ...]  当前所有待分配乘客
        drivers:    完整车辆列表（含 status / pos）
        blocked:    障碍物集合（有障碍时代价用 A* 真实路网距离，与执行层一致）
        返回: [{"passenger_id", "assigned_driver_id", "reason"}, ...]

        构造"乘客 × 空闲车"接驾距离矩阵，用匈牙利算法求全局总距离最小指派。
        """
        idle = [d for d in drivers if d["status"] == "idle"]
        if not idle or not passengers:
            return []

        ms = config.MAP_SIZE
        # 代价矩阵：行=乘客，列=空闲车，元素=接驾真实路网距离（无障碍则曼哈顿）
        cost = [
            [_path_distance(d["pos"], p["pickup"], ms, blocked) for d in idle]
            for p in passengers
        ]
        pairs = solve_assignment(cost)

        assignments = []
        for r, c in pairs:
            passenger = passengers[r]
            driver = idle[c]
            assignments.append({
                "passenger_id": passenger["id"],
                "assigned_driver_id": driver["id"],
                "reason": f"匈牙利最优匹配：车{driver['id']}接驾距离{cost[r][c]}",
            })
        return assignments

    # ── 单订单接口（向后兼容，内部转批量）──────────────────
    def assign_order(self, passenger: dict, drivers: list) -> dict:
        res = self.assign_batch([passenger], drivers)
        if res:
            return {
                "assigned_driver_id": res[0]["assigned_driver_id"],
                "reason": res[0]["reason"],
            }
        return {"assigned_driver_id": None, "reason": "无空闲车辆"}

    # ── 车辆移动（确定性 A* 绕障，与其它调度器一致，控制变量）──
    def move_driver(self, driver: dict, target: list,
                    map_size: int, event=None, blocked=None) -> dict:
        pos = driver["pos"]
        next_pos = _next_step(pos, target, map_size, blocked)
        return {"next_pos": next_pos, "message": f"[匈牙利] {pos}→{next_pos}"}

    @staticmethod
    def _move_toward(pos, target, map_size, blocked=None):
        return _next_step(pos, target, map_size, blocked)
