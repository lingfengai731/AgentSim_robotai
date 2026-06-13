"""
pathfinding.py — 栅格 A* 路径规划（绕障行驶）

灵感来源：作者大三一级项目《基于基因调控网络的改进 RRT-Connect 路径规划》中的
碰撞检测与无障碍路径搜索思想。这里在离散网格上用 A*（4 邻域，曼哈顿启发式）
实现车辆**绕开建筑障碍物**的真实行驶轨迹。

设计原则：当 blocked 为空（未开启障碍物）时，next_step 退化为与旧版完全一致的
"沿最大轴贪心移动一格"，从而保证既有实验数据不受影响。
"""

import heapq
from functools import lru_cache


def _manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _neighbors(pos, map_size):
    x, y = pos
    for nx, ny in ((x+1, y), (x-1, y), (x, y+1), (x, y-1)):
        if 0 <= nx < map_size and 0 <= ny < map_size:
            yield (nx, ny)


@lru_cache(maxsize=4096)
def _astar(start, goal, map_size, blocked):
    """
    A* 最短路。start/goal: (x,y) 元组；blocked: frozenset[(x,y)]。
    返回路径 [(x,y), ...]（含起终点）或 None（不可达）。带 LRU 缓存。
    """
    if start == goal:
        return (start,)
    # 目标恰好落在障碍内：就近放行（让车开到障碍边缘），避免永远不可达
    blocked = frozenset(b for b in blocked if b != goal)

    open_heap = [(_manhattan(start, goal), 0, start)]
    came = {start: None}
    gscore = {start: 0}

    while open_heap:
        _, g, cur = heapq.heappop(open_heap)
        if cur == goal:
            path = []
            n = cur
            while n is not None:
                path.append(n)
                n = came[n]
            return tuple(reversed(path))
        if g > gscore.get(cur, 1e9):
            continue
        for nb in _neighbors(cur, map_size):
            if nb in blocked:
                continue
            ng = g + 1
            if ng < gscore.get(nb, 1e9):
                gscore[nb] = ng
                came[nb] = cur
                heapq.heappush(open_heap, (ng + _manhattan(nb, goal), ng, nb))
    return None


def _greedy_step(pos, target, map_size, blocked):
    """
    避障贪心兜底：朝最大轴方向走一格，遇障则换另一轴；都不行则原地。
    blocked 为空时与旧版 _move_toward 行为完全一致。
    """
    x, y = pos
    tx, ty = target
    dx, dy = tx - x, ty - y
    cands = []
    if abs(dx) >= abs(dy):
        if dx != 0: cands.append((x + (1 if dx > 0 else -1), y))
        if dy != 0: cands.append((x, y + (1 if dy > 0 else -1)))
    else:
        if dy != 0: cands.append((x, y + (1 if dy > 0 else -1)))
        if dx != 0: cands.append((x + (1 if dx > 0 else -1), y))
    for nx, ny in cands:
        if 0 <= nx < map_size and 0 <= ny < map_size:
            if not blocked or (nx, ny) not in blocked:
                return [nx, ny]
    return [x, y]


def _to_frozen(blocked):
    if not blocked:
        return frozenset()
    return frozenset(tuple(b) for b in blocked)


def next_step(pos, target, map_size, blocked=None):
    """
    返回车辆朝 target 行进的下一格坐标。
    有障碍 → 走 A* 最短路的第一步；无障碍/不可达 → 避障贪心兜底。
    """
    pos = (int(pos[0]), int(pos[1]))
    target = (int(target[0]), int(target[1]))
    bf = _to_frozen(blocked)
    if bf:
        path = _astar(pos, target, map_size, bf)
        if path and len(path) >= 2:
            return [path[1][0], path[1][1]]
    return _greedy_step(pos, target, map_size, bf)


def path_distance(a, b, map_size, blocked=None):
    """
    两点间真实路网距离（A* 最短路步数）。无障碍 → 曼哈顿距离（与旧版一致）。
    供调度器构造代价矩阵：用真实路网距离而非曼哈顿近似，与执行层 A* 一致。
    """
    a = (int(a[0]), int(a[1]))
    b = (int(b[0]), int(b[1]))
    bf = _to_frozen(blocked)
    if not bf:
        return _manhattan(a, b)
    path = _astar(a, b, map_size, bf)
    if path:
        return len(path) - 1
    return _manhattan(a, b)   # 不可达兜底


def plan_path(pos, target, map_size, blocked=None):
    """返回完整规划路径 [[x,y], ...]（供 GUI 绘制），无障碍/不可达则 None。"""
    bf = _to_frozen(blocked)
    if not bf:
        return None
    path = _astar((int(pos[0]), int(pos[1])),
                  (int(target[0]), int(target[1])), map_size, bf)
    return [list(p) for p in path] if path else None
