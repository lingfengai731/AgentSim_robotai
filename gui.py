"""
gui.py — AgentSim 桌面仿真界面（PyQt5 专业版）
运行方式：python gui.py

相比 tkinter，PyQt5 提供：抗锯齿矢量绘图、QSS 样式（圆角 / 渐变 / 阴影）、
车辆在网格间的平滑补间动画 —— 达到专业仿真系统的观感。

特性：
  · QGraphicsView 矢量地图：建筑（渐变伪3D）、需求热点光晕、A* 规划路径、行驶轨迹
  · 车辆平滑滑行动画（两格之间插值），带光晕与编号
  · 四种调度策略热切换：LLM / 标准匈牙利 / 增强KM / 随机
  · QSS 深色仪表盘：渐变按钮、阴影 KPI 卡、决策日志
  · 仿真在 QThread 后台运行，pyqtSignal 推送快照
"""

import sys
import time
import threading

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QGraphicsView, QGraphicsScene,
    QHBoxLayout, QVBoxLayout, QGridLayout, QLabel, QPushButton, QComboBox,
    QSlider, QFrame, QTextEdit, QGraphicsDropShadowEffect, QSizePolicy,
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer, QPointF, QRectF
from PyQt5.QtGui import (
    QColor, QPen, QBrush, QPainter, QLinearGradient, QRadialGradient,
    QFont, QPainterPath, QPolygonF,
)

import config

# ── 配色（与 Web 版统一）────────────────────────────────
INK      = "#0E0E1A"
PANEL    = "#15152A"
CARD     = "#1C1C34"
GRID     = "#2A2A46"
TEXT     = "#CDD6F4"
MUTED    = "#8A90B8"
ACCENT   = "#89B4FA"
ACCENT2  = "#B4A0FA"
GREEN    = "#A6E3A1"
ORANGE   = "#FAB387"
CYAN     = "#89DCEB"
PINK     = "#F38BA8"
YELLOW   = "#F9E2AF"

STATUS_COLOR = {"idle": GREEN, "to_pickup": ORANGE, "to_dropoff": CYAN}

STRATEGIES = {
    "🤖  LLM 多智能体": "llm",
    "♟  标准匈牙利":     "hungarian",
    "⭐  增强 KM":       "enhanced",
    "🎲  随机调度":      "random",
}

CELL = 50          # 每格像素
FPS  = 60          # 动画帧率


# ════════════════════════════════════════════════════════
#  后台仿真线程
# ════════════════════════════════════════════════════════
class SimWorker(QThread):
    snap_ready = pyqtSignal(dict)
    log_line   = pyqtSignal(str)
    done_sim   = pyqtSignal(dict)

    def __init__(self, strategy_key, seed=42):
        super().__init__()
        self.strategy_key = strategy_key
        self.seed = seed
        self.delay = 0.35
        self._pause = threading.Event(); self._pause.set()
        self._stop = threading.Event()

    def pause(self):  self._pause.clear()
    def resume(self): self._pause.set()
    def stop(self):   self._stop.set(); self._pause.set()
    def set_delay(self, d): self.delay = d

    def _make_dispatcher(self):
        k = self.strategy_key
        if k == "llm":
            from agents.coordinator import CoordinatorAgent
            return CoordinatorAgent(num_drivers=config.NUM_DRIVERS)
        if k == "hungarian":
            from agents.hungarian_dispatcher import HungarianDispatcher
            return HungarianDispatcher()
        if k == "enhanced":
            from agents.enhanced_km_dispatcher import EnhancedKMDispatcher
            return EnhancedKMDispatcher()
        from agents.random_dispatcher import RandomDispatcher
        return RandomDispatcher()

    def run(self):
        from simulation.environment import SimEnvironment
        from agents.driver import DriverAgent as DA

        env = SimEnvironment(seed=self.seed)
        dispatcher = self._make_dispatcher()
        self.snap_ready.emit(env.snapshot())

        for step in range(config.SIM_STEPS):
            self._pause.wait()
            if self._stop.is_set():
                return

            req = env.maybe_generate_passenger()
            if req:
                self.log_line.emit(f"🙋  P{req.passenger_id}  {req.pickup} → {req.dropoff}")

            blocked = env.obstacle_cells
            pending = list(env.pending_requests)
            idle = [d for d in env.drivers if d["status"] == "idle"]
            if pending and idle:
                passengers = [{"id": r.passenger_id, "pickup": r.pickup,
                               "dropoff": r.dropoff,
                               "wait": env.step - r.created_step} for r in pending]
                if hasattr(dispatcher, "assign_batch"):
                    assigns = dispatcher.assign_batch(passengers, env.drivers, blocked=blocked)
                else:
                    assigns, taken = [], set()
                    for p in passengers:
                        cands = [d for d in env.drivers
                                 if d["status"] == "idle" and d["id"] not in taken]
                        if not cands:
                            break
                        res = dispatcher.assign_order(p, cands)
                        if res["assigned_driver_id"] is not None:
                            taken.add(res["assigned_driver_id"])
                            assigns.append({"passenger_id": p["id"],
                                            "assigned_driver_id": res["assigned_driver_id"]})
                for a in assigns:
                    env.assign(a["passenger_id"], a["assigned_driver_id"])
                    self.log_line.emit(f"   📋  P{a['passenger_id']} → 车辆 {a['assigned_driver_id']}")

            for driver in env.drivers:
                if driver["status"] == "idle":
                    if hasattr(dispatcher, "reposition_idle"):
                        env.move_driver(driver["id"],
                            dispatcher.reposition_idle(driver, config.MAP_SIZE, blocked))
                    continue
                target = env.get_driver_target(driver["id"])
                if target is None:
                    continue
                event = None
                if driver["status"] == "to_pickup" and driver["pos"] == target:
                    event = DA.EVENT_ARRIVED_PICKUP
                elif driver["status"] == "to_dropoff" and driver["pos"] == target:
                    event = DA.EVENT_ARRIVED_DROPOFF
                move = dispatcher.move_driver(driver=driver, target=target,
                                              map_size=config.MAP_SIZE,
                                              event=event, blocked=blocked)
                env.move_driver(driver["id"], move["next_pos"])
                if event == DA.EVENT_ARRIVED_PICKUP:
                    self.log_line.emit(f"   🚗  车辆 {driver['id']} 接到乘客")
                elif event == DA.EVENT_ARRIVED_DROPOFF:
                    self.log_line.emit(f"   ✅  车辆 {driver['id']} 送达，订单完成")

            env.advance_step()
            self.snap_ready.emit(env.snapshot())
            time.sleep(max(0.0, self.delay))

        kpi = self._summary(env)
        report = None
        if self.strategy_key == "llm":
            try:
                from agents import AnalystAgent
                report = AnalystAgent().run({"metrics": kpi})["report"]
            except Exception:
                report = None
        self.done_sim.emit({"kpi": kpi, "report": report})

    @staticmethod
    def _summary(env):
        # 用 env 终态构造 KPI（线程内已逐步推进）
        completed = env.completed_requests
        waits = [r.wait_steps for r in completed if r.wait_steps >= 0]
        picks = [r.pickup_distance for r in completed if r.pickup_distance is not None]
        total = len(completed) + len(env.pending_requests) + len(env.active_requests)
        return {
            "completed_orders": len(completed),
            "pending_orders": len(env.pending_requests) + len(env.active_requests),
            "avg_wait_steps": sum(waits)/len(waits) if waits else 0,
            "max_wait_steps": max(waits) if waits else 0,
            "avg_pickup_dist": sum(picks)/len(picks) if picks else 0,
            "utilization": 0,  # GUI 不强调，置0
        }


# ════════════════════════════════════════════════════════
#  地图视图（QGraphicsView，抗锯齿矢量）
# ════════════════════════════════════════════════════════
class MapView(QGraphicsView):
    def __init__(self, map_size):
        super().__init__()
        self.map_size = map_size
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)
        size = map_size * CELL
        self.scene.setSceneRect(0, 0, size, size)
        self.setFixedSize(size + 2, size + 2)
        self.setRenderHint(QPainter.Antialiasing, True)
        self.setRenderHint(QPainter.SmoothPixmapTransform, True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setStyleSheet(f"border:1px solid {GRID}; border-radius:14px; background:{INK};")
        self.setBackgroundBrush(QColor(INK))

    def _center(self, gx, gy):
        return gx * CELL + CELL / 2, gy * CELL + CELL / 2

    def render(self, snap, anim_pos, trails):
        sc = self.scene
        sc.clear()

        # 需求热点：径向渐变光晕
        for (hx, hy) in snap.get("hotspots", []):
            cx, cy = self._center(hx, hy)
            R = CELL * 2.2
            grad = QRadialGradient(cx, cy, R)
            c = QColor(PINK); c.setAlpha(70); grad.setColorAt(0, c)
            c2 = QColor(PINK); c2.setAlpha(0); grad.setColorAt(1, c2)
            sc.addEllipse(cx - R, cy - R, 2*R, 2*R, QPen(Qt.NoPen), QBrush(grad))

        # 网格
        pen_grid = QPen(QColor(GRID)); pen_grid.setWidthF(0.8)
        for i in range(self.map_size + 1):
            sc.addLine(i*CELL, 0, i*CELL, self.map_size*CELL, pen_grid)
            sc.addLine(0, i*CELL, self.map_size*CELL, i*CELL, pen_grid)

        # 建筑（垂直渐变伪3D + 圆角）
        for (ox, oy) in snap.get("obstacles", []):
            x, y = ox*CELL, oy*CELL
            grad = QLinearGradient(x, y, x, y+CELL)
            grad.setColorAt(0, QColor("#454569"))
            grad.setColorAt(1, QColor("#272740"))
            path = QPainterPath()
            path.addRoundedRect(QRectF(x+2, y+2, CELL-4, CELL-4), 5, 5)
            sc.addPath(path, QPen(QColor("#52527A"), 1), QBrush(grad))

        # 行驶轨迹（渐隐）
        for did, pts in trails.items():
            if len(pts) >= 2:
                path = QPainterPath()
                p0 = self._center(*pts[0]); path.moveTo(*p0)
                for p in pts[1:]:
                    path.lineTo(*self._center(*p))
                pen = QPen(QColor(GRID)); pen.setWidthF(2.4)
                sc.addPath(path, pen, QBrush(Qt.NoBrush))

        # A* 规划路径 + 目标
        from simulation.pathfinding import plan_path
        blocked = [tuple(c) for c in snap.get("obstacles", [])]
        for d in snap["drivers"]:
            order = d.get("order")
            if order is None or d["status"] == "idle":
                continue
            color = QColor(STATUS_COLOR.get(d["status"], MUTED))
            tgt = order.pickup if d["status"] == "to_pickup" else order.dropoff
            path_cells = plan_path(d["pos"], tgt, self.map_size, blocked)
            if path_cells and len(path_cells) >= 2:
                pp = QPainterPath(); pp.moveTo(*self._center(*path_cells[0]))
                for c in path_cells[1:]:
                    pp.lineTo(*self._center(*c))
                pen = QPen(color); pen.setWidthF(2.4)
                pen.setStyle(Qt.DashLine); pen.setDashPattern([4, 3])
                sc.addPath(pp, pen, QBrush(Qt.NoBrush))
            # 目标标记
            tx, ty = self._center(*tgt)
            pen = QPen(color); pen.setWidthF(2)
            sc.addRect(tx-9, ty-9, 18, 18, pen, QBrush(Qt.NoBrush))
            self._text(sc, "P" if d["status"] == "to_pickup" else "D",
                       tx, ty, color, 8, bold=True)

        # 待接乘客（脉冲圆点）
        for (px, py) in snap.get("pending_pts", []):
            cx, cy = self._center(px, py)
            halo = QColor(PINK); halo.setAlpha(60)
            sc.addEllipse(cx-13, cy-13, 26, 26, QPen(Qt.NoPen), QBrush(halo))
            sc.addEllipse(cx-6, cy-6, 12, 12, QPen(QColor("white"), 1.2),
                          QBrush(QColor(PINK)))

        # 车辆（光晕 + 渐变球 + 编号），使用插值位置
        for d in snap["drivers"]:
            did = d["id"]
            gx, gy = anim_pos.get(did, d["pos"])
            cx, cy = gx*CELL + CELL/2, gy*CELL + CELL/2
            color = QColor(STATUS_COLOR.get(d["status"], MUTED))
            # 光晕
            R = 22
            grad = QRadialGradient(cx, cy, R)
            c = QColor(color); c.setAlpha(90); grad.setColorAt(0, c)
            c2 = QColor(color); c2.setAlpha(0); grad.setColorAt(1, c2)
            sc.addEllipse(cx-R, cy-R, 2*R, 2*R, QPen(Qt.NoPen), QBrush(grad))
            # 主体球（径向高光）
            r = 13
            body = QRadialGradient(cx-3, cy-4, r*1.6)
            body.setColorAt(0, color.lighter(135))
            body.setColorAt(1, color.darker(110))
            sc.addEllipse(cx-r, cy-r, 2*r, 2*r, QPen(QColor(INK), 2), QBrush(body))
            self._text(sc, str(did), cx, cy, QColor(INK), 10, bold=True)

    def _text(self, sc, s, cx, cy, color, size, bold=False):
        it = sc.addSimpleText(s)
        f = QFont("Segoe UI", size); f.setBold(bold); it.setFont(f)
        it.setBrush(QBrush(color))
        br = it.boundingRect()
        it.setPos(cx - br.width()/2, cy - br.height()/2)


# ════════════════════════════════════════════════════════
#  KPI 卡片
# ════════════════════════════════════════════════════════
class KpiCard(QFrame):
    def __init__(self, label, accent):
        super().__init__()
        self.setObjectName("kpiCard")
        self.setStyleSheet(f"""
            QFrame#kpiCard {{ background:{CARD}; border-radius:14px;
                              border-left:4px solid {accent}; }}
        """)
        sh = QGraphicsDropShadowEffect(self)
        sh.setBlurRadius(22); sh.setOffset(0, 5)
        sh.setColor(QColor(0, 0, 0, 130)); self.setGraphicsEffect(sh)
        lay = QVBoxLayout(self); lay.setContentsMargins(14, 10, 12, 10); lay.setSpacing(2)
        self.lab = QLabel(label); self.lab.setStyleSheet(
            f"color:{MUTED}; font-size:11px; font-weight:600; border:none;")
        self.val = QLabel("—"); self.val.setStyleSheet(
            f"color:{TEXT}; font-size:22px; font-weight:800; border:none;")
        lay.addWidget(self.lab); lay.addWidget(self.val)

    def set_value(self, v):
        self.val.setText(str(v))


# ════════════════════════════════════════════════════════
#  主窗口
# ════════════════════════════════════════════════════════
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AgentSim · Robotaxi 多智能体调度仿真")
        self.worker = None
        self.anim_pos = {}      # {id: [fx, fy]} 浮点插值位置
        self.target_pos = {}    # {id: [x, y]}   目标格
        self.trails = {}
        self.last_snap = None

        self._build_ui()

        self.timer = QTimer(self)
        self.timer.timeout.connect(self._animate)
        self.timer.start(int(1000 / FPS))

    # ── UI ───────────────────────────────────────────────
    def _build_ui(self):
        central = QWidget(); self.setCentralWidget(central)
        central.setStyleSheet(f"background:{INK};")
        root = QVBoxLayout(central); root.setContentsMargins(18, 14, 18, 14); root.setSpacing(10)

        # 顶部标题
        head = QHBoxLayout()
        t1 = QLabel("🚖  AgentSim")
        t1.setStyleSheet(f"color:{ACCENT}; font-size:22px; font-weight:800;")
        t2 = QLabel("Robotaxi 智能调度仿真 · A* 绕障真实路网")
        t2.setStyleSheet(f"color:{MUTED}; font-size:13px; padding-top:8px;")
        head.addWidget(t1); head.addSpacing(10); head.addWidget(t2); head.addStretch()
        root.addLayout(head)

        body = QHBoxLayout(); body.setSpacing(16); root.addLayout(body)

        # 左：地图
        left = QVBoxLayout(); left.setSpacing(8)
        self.map_view = MapView(config.MAP_SIZE)
        left.addWidget(self.map_view)
        left.addLayout(self._legend())
        left.addStretch()
        body.addLayout(left)

        # 右：控制 + KPI + 日志
        right = QVBoxLayout(); right.setSpacing(12)
        right.addWidget(self._control_panel())
        right.addWidget(self._kpi_panel())
        right.addWidget(self._log_panel(), 1)
        body.addLayout(right, 1)

        self._apply_qss()

    def _legend(self):
        lay = QHBoxLayout(); lay.setSpacing(4)
        for color, label in [(GREEN, "空闲"), (ORANGE, "接客中"),
                             (CYAN, "送客中"), (PINK, "待接乘客"), ("#454569", "建筑")]:
            dot = QLabel("●"); dot.setStyleSheet(f"color:{color}; font-size:13px;")
            txt = QLabel(label); txt.setStyleSheet(f"color:{MUTED}; font-size:11px;")
            lay.addWidget(dot); lay.addWidget(txt); lay.addSpacing(8)
        lay.addStretch()
        return lay

    def _control_panel(self):
        panel = QFrame(); panel.setObjectName("panel")
        lay = QVBoxLayout(panel); lay.setContentsMargins(16, 14, 16, 14); lay.setSpacing(10)

        lab = QLabel("调度策略"); lab.setStyleSheet(f"color:{MUTED}; font-size:11px; font-weight:700;")
        lay.addWidget(lab)
        self.combo = QComboBox(); self.combo.addItems(list(STRATEGIES.keys()))
        self.combo.setCurrentIndex(2)  # 默认增强KM
        lay.addWidget(self.combo)

        btnrow = QHBoxLayout(); btnrow.setSpacing(8)
        self.btn_start = QPushButton("▶  开始"); self.btn_start.setObjectName("primary")
        self.btn_pause = QPushButton("⏸  暂停")
        self.btn_reset = QPushButton("🔄  重置")
        self.btn_start.clicked.connect(self.on_start)
        self.btn_pause.clicked.connect(self.on_pause)
        self.btn_reset.clicked.connect(self.on_reset)
        self.btn_pause.setEnabled(False)
        # 主按钮显式样式（确保 Fusion 下蓝色背景可见）
        self.btn_start.setStyleSheet(
            f"QPushButton{{background:{ACCENT};color:{INK};border:none;"
            f"border-radius:11px;padding:9px 8px;font-size:13px;font-weight:700;}}"
            f"QPushButton:hover{{background:{ACCENT2};}}"
            f"QPushButton:disabled{{background:{GRID};color:{MUTED};}}")
        for b in (self.btn_start, self.btn_pause, self.btn_reset):
            btnrow.addWidget(b)
        lay.addLayout(btnrow)

        sprow = QHBoxLayout()
        sprow.addWidget(QLabel("速度"));
        self.speed = QSlider(Qt.Horizontal); self.speed.setMinimum(1); self.speed.setMaximum(5)
        self.speed.setValue(3); self.speed.valueChanged.connect(self._on_speed)
        sprow.addWidget(self.speed)
        lay.addLayout(sprow)
        for i in range(sprow.count()):
            w = sprow.itemAt(i).widget()
            if isinstance(w, QLabel): w.setStyleSheet(f"color:{MUTED}; font-size:11px;")

        self.status = QLabel("就绪 — 选择策略并点击「开始」")
        self.status.setStyleSheet(f"color:{MUTED}; font-size:11px; font-style:italic;")
        self.status.setWordWrap(True)
        lay.addWidget(self.status)
        return panel

    def _kpi_panel(self):
        wrap = QWidget()
        grid = QGridLayout(wrap); grid.setContentsMargins(0, 0, 0, 0); grid.setSpacing(8)
        self.cards = {}
        specs = [("step", "步数", ACCENT), ("done", "完成", GREEN),
                 ("active", "进行中", ORANGE), ("pending", "等待", PINK),
                 ("rate", "完成率", GREEN), ("wait", "平均等待", ACCENT),
                 ("pick", "接驾距离", ACCENT2), ("max", "最长等待", YELLOW)]
        for i, (key, label, color) in enumerate(specs):
            card = KpiCard(label, color)
            self.cards[key] = card
            grid.addWidget(card, i // 4, i % 4)
        self.cards["step"].set_value(f"0/{config.SIM_STEPS}")
        for k in ("done", "active", "pending"):
            self.cards[k].set_value("0")
        return wrap

    def _log_panel(self):
        panel = QFrame(); panel.setObjectName("panel")
        lay = QVBoxLayout(panel); lay.setContentsMargins(14, 12, 14, 12); lay.setSpacing(6)
        lab = QLabel("🤖  Agent 决策日志")
        lab.setStyleSheet(f"color:{ACCENT}; font-size:13px; font-weight:700;")
        lay.addWidget(lab)
        self.log = QTextEdit(); self.log.setReadOnly(True)
        self.log.setStyleSheet(
            f"background:{INK}; color:{TEXT}; border:none; border-radius:10px;"
            f"font-family:Consolas,monospace; font-size:12px; padding:8px;")
        lay.addWidget(self.log)
        return panel

    def _apply_qss(self):
        self.setStyleSheet(f"""
            QFrame#panel {{ background:{PANEL}; border-radius:16px; }}
            QLabel {{ color:{TEXT}; }}
            QComboBox {{ background:{CARD}; color:{TEXT}; border:1px solid {GRID};
                         border-radius:10px; padding:8px 12px; font-size:13px; font-weight:600; }}
            QComboBox:hover {{ border:1px solid {ACCENT}; }}
            QComboBox QAbstractItemView {{ background:{PANEL}; color:{TEXT};
                         selection-background-color:{ACCENT}; selection-color:{INK};
                         border:1px solid {GRID}; outline:none; }}
            QComboBox::drop-down {{ border:none; width:22px; }}
            QPushButton {{ background:{CARD}; color:{TEXT}; border:1px solid {GRID};
                           border-radius:11px; padding:9px 8px; font-size:13px; font-weight:700; }}
            QPushButton:hover {{ border:1px solid {ACCENT}; }}
            QPushButton:disabled {{ color:{MUTED}; }}
            QPushButton#primary {{ background:{ACCENT}; color:{INK}; border:none; }}
            QPushButton#primary:hover {{ background:{ACCENT2}; }}
            QSlider::groove:horizontal {{ height:6px; background:{GRID}; border-radius:3px; }}
            QSlider::handle:horizontal {{ background:{ACCENT}; width:16px; height:16px;
                           margin:-6px 0; border-radius:8px; }}
            QScrollBar:vertical {{ background:{INK}; width:8px; border-radius:4px; }}
            QScrollBar::handle:vertical {{ background:{GRID}; border-radius:4px; }}
        """)

    # ── 动画循环（插值 + 重绘）────────────────────────────
    def _animate(self):
        if self.last_snap is None:
            return
        moved = False
        for did, tgt in self.target_pos.items():
            cur = self.anim_pos.setdefault(did, list(tgt))
            nx = cur[0] + (tgt[0] - cur[0]) * 0.22
            ny = cur[1] + (tgt[1] - cur[1]) * 0.22
            if abs(nx - tgt[0]) < 0.02 and abs(ny - tgt[1]) < 0.02:
                nx, ny = float(tgt[0]), float(tgt[1])
            if cur != [nx, ny]:
                moved = True
            self.anim_pos[did] = [nx, ny]
        self.map_view.render(self.last_snap, self.anim_pos, self.trails)

    # ── 信号处理 ─────────────────────────────────────────
    def on_snap(self, snap):
        self.last_snap = snap
        for d in snap["drivers"]:
            did = d["id"]
            self.target_pos[did] = list(d["pos"])
            self.anim_pos.setdefault(did, list(d["pos"]))
            tr = self.trails.setdefault(did, [])
            if not tr or tr[-1] != tuple(d["pos"]):
                tr.append(tuple(d["pos"]))
                if len(tr) > 14: tr.pop(0)
        total = snap["completed"] + snap["active"] + snap["pending"]
        self.cards["step"].set_value(f"{snap['step']}/{config.SIM_STEPS}")
        self.cards["done"].set_value(snap["completed"])
        self.cards["active"].set_value(snap["active"])
        self.cards["pending"].set_value(snap["pending"])
        if total:
            self.cards["rate"].set_value(f"{snap['completed']/total*100:.0f}%")

    def on_log(self, line):
        self.log.append(line)

    def on_done(self, data):
        kpi = data["kpi"]; report = data.get("report")
        total = kpi["completed_orders"] + kpi["pending_orders"]
        rate = kpi["completed_orders"] / total * 100 if total else 0
        self.cards["rate"].set_value(f"{rate:.1f}%")
        self.cards["wait"].set_value(f"{kpi['avg_wait_steps']:.1f}")
        self.cards["pick"].set_value(f"{kpi.get('avg_pickup_dist',0):.1f}")
        self.cards["max"].set_value(f"{kpi['max_wait_steps']}")
        self.log.append("─" * 30)
        self.log.append(f"🏁  完成率 {rate:.1f}%  ·  平均等待 {kpi['avg_wait_steps']:.1f} 步"
                        f"  ·  接驾 {kpi.get('avg_pickup_dist',0):.1f} 步")
        if report:
            self.log.append("📝  Analyst 报告：")
            for seg in report.split("。"):
                if seg.strip():
                    self.log.append("   " + seg.strip() + "。")
        self.btn_start.setEnabled(False); self.btn_pause.setEnabled(False)
        self.status.setText(f"✅ 仿真完成 · 完成率 {rate:.1f}%")

    # ── 按钮 ─────────────────────────────────────────────
    def _delay(self):
        return (6 - self.speed.value()) * 0.42

    def _on_speed(self):
        if self.worker:
            self.worker.set_delay(self._delay())

    def on_start(self):
        if self.worker and self.worker.isRunning():
            self.worker.resume()
            self.btn_start.setEnabled(False); self.btn_pause.setEnabled(True)
            self.status.setText("仿真运行中…")
            return
        self.log.clear(); self.trails = {}; self.anim_pos = {}; self.target_pos = {}
        key = STRATEGIES[self.combo.currentText()]
        self.worker = SimWorker(key, seed=42)
        self.worker.set_delay(self._delay())
        self.worker.snap_ready.connect(self.on_snap)
        self.worker.log_line.connect(self.on_log)
        self.worker.done_sim.connect(self.on_done)
        self.worker.start()
        self.btn_start.setEnabled(False); self.btn_pause.setEnabled(True)
        self.status.setText(f"仿真运行中… 策略：{self.combo.currentText()}")

    def on_pause(self):
        if self.worker:
            self.worker.pause()
        self.btn_start.setEnabled(True); self.btn_start.setText("▶  继续")
        self.btn_pause.setEnabled(False)
        self.status.setText("已暂停")

    def on_reset(self):
        if self.worker:
            self.worker.stop(); self.worker.wait(500); self.worker = None
        self.last_snap = None; self.trails = {}; self.anim_pos = {}; self.target_pos = {}
        self.map_view.scene.clear()
        self.log.clear()
        for k, c in self.cards.items():
            c.set_value("—")
        self.cards["step"].set_value(f"0/{config.SIM_STEPS}")
        for k in ("done", "active", "pending"):
            self.cards[k].set_value("0")
        self.btn_start.setEnabled(True); self.btn_start.setText("▶  开始")
        self.btn_pause.setEnabled(False)
        self.status.setText("已重置 — 点击「开始」重新仿真")

    def closeEvent(self, e):
        if self.worker:
            self.worker.stop(); self.worker.wait(500)
        e.accept()


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
