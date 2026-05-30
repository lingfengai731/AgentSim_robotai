"""
gui.py — AgentSim 交互式仿真界面
运行方式：python gui.py

布局：
  左侧：实时地图画布（车辆 + 乘客位置动态更新）
  右上：KPI 数据面板
  右下：Agent 决策日志滚动列表
  底部：控制栏（开始 / 暂停 / 重置 + 速度滑块）

仿真在后台线程运行，通过 Queue 把更新推给 GUI 主线程。
"""

import tkinter as tk
from tkinter import ttk, font as tkfont
import threading
import queue
import time
import config

# ── 颜色常量 ──────────────────────────────────────────────
CLR_BG          = "#1E1E2E"   # 深色背景
CLR_GRID        = "#2E2E3E"   # 网格线
CLR_PANEL       = "#28283A"   # 面板背景
CLR_TEXT        = "#CDD6F4"   # 主文字
CLR_ACCENT      = "#89B4FA"   # 强调色（蓝）
CLR_IDLE        = "#A6E3A1"   # 空闲车辆（绿）
CLR_TOPICKUP    = "#FAB387"   # 前往接客（橙）
CLR_TODROPOFF   = "#89DCEB"   # 送客途中（青）
CLR_PASSENGER   = "#F38BA8"   # 乘客（红）
CLR_DONE        = "#6C7086"   # 已完成（灰）
CLR_BTN         = "#313244"   # 按钮背景
CLR_BTN_ACTIVE  = "#45475A"   # 按钮悬停

CELL  = 52          # 每格像素
PAD   = 20          # 画布边距


# ════════════════════════════════════════════════════════
#  SimThread — 后台仿真线程
# ════════════════════════════════════════════════════════

class SimThread(threading.Thread):
    def __init__(self, update_queue: queue.Queue, speed_var: tk.DoubleVar):
        super().__init__(daemon=True)
        self.q          = update_queue
        self.speed_var  = speed_var
        self._pause     = threading.Event()
        self._stop      = threading.Event()
        self._pause.set()   # 初始暂停

    def pause(self):  self._pause.clear()
    def resume(self): self._pause.set()
    def stop(self):   self._stop.set(); self._pause.set()

    def run(self):
        # 延迟导入，避免在主线程初始化时卡住
        from agents     import CoordinatorAgent, AnalystAgent
        from agents.driver import DriverAgent as _DA
        from simulation import SimEnvironment, MetricsTracker

        env         = SimEnvironment(seed=42)
        metrics     = MetricsTracker()
        coordinator = CoordinatorAgent(num_drivers=config.NUM_DRIVERS)
        analyst     = AnalystAgent()

        self.q.put(("init", env.snapshot()))

        for step in range(config.SIM_STEPS):
            # 等待恢复 / 检查停止
            self._pause.wait()
            if self._stop.is_set():
                return

            # 1. 生成乘客
            req = env.maybe_generate_passenger()
            if req:
                self.q.put(("log",
                    f"🙋 P{req.passenger_id} 上车{req.pickup}→{req.dropoff}"))

                result = coordinator.assign_order(
                    passenger={"id": req.passenger_id,
                               "pickup": req.pickup,
                               "dropoff": req.dropoff},
                    drivers=env.drivers,
                )
                did = result["assigned_driver_id"]
                if did is not None:
                    env.assign(req.passenger_id, did)
                    tag = "LLM" if "兜底" not in result["reason"] else "兜底"
                    self.q.put(("log",
                        f"  📋 [{tag}] P{req.passenger_id} → 车辆{did}"))
                else:
                    self.q.put(("log", f"  ⚠ 无空闲车辆"))

            # 2. 移动所有有任务车辆
            for driver in env.drivers:
                if driver["status"] == "idle":
                    continue
                target = env.get_driver_target(driver["id"])
                if target is None:
                    continue

                event = None
                if driver["status"] == "to_pickup" and driver["pos"] == target:
                    event = _DA.EVENT_ARRIVED_PICKUP
                elif driver["status"] == "to_dropoff" and driver["pos"] == target:
                    event = _DA.EVENT_ARRIVED_DROPOFF

                move = coordinator.move_driver(
                    driver=driver, target=target,
                    map_size=config.MAP_SIZE, event=event,
                )
                old_pos = list(driver["pos"])
                env.move_driver(driver["id"], move["next_pos"])

                if event == _DA.EVENT_ARRIVED_PICKUP:
                    self.q.put(("log",
                        f"  🚗 车辆{driver['id']} 到达接客点，乘客上车"))
                elif event == _DA.EVENT_ARRIVED_DROPOFF:
                    self.q.put(("log",
                        f"  ✅ 车辆{driver['id']} 送达目的地，订单完成"))

            # 3. 推进步数 & 快照
            env.advance_step()
            snap = env.snapshot()
            metrics.record(snap)
            self.q.put(("snap", snap))

            # 4. 速度控制（滑块值 1~5，对应 0.1~2.5s 延迟）
            delay = (6 - self.speed_var.get()) * 0.5
            time.sleep(delay)

        # 仿真结束
        kpi = metrics.summary(
            completed=env.completed_requests,
            pending=env.pending_requests + env.active_requests,
            total_steps=config.SIM_STEPS,
            num_drivers=config.NUM_DRIVERS,
        )
        report = analyst.run({"metrics": kpi})["report"]
        self.q.put(("done", {"kpi": kpi, "report": report}))


# ════════════════════════════════════════════════════════
#  SimGUI — 主窗口
# ════════════════════════════════════════════════════════

class SimGUI:
    def __init__(self, root: tk.Tk):
        self.root       = root
        self.q          = queue.Queue()
        self.sim_thread : SimThread = None
        self.running    = False
        self.snap       = None              # 当前仿真快照

        root.title("AgentSim — Robotaxi 多智能体调度仿真")
        root.configure(bg=CLR_BG)
        root.resizable(False, False)

        self._build_ui()
        self._poll()    # 启动 GUI 轮询

    # ── UI 构建 ──────────────────────────────────────────

    def _build_ui(self):
        # 顶部标题
        title_frame = tk.Frame(self.root, bg=CLR_BG)
        title_frame.pack(fill="x", padx=16, pady=(12, 4))
        tk.Label(title_frame,
                 text="🚖  AgentSim — Robotaxi 多智能体调度仿真",
                 bg=CLR_BG, fg=CLR_ACCENT,
                 font=("Segoe UI", 14, "bold")).pack(side="left")

        # 主体区（画布 + 右侧面板）
        body = tk.Frame(self.root, bg=CLR_BG)
        body.pack(padx=16, pady=4, fill="both")

        # 左：地图画布
        map_size_px = config.MAP_SIZE * CELL + PAD * 2
        self.canvas = tk.Canvas(body, width=map_size_px, height=map_size_px,
                                bg=CLR_BG, highlightthickness=0)
        self.canvas.pack(side="left", padx=(0, 12))
        self._draw_grid()

        # 右：KPI + 日志
        right = tk.Frame(body, bg=CLR_BG, width=300)
        right.pack(side="left", fill="both", expand=True)

        # KPI 面板
        kpi_frame = tk.LabelFrame(right, text=" 📊 实时 KPI ",
                                   bg=CLR_PANEL, fg=CLR_ACCENT,
                                   font=("Segoe UI", 10, "bold"),
                                   bd=1, relief="solid")
        kpi_frame.pack(fill="x", pady=(0, 8))

        self.kpi_vars = {}
        kpi_items = [
            ("step",     "步数",     "0 / " + str(config.SIM_STEPS)),
            ("done",     "✅ 完成",   "0"),
            ("active",   "🚗 进行中", "0"),
            ("pending",  "⏳ 等待",   "0"),
            ("util",     "📈 利用率", "—"),
        ]
        for row, (key, label, init) in enumerate(kpi_items):
            tk.Label(kpi_frame, text=label + "：",
                     bg=CLR_PANEL, fg=CLR_TEXT,
                     font=("Segoe UI", 10)).grid(
                row=row, column=0, sticky="w", padx=10, pady=3)
            var = tk.StringVar(value=init)
            self.kpi_vars[key] = var
            tk.Label(kpi_frame, textvariable=var,
                     bg=CLR_PANEL, fg=CLR_ACCENT,
                     font=("Segoe UI", 10, "bold")).grid(
                row=row, column=1, sticky="w", padx=4, pady=3)

        # 日志面板
        log_frame = tk.LabelFrame(right, text=" 🤖 Agent 决策日志 ",
                                   bg=CLR_PANEL, fg=CLR_ACCENT,
                                   font=("Segoe UI", 10, "bold"),
                                   bd=1, relief="solid")
        log_frame.pack(fill="both", expand=True)

        self.log_box = tk.Listbox(log_frame,
                                   bg=CLR_PANEL, fg=CLR_TEXT,
                                   font=("Consolas", 9),
                                   selectbackground=CLR_BTN_ACTIVE,
                                   bd=0, highlightthickness=0,
                                   activestyle="none")
        scrollbar = ttk.Scrollbar(log_frame, orient="vertical",
                                   command=self.log_box.yview)
        self.log_box.config(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        self.log_box.pack(fill="both", expand=True, padx=4, pady=4)

        # 图例
        legend_frame = tk.Frame(self.root, bg=CLR_BG)
        legend_frame.pack(padx=16, pady=(0, 4), fill="x")
        for color, label in [(CLR_IDLE, "空闲"), (CLR_TOPICKUP, "前往接客"),
                              (CLR_TODROPOFF, "送客中"), (CLR_PASSENGER, "等待乘客")]:
            dot = tk.Label(legend_frame, text="■", fg=color, bg=CLR_BG,
                           font=("Segoe UI", 12))
            dot.pack(side="left", padx=(6, 1))
            tk.Label(legend_frame, text=label, fg=CLR_TEXT, bg=CLR_BG,
                     font=("Segoe UI", 9)).pack(side="left", padx=(0, 8))

        # 底部控制栏
        ctrl = tk.Frame(self.root, bg=CLR_PANEL)
        ctrl.pack(fill="x", padx=16, pady=(4, 12))

        btn_style = {"bg": CLR_BTN, "fg": CLR_TEXT,
                     "font": ("Segoe UI", 10, "bold"),
                     "relief": "flat", "cursor": "hand2",
                     "padx": 14, "pady": 6,
                     "activebackground": CLR_BTN_ACTIVE,
                     "activeforeground": CLR_ACCENT}

        self.btn_start = tk.Button(ctrl, text="▶  开始", command=self._on_start,
                                    **btn_style)
        self.btn_start.pack(side="left", padx=6, pady=6)

        self.btn_pause = tk.Button(ctrl, text="⏸  暂停", command=self._on_pause,
                                    state="disabled", **btn_style)
        self.btn_pause.pack(side="left", padx=6, pady=6)

        self.btn_reset = tk.Button(ctrl, text="🔄  重置", command=self._on_reset,
                                    **btn_style)
        self.btn_reset.pack(side="left", padx=6, pady=6)

        # 速度滑块
        tk.Label(ctrl, text="速度：", bg=CLR_PANEL, fg=CLR_TEXT,
                 font=("Segoe UI", 9)).pack(side="left", padx=(20, 2))
        tk.Label(ctrl, text="慢", bg=CLR_PANEL, fg=CLR_TEXT,
                 font=("Segoe UI", 9)).pack(side="left")
        self.speed_var = tk.DoubleVar(value=3)
        speed_slider = ttk.Scale(ctrl, from_=1, to=5,
                                  orient="horizontal",
                                  variable=self.speed_var, length=100)
        speed_slider.pack(side="left", padx=4)
        tk.Label(ctrl, text="快", bg=CLR_PANEL, fg=CLR_TEXT,
                 font=("Segoe UI", 9)).pack(side="left")

        # 状态栏
        self.status_var = tk.StringVar(value="就绪 — 点击「开始」启动仿真")
        tk.Label(ctrl, textvariable=self.status_var,
                 bg=CLR_PANEL, fg=CLR_DONE,
                 font=("Segoe UI", 9, "italic")).pack(
                     side="right", padx=10)

    # ── 地图绘制 ──────────────────────────────────────────

    def _draw_grid(self):
        n = config.MAP_SIZE
        for i in range(n + 1):
            x = PAD + i * CELL
            self.canvas.create_line(x, PAD, x, PAD + n * CELL,
                                     fill=CLR_GRID, width=1)
            self.canvas.create_line(PAD, x, PAD + n * CELL, x,
                                     fill=CLR_GRID, width=1)

    def _xy(self, pos):
        """网格坐标 → 画布像素中心"""
        return PAD + pos[0] * CELL + CELL // 2, PAD + pos[1] * CELL + CELL // 2

    def _render_snap(self, snap: dict):
        self.canvas.delete("dynamic")

        # 画车辆
        for d in snap["drivers"]:
            cx, cy = self._xy(d["pos"])
            color = {
                "idle":       CLR_IDLE,
                "to_pickup":  CLR_TOPICKUP,
                "to_dropoff": CLR_TODROPOFF,
            }.get(d["status"], CLR_DONE)

            r = CELL // 2 - 5
            self.canvas.create_oval(cx-r, cy-r, cx+r, cy+r,
                                     fill=color, outline=CLR_BG,
                                     width=2, tags="dynamic")
            self.canvas.create_text(cx, cy, text=str(d["id"]),
                                     fill=CLR_BG,
                                     font=("Segoe UI", 10, "bold"),
                                     tags="dynamic")

        # 更新 KPI
        self.kpi_vars["step"].set(f"{snap['step']} / {config.SIM_STEPS}")
        self.kpi_vars["done"].set(str(snap["completed"]))
        self.kpi_vars["active"].set(str(snap["active"]))
        self.kpi_vars["pending"].set(str(snap["pending"]))

    # ── 按钮回调 ──────────────────────────────────────────

    def _on_start(self):
        if self.sim_thread and self.sim_thread.is_alive():
            # 已在运行中则恢复
            self.sim_thread.resume()
            self.running = True
            self.btn_start.config(state="disabled")
            self.btn_pause.config(state="normal")
            self.status_var.set("仿真运行中…")
            return

        # 全新启动
        self._clear_log()
        self.sim_thread = SimThread(self.q, self.speed_var)
        self.sim_thread.start()
        self.sim_thread.resume()
        self.running = True
        self.btn_start.config(state="disabled")
        self.btn_pause.config(state="normal")
        self.status_var.set("仿真运行中…")

    def _on_pause(self):
        if self.sim_thread:
            self.sim_thread.pause()
        self.running = False
        self.btn_start.config(state="normal", text="▶  继续")
        self.btn_pause.config(state="disabled")
        self.status_var.set("已暂停")

    def _on_reset(self):
        if self.sim_thread:
            self.sim_thread.stop()
            self.sim_thread = None
        self.running = False
        self.canvas.delete("dynamic")
        self._clear_log()
        self.kpi_vars["step"].set(f"0 / {config.SIM_STEPS}")
        self.kpi_vars["done"].set("0")
        self.kpi_vars["active"].set("0")
        self.kpi_vars["pending"].set("0")
        self.kpi_vars["util"].set("—")
        self.btn_start.config(state="normal", text="▶  开始")
        self.btn_pause.config(state="disabled")
        self.status_var.set("已重置 — 点击「开始」重新仿真")

    # ── 日志 ──────────────────────────────────────────────

    def _log(self, msg: str):
        self.log_box.insert("end", msg)
        self.log_box.see("end")
        if self.log_box.size() > 200:   # 控制最大行数
            self.log_box.delete(0, 10)

    def _clear_log(self):
        self.log_box.delete(0, "end")

    # ── GUI 轮询（每 100ms 消费队列消息）──────────────────

    def _poll(self):
        try:
            while True:
                msg_type, data = self.q.get_nowait()

                if msg_type == "init":
                    self._render_snap(data)

                elif msg_type == "snap":
                    self.snap = data
                    self._render_snap(data)

                elif msg_type == "log":
                    self._log(data)

                elif msg_type == "done":
                    kpi    = data["kpi"]
                    report = data["report"]
                    total  = kpi["completed_orders"] + kpi["pending_orders"]
                    rate   = kpi["completed_orders"] / total * 100 if total else 0
                    self.kpi_vars["util"].set(f"{kpi['utilization']*100:.1f}%")
                    self._log("─" * 38)
                    self._log(f"🏁 仿真结束！完成率 {rate:.1f}%")
                    self._log(f"   平均等待 {kpi['avg_wait_steps']:.1f} 步")
                    self._log("─" * 38)
                    self._log("📝 Analyst Agent 报告：")
                    for line in (report or "（报告生成中…）").split("。"):
                        if line.strip():
                            self._log("  " + line.strip() + "。")
                    self.btn_start.config(state="disabled")
                    self.btn_pause.config(state="disabled")
                    self.status_var.set(
                        f"✅ 仿真完成 | 完成率 {rate:.1f}% | "
                        f"均等待 {kpi['avg_wait_steps']:.1f} 步"
                    )
                    self.running = False

        except queue.Empty:
            pass

        # 每 100ms 再次轮询
        self.root.after(100, self._poll)


# ════════════════════════════════════════════════════════
#  入口
# ════════════════════════════════════════════════════════

def main():
    root = tk.Tk()
    app  = SimGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
