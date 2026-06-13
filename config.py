"""
config.py — 全局配置加载
从 .env 读取 API 凭证，同时定义仿真超参数。
"""

import os
from dotenv import load_dotenv

load_dotenv()  # 自动读取同目录下的 .env 文件

# ── LLM 配置 ──────────────────────────────────────────────
API_KEY  = os.getenv("API_KEY",  "")
BASE_URL = os.getenv("BASE_URL", "https://api.xiaomimimo.com/v1")
MODEL    = os.getenv("MODEL",    "mimo-v2.5-pro")

# ── 仿真地图配置 ───────────────────────────────────────────
MAP_SIZE       = 10          # 10×10 网格地图
NUM_DRIVERS    = 4           # 初始车辆数量
SIM_STEPS      = 40          # 每轮仿真步数（提升至40，给足完成时间）
PASSENGER_RATE = 0.25        # 每步产生乘客请求的概率（降低，避免积压）
DRIVER_SPEED   = 1           # 每步移动格数

# Driver Agent 仅在状态变化时调用 LLM（接单/上车/下车），减少 token 消耗
DRIVER_LLM_ON_EVENT_ONLY = True

# ── 需求空间分布 ───────────────────────────────────────────
# True: 乘客上车点聚集在若干"需求热点"（更贴近真实城市，市中心/CBD/枢纽）
#       此时"前瞻指派 + 空闲车重定位"等高级策略才有可利用的空间结构。
# False: 全地图均匀随机（旧行为）。
USE_HOTSPOTS   = True
NUM_HOTSPOTS   = 2           # 热点数量
HOTSPOT_SIGMA  = 1.4         # 热点高斯散布半径（格）

# ── 障碍物 / 路径规划 ──────────────────────────────────────
# True: 地图上随机放置"建筑障碍物"，车辆用 A* 绕障行驶（真实路网）。
#       灵感来自作者一级项目的 RRT-Connect 路径规划（碰撞检测 + 无障碍路径）。
# False: 无障碍，车辆曼哈顿直行（旧行为，实验数据不受影响）。
USE_OBSTACLES     = True
NUM_OBSTACLES     = 6         # 建筑数量
OBSTACLE_MAX_SIZE = 2        # 每栋建筑最大边长（格）

# ── 指标输出 ───────────────────────────────────────────────
OUTPUT_DIR = "outputs"       # 图表保存目录
