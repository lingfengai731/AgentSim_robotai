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

# ── 指标输出 ───────────────────────────────────────────────
OUTPUT_DIR = "outputs"       # 图表保存目录
