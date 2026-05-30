"""
events.py — 仿真事件类型定义
"""
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class PassengerRequest:
    passenger_id: int
    pickup:       list   # [x, y]
    dropoff:      list   # [x, y]
    created_step: int    # 产生于第几步
    assigned_driver_id: Optional[int] = None
    pickup_step:   Optional[int] = None   # 上车时间步
    dropoff_step:  Optional[int] = None   # 下车时间步

    @property
    def wait_steps(self) -> int:
        if self.pickup_step is not None:
            return self.pickup_step - self.created_step
        return -1  # 未完成

    @property
    def is_completed(self) -> bool:
        return self.dropoff_step is not None
