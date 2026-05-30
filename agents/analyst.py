"""
analyst.py — 分析 Agent
职责：仿真结束后，汇总 KPI 数据，用 LLM 生成一段分析报告。
"""

from .base_agent import BaseAgent


class AnalystAgent(BaseAgent):

    system_prompt = """你是 Robotaxi 运营数据分析师。
给你一份仿真运行后的 KPI 汇总，请用 3~5 句话写一段专业的运营分析报告，
指出亮点、问题，并给出 1 条具体改进建议。用中文回答。"""

    def __init__(self):
        super().__init__("Analyst")

    def run(self, state: dict) -> dict:
        """
        state:
          - metrics: dict，包含 avg_wait、completion_rate、utilization 等
        返回:
          {"report": str}
        """
        m = state["metrics"]
        user_msg = (
            f"本次仿真 KPI 汇总：\n"
            f"  · 完成订单数：{m.get('completed_orders', 0)}\n"
            f"  · 未完成订单数：{m.get('pending_orders', 0)}\n"
            f"  · 平均等待时间：{m.get('avg_wait_steps', 0):.1f} 步\n"
            f"  · 车辆平均里程利用率：{m.get('utilization', 0)*100:.1f}%\n"
            f"  · 最长单次等待：{m.get('max_wait_steps', 0)} 步\n"
            f"请给出运营分析报告。"
        )
        report = self.chat(user_msg, temperature=0.6)

        # 兜底：若 LLM 仍返回空，用模板生成报告
        if not report:
            print(f"[{self.name}] LLM 报告为空，启用模板兜底")
            report = self._fallback_report(m)

        return {"report": report}

    @staticmethod
    def _fallback_report(m: dict) -> str:
        total   = m.get("completed_orders", 0) + m.get("pending_orders", 0)
        rate    = m.get("completed_orders", 0) / total * 100 if total else 0
        util    = m.get("utilization", 0) * 100
        avg_w   = m.get("avg_wait_steps", 0)
        return (
            f"本次仿真共产生 {total} 笔订单，完成 {m.get('completed_orders',0)} 笔，"
            f"订单完成率 {rate:.1f}%。"
            f"车辆平均里程利用率为 {util:.1f}%，"
            f"{'表明车辆调度较为紧凑，空驶率低；' if util > 60 else '存在较大空驶优化空间；'}"
            f"乘客平均等待 {avg_w:.1f} 步，"
            f"{'响应较及时。' if avg_w < 5 else '等待时间偏长，建议增加车辆数量或降低乘客请求频率。'}"
            f"建议在后续版本中引入动态优先级机制，对高等待时间订单优先派单，"
            f"以在维持高利用率的同时提升完成率与用户体验。"
        )
