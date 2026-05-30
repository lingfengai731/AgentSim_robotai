"""
base_agent.py — 所有 Agent 的基类
封装 OpenAI 兼容接口的调用，子类只需定义 system_prompt 和 run()。
"""

from openai import OpenAI
import config


class BaseAgent:
    """
    每个 Agent = 一个角色 + 一次 LLM 对话。
    子类覆写 system_prompt，调用 self.chat(user_msg) 即可。
    """

    system_prompt: str = "你是一个智能助手。"

    def __init__(self, name: str):
        self.name = name
        self._client = OpenAI(
            api_key=config.API_KEY,
            base_url=config.BASE_URL,
        )

    def chat(self, user_msg: str, temperature: float = 0.3,
             max_retries: int = 2, retry_delay: float = 0.5) -> str:
        """
        向 LLM 发送一条消息，返回回复文本。
        失败自动重试，间隔 retry_delay 秒（默认 0.5s，比原来的 1.5s 快）。
        """
        import time
        last_err = None
        for attempt in range(1, max_retries + 1):
            try:
                response = self._client.chat.completions.create(
                    model=config.MODEL,
                    messages=[
                        {"role": "system",  "content": self.system_prompt},
                        {"role": "user",    "content": user_msg},
                    ],
                    temperature=temperature,
                    max_tokens=512,
                )
                msg = response.choices[0].message

                # 优先取 content
                result = (msg.content or "").strip()

                # 推理模型（如 mimo-v2.5-pro）会把回答放在 reasoning_content
                if not result:
                    result = (getattr(msg, "reasoning_content", None) or "").strip()

                # 还是空：打印调试信息
                if not result:
                    if attempt == 1:   # 只在首次失败时打印完整结构
                        try:
                            raw_dict = response.model_dump()
                            msg_dict = raw_dict['choices'][0]['message']
                            print(f"[{self.name}] ⚠ 空响应 DEBUG: {msg_dict}")
                        except Exception:
                            pass
                    if attempt < max_retries:
                        print(f"[{self.name}] 空内容，重试 ({attempt}/{max_retries})")
                else:
                    return result
            except Exception as e:
                last_err = e
                print(f"[{self.name}] 调用失败 (第{attempt}次): {e}")
            if attempt < max_retries:
                time.sleep(retry_delay)

        print(f"[{self.name}] 已达最大重试次数，启用兜底。最后错误: {last_err}")
        return ""

    def run(self, state: dict) -> dict:
        """子类实现：接收仿真状态，返回决策结果。"""
        raise NotImplementedError
