"""
test_api.py — API 连通性测试
运行方式：python test_api.py
成功则打印模型回复，失败则打印错误原因（帮助排查 base_url）。
"""

from openai import OpenAI
import config

print(f"BASE_URL : {config.BASE_URL}")
print(f"MODEL    : {config.MODEL}")
print(f"API_KEY  : {config.API_KEY[:8]}...（已隐藏）\n")

client = OpenAI(api_key=config.API_KEY, base_url=config.BASE_URL)

try:
    resp = client.chat.completions.create(
        model=config.MODEL,
        messages=[
            {"role": "system", "content": "你是一个助手，回答要简短。"},
            {"role": "user",   "content": "用一句话介绍 Robotaxi。"},
        ],
        max_tokens=100,
    )
    print("✅ API 调用成功！\n")
    print("模型回复：", resp.choices[0].message.content.strip())
except Exception as e:
    print("❌ API 调用失败：")
    print(e)
    print("\n排查建议：")
    print("  1. 检查 .env 里 BASE_URL 是否正确")
    print("  2. 登录 https://platform.xiaomimimo.com 查看 API 文档里的 endpoint")
    print("  3. 确认 API Key 没有过期或用尽额度")
