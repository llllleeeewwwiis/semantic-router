import asyncio
from openai import AsyncOpenAI

# 配置信息
NVIDIA_API_KEY = "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"
BASE_URL = "https://integrate.api.nvidia.com/v1"

# 你之前定义的模型列表
MODELS_TO_CHECK = [
    "deepseek-ai/deepseek-v3.1-terminus",                     # default
    "mistralai/mistral-large-3-675b-instruct-2512",  # business
    "qwen/qwen3-coder-480b-a35b-instruct",           # cs
    "meta/llama-4-maverick-17b-128e-instruct",       # humanities
    "google/gemma-3-27b-it",                         # science
    "mistralai/devstral-2-123b-instruct-2512"        # stem
]

async def check_model(client, model_name):
    print(f"正在测试模型: {model_name}...")
    try:
        response = await client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": "Hi"}],
            max_tokens=10
        )
        print(f"✅ [成功] {model_name} | 响应: {response.choices[0].message.content.strip()}")
        return True
    except Exception as e:
        print(f"❌ [失败] {model_name} | 错误详情: {str(e)}")
        return False

async def main():
    client = AsyncOpenAI(api_key=NVIDIA_API_KEY, base_url=BASE_URL)
    
    tasks = [check_model(client, m) for m in MODELS_TO_CHECK]
    results = await asyncio.gather(*tasks)

    print("\n" + "="*30)
    print(f"验证结束: 成功 {sum(results)}/{len(MODELS_TO_CHECK)}")
    print("="*30)

if __name__ == "__main__":
    asyncio.run(main())
