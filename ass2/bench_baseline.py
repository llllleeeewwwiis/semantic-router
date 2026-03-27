# bench_baseline.py
import asyncio, aiohttp, time, statistics, json

ROUTER_URL = "http://localhost:8899/v1/chat/completions"

# 路径 A: 命中 Keyword（priority=300，最快信号但仍需等 preference）
KEYWORD_PAYLOAD = {
    "model": "MoM",
    "messages": [{"role": "user", "content": "I need urgent help with my server"}],
    "max_tokens": 1, "stream": False
}

# 路径 B: 命中 Preference（纯 ML 推理路径）
PREFERENCE_PAYLOAD = {
    "model": "MoM",
    "messages": [{"role": "user", "content": "Write a Python function to implement quicksort"}],
    "max_tokens": 1, "stream": False
}

# 路径 C: 走 default-route（MMLU-Pro 学术问题，keyword 和 preference 都不命中）
DEFAULT_PAYLOAD = {
    "model": "MoM",
    "messages": [{"role": "user", "content": "What is the acceleration due to gravity on Earth?"}],
    "max_tokens": 1, "stream": False
}

async def bench(session, payload, n=100):
    latencies = []
    first_headers = {}
    for i in range(n):
        start = time.perf_counter()
        async with session.post(ROUTER_URL, json=payload) as resp:
            await resp.read()
            if i == 0:
                first_headers = {k: v for k, v in resp.headers.items() if k.startswith("x-vsr")}
            latencies.append((time.perf_counter() - start) * 1000)
    return latencies, first_headers

async def main():
    payloads = [
        ("Keyword 路径 (urgent_request)",  KEYWORD_PAYLOAD),
        ("Preference 路径 (code_gen)",     PREFERENCE_PAYLOAD),
        ("Default 路径 (学术问题)",         DEFAULT_PAYLOAD),
    ]
    async with aiohttp.ClientSession() as session:
        print(f"{'路径':<30} {'p50(ms)':>10} {'p95(ms)':>10} {'p99(ms)':>10} {'mean(ms)':>10}")
        print("-" * 75)
        for label, payload in payloads:
            lats, hdrs = await bench(session, payload)
            s = sorted(lats)
            print(f"{label:<30} {s[len(s)//2]:>10.1f} {s[int(len(s)*0.95)]:>10.1f} "
                  f"{s[int(len(s)*0.99)]:>10.1f} {statistics.mean(lats):>10.1f}")
            for k, v in hdrs.items():
                print(f"  {k}: {v}")
            print()

asyncio.run(main())
