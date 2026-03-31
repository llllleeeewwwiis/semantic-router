# bench_throughput.py
import asyncio, aiohttp, time, json
from datasets import load_dataset

ROUTER_URL = "http://localhost:8899/v1/chat/completions"

# 从 MMLU-Pro 中取 500 条题目作为负载
ds = load_dataset("TIGER-Lab/MMLU-Pro", split="test")
questions = [ds[i]["question"] for i in range(500)]

async def run_bench(concurrency, total=500):
    sem = asyncio.Semaphore(concurrency)
    latencies = []
    errors = 0

    async def one_req(text):
        nonlocal errors
        payload = {
            "model": "MoM",
            "messages": [{"role": "user", "content": text}],
            "max_tokens": 1, "stream": False
        }
        async with sem:
            try:
                t0 = time.perf_counter()
                async with aiohttp.ClientSession() as s:
                    async with s.post(ROUTER_URL, json=payload,
                                      timeout=aiohttp.ClientTimeout(total=10)) as r:
                        await r.read()
                latencies.append((time.perf_counter() - t0) * 1000)
            except Exception:
                errors += 1

    wall_start = time.perf_counter()
    await asyncio.gather(*[one_req(questions[i % len(questions)]) for i in range(total)])
    wall = time.perf_counter() - wall_start

    s = sorted(latencies) if latencies else [0]
    qps = total / wall
    return {
        "concurrency": concurrency,
        "qps": qps,
        "p50": s[len(s)//2],
        "p99": s[min(int(len(s)*0.99), len(s)-1)],
        "errors": errors
    }

async def main():
    print(f"{'并发度':>8} {'QPS':>8} {'p50(ms)':>10} {'p99(ms)':>10} {'errors':>8}")
    print("-" * 50)
    for c in [1, 5, 10, 20, 50]:
        r = await run_bench(c)
        print(f"{r['concurrency']:>8} {r['qps']:>8.1f} {r['p50']:>10.1f} "
              f"{r['p99']:>10.1f} {r['errors']:>8}")

asyncio.run(main())
