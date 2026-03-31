import json, time, requests, random
from collections import defaultdict
from datasets import load_dataset

ROUTER_URL = "http://localhost:8899/v1/chat/completions"

# 按自己的方案调整此映射
SUBJECT_TO_DECISION = {
    "math":             "route_stem",
    "physics":          "route_stem",
    "engineering":      "route_stem",
    "computer science": "route_cs",
    "biology":          "route_science",
    "chemistry":        "route_science",
    "health":           "route_science",
    "law":              "route_humanities",
    "history":          "route_humanities",
    "philosophy":       "route_humanities",
    "psychology":       "route_humanities",
    "economics":        "route_business",
    "business":         "route_business",
    "other":            "default-route",
}

print("==== Routing Test Start ====")
print(f"Router URL: {ROUTER_URL}")

random.seed(42)
ds = load_dataset("TIGER-Lab/MMLU-Pro", split="test")

# 每学科取 30 题，共 420 题
by_cat = defaultdict(list)
for item in ds:
    by_cat[item["category"]].append(item)

samples = []
for cat, items in by_cat.items():
    samples.extend(random.sample(items, min(30, len(items))))

print(f"Total samples: {len(samples)}")
print("============================")

results = []
start_time = time.perf_counter()

for i, item in enumerate(samples):
    options_str = "\n".join(
        f"{chr(65+j)}. {opt}" for j, opt in enumerate(item["options"]) if opt
    )

    payload = {
        "model": "MoM",
        "messages": [{"role": "user", "content": f"{item['question']}\n{options_str}"}],
        "max_tokens": 1,
        "stream": False
    }

    t0 = time.perf_counter()

    try:
        resp = requests.post(ROUTER_URL, json=payload, timeout=15)
        latency_ms = (time.perf_counter() - t0) * 1000

        if resp.status_code != 200:
            print(f"[WARN] 非200响应 idx={i} status={resp.status_code}")
            print(f"       body: {resp.text[:200]}")

        actual = resp.headers.get("x-vsr-selected-decision", "default-route") or "default-route"

    except Exception as e:
        latency_ms = -1
        actual = "ERROR"
        print(f"[ERROR] 请求失败 idx={i}, category={item['category']}")
        print(f"        error={repr(e)}")

    expected = SUBJECT_TO_DECISION.get(item["category"], "default-route")

    results.append({
        "category":          item["category"],
        "expected_decision": expected,
        "actual_decision":   actual,
        "correct":           actual == expected,
        "latency_ms":        latency_ms,
        "question_id":       item["question_id"],
        "answer":            item["answer"],
    })

    # ---- 轻量调试输出 ----
    if (i + 1) % 10 == 0:
        print(f"[DEBUG] 已完成 {i+1}/{len(samples)} | 当前category={item['category']}")

    # ---- 进度 + ETA ----
    if (i + 1) % 60 == 0:
        acc = sum(r["correct"] for r in results) / len(results) * 100
        elapsed = time.perf_counter() - start_time
        avg_time = elapsed / (i + 1)
        eta = avg_time * (len(samples) - i - 1)

        print(f"[PROGRESS] {i+1}/{len(samples)} | acc={acc:.1f}% | avg={avg_time:.2f}s | ETA={eta/60:.1f} min")

# ── 总体准确率 ──
total   = len(results)
correct = sum(r["correct"] for r in results)
print(f"\n总体路由准确率: {correct}/{total} = {correct/total*100:.1f}%")

# ── 按学科统计 ──
by_subject = defaultdict(lambda: {"total": 0, "correct": 0, "latencies": []})
for r in results:
    s = by_subject[r["category"]]
    s["total"] += 1
    if r["correct"]:
        s["correct"] += 1
    if r["latency_ms"] > 0:
        s["latencies"].append(r["latency_ms"])

print(f"\n{'学科':<20} {'路由准确率':>10} {'正确':>5} {'总数':>5} {'avg lat(ms)':>12}")
print("-" * 55)
for cat in sorted(by_subject):
    s = by_subject[cat]
    acc = s["correct"] / s["total"] * 100
    avg = sum(s["latencies"]) / len(s["latencies"]) if s["latencies"] else 0
    print(f"{cat:<20} {acc:>9.1f}% {s['correct']:>5} {s['total']:>5} {avg:>12.1f}")

# ── 按 decision 统计 Precision / Recall / F1 ──
decisions = list(set(SUBJECT_TO_DECISION.values()))
print(f"\n{'Decision':<25} {'Precision':>10} {'Recall':>10} {'F1':>8} {'TP':>5} {'FP':>5} {'FN':>5}")
print("-" * 75)
for d in sorted(decisions):
    tp = sum(1 for r in results if r["expected_decision"] == d and r["actual_decision"] == d)
    fp = sum(1 for r in results if r["expected_decision"] != d and r["actual_decision"] == d)
    fn = sum(1 for r in results if r["expected_decision"] == d and r["actual_decision"] != d)
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
    print(f"{d:<25} {prec:>9.1%} {rec:>9.1%} {f1:>7.3f} {tp:>5} {fp:>5} {fn:>5}")

with open("routing_accuracy_results.json", "w") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"\n完整结果已保存至 routing_accuracy_results.json")
print("==== Routing Test Finished ====")