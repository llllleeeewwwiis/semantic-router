# eval_answer_accuracy.py
import json, re, time, requests
from collections import defaultdict
from datasets import load_dataset

ROUTER_URL = "http://localhost:8899/v1/chat/completions"

with open("data_cleaned.json") as f:
    routing_results = {r["question_id"]: r for r in json.load(f)}

ds = load_dataset("TIGER-Lab/MMLU-Pro", split="test")
samples = [item for item in ds if item["question_id"] in routing_results]

def extract_answer(text):
    text = text.strip()
    for pat in [r'[Tt]he answer is\s*\(?([A-J])\)?',
                r'[Aa]nswer:\s*\(?([A-J])\)?',
                r'^([A-J])[.\s\)]']:
        m = re.search(pat, text)
        if m:
            return m.group(1)
    letters = re.findall(r'\b([A-J])\b', text)
    return letters[0] if len(letters) == 1 else ""

results = []
run_start = time.perf_counter()
print(f"[debug] Loaded {len(samples)} samples for answer-accuracy evaluation.")
for i, item in enumerate(samples):
    options_str = "\n".join(
        f"{chr(65+j)}. {opt}" for j, opt in enumerate(item["options"]) if opt
    )
    prompt = f"{item['question']}\n\n{options_str}\n\nAnswer with just the letter (A-J)."
    payload = {
        "model": "MoM",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 100, "stream": False, "temperature": 0
    }
    t0 = time.perf_counter()
    try:
        resp = requests.post(ROUTER_URL, json=payload, timeout=30)
        latency_ms = (time.perf_counter() - t0) * 1000
        body = resp.json()
        predicted = extract_answer(body["choices"][0]["message"]["content"])
        decision = resp.headers.get("x-vsr-selected-decision", "default-route") or "default-route"
    except Exception as e:
        latency_ms, predicted, decision = -1, "", "ERROR"
        print(
            f"[debug][error] idx={i+1}/{len(samples)} qid={item['question_id']} failed: {type(e).__name__}: {e}"
        )

    routing_meta = routing_results[item["question_id"]]
    results.append({
        "question_id":       item["question_id"],
        "category":          item["category"],
        "expected_decision": routing_meta["expected_decision"],
        "actual_decision":   decision,
        "routing_correct":   routing_meta["correct"],
        "expected_answer":   item["answer"],
        "predicted_answer":  predicted,
        "answer_correct":    predicted == item["answer"],
        "latency_ms":        latency_ms,
    })

    if (i + 1) == 1 or (i + 1) % 20 == 0 or (i + 1) == len(samples):
        elapsed_s = time.perf_counter() - run_start
        done = i + 1
        avg_s = elapsed_s / done
        eta_s = avg_s * (len(samples) - done)
        valid_lat = [r["latency_ms"] for r in results if r["latency_ms"] >= 0]
        avg_latency = sum(valid_lat) / len(valid_lat) if valid_lat else float("nan")
        latest = results[-1]
        print(
            "[debug][progress] "
            f"{done}/{len(samples)} "
            f"elapsed={elapsed_s:.1f}s "
            f"eta={eta_s:.1f}s "
            f"avg_latency={avg_latency:.1f}ms "
            f"latest_qid={latest['question_id']} "
            f"latest_decision={latest['actual_decision']} "
            f"latest_pred={latest['predicted_answer'] or '-'}"
        )

    if (i + 1) % 60 == 0:
        acc = sum(r["answer_correct"] for r in results) / len(results) * 100
        print(f"  进度 {i+1}/{len(samples)}, 当前答题准确率: {acc:.1f}%")

by_cat = defaultdict(lambda: {"routed_correct": [], "routed_wrong": []})
for r in results:
    key = "routed_correct" if r["routing_correct"] else "routed_wrong"
    by_cat[r["category"]][key].append(r["answer_correct"])

print(f"\n{'学科':<20} {'路由正确时答题率':>17} {'路由错误时答题率':>17} {'样本数':>7}")
print("-" * 65)
for cat in sorted(by_cat):
    s = by_cat[cat]
    acc_ok  = sum(s["routed_correct"]) / len(s["routed_correct"]) * 100 if s["routed_correct"] else float("nan")
    acc_bad = sum(s["routed_wrong"])   / len(s["routed_wrong"])   * 100 if s["routed_wrong"]   else float("nan")
    total   = len(s["routed_correct"]) + len(s["routed_wrong"])
    print(f"{cat:<20} {acc_ok:>16.1f}% {acc_bad:>16.1f}% {total:>7}")

with open("answer_accuracy_results.json", "w") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print(f"\n完整结果已保存至 answer_accuracy_results.json")
