# eval_subject_routing_holdout.py
"""
学科路由准确率评估 —— 使用 generate_training_set 产出的 holdout 测试集
(training_data/test_holdout.jsonl)
"""
import argparse
import json
import time
from collections import defaultdict

import requests

ROUTER_URL = "http://localhost:8899/v1/chat/completions"

# 与 eval_subject_routing.py 保持一致
SUBJECT_TO_DECISION = {
    "math": "route_stem",
    "physics": "route_stem",
    "engineering": "route_stem",
    "computer science": "route_cs",
    "biology": "route_science",
    "chemistry": "route_science",
    "health": "route_science",
    "law": "route_humanities",
    "history": "route_humanities",
    "philosophy": "route_humanities",
    "psychology": "route_humanities",
    "economics": "route_business",
    "business": "route_business",
    "other": "default-route",
}


def label_jsonl_to_category(label: str) -> str:
    """test_holdout.jsonl 的 label 与 generate_training_set.CAT_NORMALIZE 一致。"""
    if label == "computer_science":
        return "computer science"
    return label.replace("_", " ")


def load_holdout_jsonl(path: str) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description="在 test_holdout.jsonl 上评估学科路由")
    p.add_argument(
        "--data",
        default="../training_data/test_holdout.jsonl",
        help="holdout jsonl 路径（默认 training_data/test_holdout.jsonl)",
    )
    p.add_argument(
        "--output",
        default="routing_accuracy_holdout_results.json",
        help="逐条结果输出路径",
    )
    p.add_argument("--model", default="MoM", help="请求体中的 model 字段")
    p.add_argument("--timeout", type=float, default=15.0, help="单次请求超时秒数")
    p.add_argument("--progress-every", type=int, default=60, help="每 N 条打印进度")
    args = p.parse_args()

    samples = load_holdout_jsonl(args.data)
    print(f"已加载 {len(samples)} 条 from {args.data}")

    results = []
    for i, item in enumerate(samples):
        label = item.get("label", "")
        category = label_jsonl_to_category(label)
        text = item.get("text", "").strip()
        if not text:
            continue

        payload = {
            "model": args.model,
            "messages": [{"role": "user", "content": text}],
            "max_tokens": 1,
            "stream": False,
        }
        t0 = time.perf_counter()
        try:
            resp = requests.post(ROUTER_URL, json=payload, timeout=args.timeout)
            latency_ms = (time.perf_counter() - t0) * 1000
            actual = resp.headers.get("x-vsr-selected-decision", "default-route") or "default-route"
        except Exception:
            latency_ms = -1
            actual = "ERROR"

        expected = SUBJECT_TO_DECISION.get(category, "default-route")
        results.append(
            {
                "label": label,
                "category": category,
                "expected_decision": expected,
                "actual_decision": actual,
                "correct": actual == expected,
                "latency_ms": latency_ms,
                "source_id": item.get("source_id"),
                "answer": item.get("answer", ""),
            }
        )

        pe = args.progress_every
        if pe > 0 and (i + 1) % pe == 0:
            acc = sum(r["correct"] for r in results) / len(results) * 100
            print(f"  进度 {i+1}/{len(samples)}, 当前路由准确率: {acc:.1f}%")

    total = len(results)
    correct = sum(r["correct"] for r in results)
    print(f"\n总体路由准确率: {correct}/{total} = {correct/total*100:.1f}%")

    by_subject = defaultdict(lambda: {"total": 0, "correct": 0, "latencies": []})
    for r in results:
        s = by_subject[r["category"]]
        s["total"] += 1
        if r["correct"]:
            s["correct"] += 1
        if r["latency_ms"] > 0:
            s["latencies"].append(r["latency_ms"])

    print(f"\n{'学科':<22} {'路由准确率':>10} {'正确':>5} {'总数':>5} {'avg lat(ms)':>12}")
    print("-" * 57)
    for cat in sorted(by_subject):
        s = by_subject[cat]
        acc = s["correct"] / s["total"] * 100
        avg = sum(s["latencies"]) / len(s["latencies"]) if s["latencies"] else 0
        print(f"{cat:<22} {acc:>9.1f}% {s['correct']:>5} {s['total']:>5} {avg:>12.1f}")

    decisions = list(set(SUBJECT_TO_DECISION.values()))
    print(f"\n{'Decision':<25} {'Precision':>10} {'Recall':>10} {'F1':>8} {'TP':>5} {'FP':>5} {'FN':>5}")
    print("-" * 75)
    for d in sorted(decisions):
        tp = sum(1 for r in results if r["expected_decision"] == d and r["actual_decision"] == d)
        fp = sum(1 for r in results if r["expected_decision"] != d and r["actual_decision"] == d)
        fn = sum(1 for r in results if r["expected_decision"] == d and r["actual_decision"] != d)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        print(f"{d:<25} {prec:>9.1%} {rec:>9.1%} {f1:>7.3f} {tp:>5} {fp:>5} {fn:>5}")

    out = {
        "data_path": args.data,
        "router_url": ROUTER_URL,
        "total": total,
        "correct": correct,
        "accuracy": correct / total if total else 0,
        "results": results,
    }
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\n完整结果已保存至 {args.output}")


if __name__ == "__main__":
    main()
