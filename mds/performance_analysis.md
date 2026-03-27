# Keyword + Preference Routing 性能与准确率分析

## 作业目标

对 vLLM Semantic Router 的 **Keyword + Preference Routing** 链路进行系统性测试，使用 [MMLU-Pro](https://github.com/TIGER-AI-Lab/MMLU-Pro) 作为标准测试集，完成以下两项核心评估：

1. **性能评估**：量化每个环节（Keyword 信号、Preference 信号、Decision 引擎）的延迟消耗
2. **准确率评估**：验证路由决策的正确性，以及不同路由路径对最终回答质量的影响

---

## 一、环境准备

### 1.1 启动 Router（Docker 方式）

```bash
source vsr/bin/activate
vllm-sr serve --image semantic-router/vllm-sr:v0.2.0-0310 --image-pull-policy ifnotpresent
```

等待日志出现以下内容，确认 Preference Contrastive 模式就绪：

```
[Preference Contrastive] preloaded 3/3 example embeddings using model=mmbert in 162ms
Preference classifier initialized successfully with 3 routes
```

### 1.2 确认服务端口

| 端口 | 服务 | 用途 |
|------|------|------|
| 8899 | Envoy 入口 | 客户端请求入口 |
| 9090 | Prometheus | 指标采集 |
| 16686 | Jaeger UI | 链路追踪 |
| 3000 | Grafana | 仪表盘 |

### 1.3 安装 Python 依赖

```bash
pip install datasets aiohttp pandas numpy tabulate
```

### 1.4 下载 MMLU-Pro 测试集

```python
from datasets import load_dataset
ds = load_dataset("TIGER-Lab/MMLU-Pro")
print(f"test: {len(ds['test'])} 条, validation: {len(ds['validation'])} 条")
# test: 12032 条, validation: 70 条
```

MMLU-Pro 数据格式：

| 字段 | 类型 | 说明 |
|------|------|------|
| `question` | string | 题目文本 |
| `options` | list[str] | 10 个选项（A-J） |
| `answer` | string | 正确答案字母（A-J） |
| `answer_index` | int | 正确答案索引（0-9） |
| `category` | string | 学科分类（14 类） |
| `cot_content` | string | Chain-of-Thought 参考推理 |

MMLU-Pro 的 14 个学科分类：

```
math, physics, chemistry, biology, computer_science, engineering,
economics, business, law, health, psychology, philosophy, history, other
```

---

## 二、理解被测链路

### 2.1 config.yaml 路由拓扑

当前配置中共 **7 个路由决策**，按 priority 从高到低：

```
priority 300  urgent_request              ← keyword: urgent/immediate/asap/emergency
priority 290  sensitive_data              ← keyword: SSN/social security/credit card/password
priority 280  filter_spam                 ← keyword: buy now/free money/click here
priority 200  preference_code_generation  ← preference: 语义匹配 "代码生成"
priority 200  preference_bug_fixing       ← preference: 语义匹配 "调试修复"
priority 200  preference_code_review      ← preference: 语义匹配 "代码审查"
priority  50  default-route               ← 兜底
```

### 2.2 请求处理时序

```
请求进入 performDecisionEvaluation()
│
├─ Layer 1: 信号评估（并行 goroutine）
│    ├── Keyword Signal    ~<1ms     正则/BM25/N-gram 匹配
│    └── Preference Signal ~10-100ms Contrastive embedding 推理（Rust FFI）
│    总耗时 = max(keyword, preference)
│
├─ Layer 2: Decision Engine 评估  ~<0.1ms
│    遍历 decisions, 按 priority 选最高命中
│
└─ Layer 3: 模型选择 + Plugin 注入
     注入 system_prompt → 转发到 LLM 后端
```

**关键观察**：Keyword 信号和 Preference 信号**并行执行**，即使 Keyword 已命中高优先级 decision，Preference 的 embedding 推理仍然会跑完。总延迟由最慢的信号决定。

---

## 三、性能测试

### 实验 1：基线延迟 — 三条路径对比

分别构造必定命中 Keyword、必定命中 Preference、两者均不命中的请求，对比延迟差异。

```python
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
```

**需填写的结果表格：**

| 路径 | p50 (ms) | p95 (ms) | p99 (ms) | mean (ms) | 命中 Decision |
|------|----------|----------|----------|-----------|--------------|
| Keyword (urgent_request) | | | | | |
| Preference (code_gen) | | | | | |
| Default (学术问题) | | | | | |

> **思考题 1**：三条路径的 p50 是否接近？如果 Keyword 路径和 Default 路径延迟差异不大，原因是什么？
>
> 提示：查看 `classifier.go:1275` — 所有 signal 在 `isSignalTypeUsed()` 为 true 时**并行启动**，Keyword 虽然 <1ms 完成，但 `wg.Wait()` 必须等 Preference goroutine 跑完。

---

### 实验 2：并发吞吐量测试

使用 MMLU-Pro 题目作为真实负载，测试不同并发度下的 QPS 和尾部延迟。

```python
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
```

**需填写的吞吐量表格：**

| 并发度 | QPS | p50 (ms) | p99 (ms) | 错误数 |
|--------|-----|----------|----------|--------|
| 1 | | | | |
| 5 | | | | |
| 10 | | | | |
| 20 | | | | |
| 50 | | | | |

> **思考题 2**：并发度从 10 提升到 50 时，p99 是否出现明显跳增？如果是，瓶颈在哪一层？
>
> 提示：Preference signal 的 contrastive 分类器在 `contrastive_preference_classifier.go:174` 使用了 `sync.RWMutex` 读锁。embedding 推理本身是否有并发瓶颈取决于 Rust FFI 层的线程安全实现。

---

## 四、准确率测试

> **本节为开放性实验。** 没有固定的"正确配置"——你需要自己设计路由方案，用 MMLU-Pro 的学科标签作为 ground truth 来评估效果。

### 4.1 实验目标

MMLU-Pro 每道题都自带 `category` 字段，这是天然的 ground truth 标签。实验目标是：

> **设计一套路由决策，使 Router 能将 14 个学科的题目自动分配到对应专项模型，并通过答题准确率验证路由的实际收益。**

**学科 → 路由组的参考映射**（可自行调整合并粒度）：

| 学科群 | 包含学科 | 参考 decision 名 | 适合的模型类型 |
|--------|---------|----------------|--------------|
| STEM 推理 | math, physics, engineering | `route_stem` | 数学推理模型（Qwen2.5-Math、DeepSeek-R1） |
| 计算机 | computer_science | `route_cs` | 代码模型（DeepSeek-Coder、Qwen2.5-Coder） |
| 生命科学 | biology, chemistry, health | `route_science` | 理科通识模型 |
| 人文社科 | law, history, philosophy, psychology | `route_humanities` | 强语言理解模型（Llama 系列） |
| 商科 | economics, business | `route_business` | 通用推理模型 |
| 兜底 | other | `default-route` | 通用最强模型 |

---

### 4.2 路由方案设计（三选一实现）

从以下三种信号类型中选择一种，在 `config.yaml` 中实现完整的学科路由配置：

#### 方案 A：Keyword Signal（<1ms，实现最简单）

为每个学科群挑选高区分度的领域词，填入 `keywords` 列表。挑战在于覆盖率：MMLU 题目措辞多样，纯关键词容易漏判。

```yaml
# 示例框架，需自行补充各学科词表
decisions:
  - name: route_stem
    priority: 160
    conditions:
      signal_type: keyword
      keywords:
        - "integral"
        - "eigenvalue"
        # 继续补充...
    backend: math-model
    system_prompt: "You are a mathematics and physics expert. Reason step by step."

  - name: route_cs
    priority: 160
    conditions:
      signal_type: keyword
      keywords:
        - "time complexity"
        - "binary tree"
        # 继续补充...
    backend: code-model
    system_prompt: "You are a computer science expert."

  # route_science / route_humanities / route_business 同理
```

#### 方案 B：Preference Contrastive Signal（10-100ms，零代码改动，推荐）

为每个学科群提供 4-6 条典型例题作为锚点，Router 通过余弦相似度匹配。核心挑战在于锚点选取和 `threshold` 调优。

```yaml
# 示例框架，需自行补充锚点和调整 threshold
decisions:
  - name: route_stem
    priority: 160
    conditions:
      signal_type: preference
      examples:
        - "Solve the integral of x squared from 0 to 1"
        - "Calculate the net force using Newton's second law"
        # 继续补充 4-6 条，覆盖 math/physics/engineering 典型句式
    threshold: 0.75   # 建议从 0.70 开始调，观察 4.3 的 Recall 变化
    backend: stem-model
    system_prompt: "You are an expert in mathematics and physics. Reason step by step."

  - name: route_cs
    priority: 160
    conditions:
      signal_type: preference
      examples:
        - "What is the time complexity of quicksort in the worst case"
        # 继续补充...
    threshold: 0.75
    backend: code-model
    system_prompt: "You are a computer science expert."

  # route_science / route_humanities / route_business 同理
```

#### 方案 C：Category Classifier（5-20ms，精度最高，需训练）

利用 `candle-binding` 的 ModernBERT 基础设施，在 MMLU-Pro 训练集上 LoRA fine-tune 一个 14 分类头。

```
MMLU-Pro question
    → ModernBERT embedding（candle-binding）
    → 14-class softmax head（LoRA fine-tune）
    → subject label + confidence
    → 按 SUBJECT_TO_DECISION 映射路由
    （confidence < 0.6 → fallback 到 default-route）
```

需在 `classification/classifier.go` 新增 `SignalTypeCategory` 信号类型，与现有 Keyword/Preference 并行执行。适合追求最高精度、愿意投入训练成本的情况。

---

### 4.3 路由准确率评估

**目标**：以 MMLU category 标签为 ground truth，量化你的路由方案对每个学科的识别准确率。

```python
# eval_subject_routing.py
"""
学科路由准确率评估
ground truth：MMLU-Pro category 字段 → SUBJECT_TO_DECISION 映射
"""
import json, time, requests, random
from collections import defaultdict
from datasets import load_dataset

ROUTER_URL = "http://localhost:8899/v1/chat/completions"

# 按自己的方案调整此映射
SUBJECT_TO_DECISION = {
    "math":             "route_stem",
    "physics":          "route_stem",
    "engineering":      "route_stem",
    "computer_science": "route_cs",
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

random.seed(42)
ds = load_dataset("TIGER-Lab/MMLU-Pro", split="test")

# 每学科取 30 题，共 420 题
by_cat = defaultdict(list)
for item in ds:
    by_cat[item["category"]].append(item)

samples = []
for cat, items in by_cat.items():
    samples.extend(random.sample(items, min(30, len(items))))

results = []
for i, item in enumerate(samples):
    options_str = "\n".join(
        f"{chr(65+j)}. {opt}" for j, opt in enumerate(item["options"]) if opt
    )
    payload = {
        "model": "MoM",
        "messages": [{"role": "user", "content": f"{item['question']}\n{options_str}"}],
        "max_tokens": 1, "stream": False
    }
    t0 = time.perf_counter()
    try:
        resp = requests.post(ROUTER_URL, json=payload, timeout=15)
        latency_ms = (time.perf_counter() - t0) * 1000
        actual = resp.headers.get("x-vsr-selected-decision", "default-route") or "default-route"
    except Exception:
        latency_ms = -1
        actual = "ERROR"

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

    if (i + 1) % 60 == 0:
        acc = sum(r["correct"] for r in results) / len(results) * 100
        print(f"  进度 {i+1}/{len(samples)}, 当前路由准确率: {acc:.1f}%")

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
```

**需填写的按学科路由准确率表格：**

| 学科 | 预期 Decision | 路由准确率 | 正确数 | 总数 |
|------|-------------|-----------|-------|------|
| math | route_stem | | | 30 |
| physics | route_stem | | | 30 |
| engineering | route_stem | | | 30 |
| computer_science | route_cs | | | 30 |
| biology | route_science | | | 30 |
| chemistry | route_science | | | 30 |
| health | route_science | | | 30 |
| law | route_humanities | | | 30 |
| history | route_humanities | | | 30 |
| philosophy | route_humanities | | | 30 |
| psychology | route_humanities | | | 30 |
| economics | route_business | | | 30 |
| business | route_business | | | 30 |
| other | default-route | | | 30 |
| **总计** | | | | **420** |

**需填写的 Per-Decision Precision/Recall/F1 表格：**

| Decision | Precision | Recall | F1 | TP | FP | FN |
|----------|-----------|--------|-----|----|----|-----|
| route_stem | | | | | | |
| route_cs | | | | | | |
| route_science | | | | | | |
| route_humanities | | | | | | |
| route_business | | | | | | |
| default-route | | | | | | |

> **思考题 3**：哪些学科群之间最容易混淆（即 FP 主要来自哪些 category）？结合你选择的路由方案分析根本原因：是信号本身的局限（关键词覆盖不足 / 锚点语义重叠 / 分类边界模糊），还是学科本身内容交叉？

---

### 4.4 端到端答题准确率评估

**目标**：验证路由到专项模型是否真正提升了答题准确率，对比「专项路由」与「全走 default-route」的效果差异。

```python
# eval_answer_accuracy.py
"""
端到端答题准确率：Router 路由 + LLM 回答 MMLU-Pro 多选题
复用 routing_accuracy_results.json 中的 420 条样本，追加 LLM 答案
"""
import json, re, time, requests
from collections import defaultdict
from datasets import load_dataset

ROUTER_URL = "http://localhost:8899/v1/chat/completions"

with open("routing_accuracy_results.json") as f:
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
    except Exception:
        latency_ms, predicted, decision = -1, "", "ERROR"

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

    if (i + 1) % 60 == 0:
        acc = sum(r["answer_correct"] for r in results) / len(results) * 100
        print(f"  进度 {i+1}/{len(samples)}, 当前答题准确率: {acc:.1f}%")

# ── 按学科 + 路由是否正确 分层统计 ──
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
```

**需填写的学科路由效果对比表格：**

| 学科 | 路由准确率 | 路由正确时答题率 | 路由错误时答题率 | Δ（正确-错误） |
|------|-----------|---------------|---------------|--------------|
| math | | | | |
| physics | | | | |
| engineering | | | | |
| computer_science | | | | |
| biology | | | | |
| chemistry | | | | |
| health | | | | |
| law | | | | |
| history | | | | |
| philosophy | | | | |
| psychology | | | | |
| economics | | | | |
| business | | | | |
| other | | | | |

> **思考题 4**：Δ 值（路由正确时 vs 路由错误时的答题率差）是否在所有学科上都是正数？如果某学科 Δ 为负，说明什么？结合你的路由方案和 system_prompt 设计分析原因。

> **思考题 5**：综合路由准确率（4.3 表格）和答题率提升（4.4 表格），哪个学科群的「路由收益」最大？你的方案在哪个学科群上表现最差？如果要改进，你会选择调整信号类型、锚点/词表内容，还是调整学科合并粒度（如把 engineering 从 STEM 组单独拆出来）？

---


## 五、综合分析

### 5.1 性能数据汇总

将实验 1、2 的核心指标填入以下汇总表：

```
┌──────────────────────────────────────────────────────────────────────┐
│                    性能数据汇总                                       │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  各路径端到端延迟 (实验 1)                                            │
│  ┌──────────────────────────────┐                                    │
│  │ Keyword 路径 p50:  _____ ms  │                                    │
│  │ Preference 路径 p50:_____ ms │                                    │
│  │ Default 路径 p50:  _____ ms  │                                    │
│  └──────────────────────────────┘                                    │
│                                                                      │
│  端到端吞吐 (实验 2, 并发=10)                                         │
│  ┌──────────────────────────────┐                                    │
│  │ QPS:               _____     │                                    │
│  │ p50:               _____ ms  │                                    │
│  │ p99:               _____ ms  │                                    │
│  └──────────────────────────────┘                                    │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

### 5.2 准确率数据汇总

```
┌──────────────────────────────────────────────────────────────────────┐
│                    准确率数据汇总                                      │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  路由准确率 (实验 3.2)                                                │
│  ┌──────────────────────────────────────────┐                        │
│  │ A. 原始 MMLU → default 准确率: _____%    │ ← 假阳性率 = 100%-此值 │
│  │ B. 注入 Keyword 路由准确率:     _____%    │                        │
│  │ C. 改写 Preference 路由准确率:  _____%    │                        │
│  │ D. 学科专项路由准确率:          _____%    │                        │
│  │ 总体路由准确率:                 _____%    │                        │
│  └──────────────────────────────────────────┘                        │
│                                                                      │
│  MMLU-Pro 答题准确率 (实验 3.3)                                       │
│  ┌──────────────────────────────────────────┐                        │
│  │ 总体准确率:                     _____%    │                        │
│  │ default-route 路径准确率:       _____%    │                        │
│  │ 专项路由路径准确率（均值）:      _____%    │ ← 对比专项路由效果     │
│  │ 被误路由路径准确率:              _____%    │ ← 对比误路由的影响     │
│  └──────────────────────────────────────────┘                        │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 六、提交清单

| # | 内容 | 格式 |
|---|------|------|
| 1 | 实验 1、2 所有表格的实测数据 | 填入本文档 |
| 2 | 思考题 1-5 的回答（需结合代码和数据） | 文字说明 |
| 3 | 第五部分性能 + 准确率汇总图 | 填入汇总框 |
| 4 | `routing_accuracy_results.json` 完整路由评估结果 | JSON 文件 |
| 5 | `answer_accuracy_results.json` 完整答题评估结果 | JSON 文件 |
| 6 | (选做) 从数据中发现的 1-2 个性能或准确率瓶颈分析 | 自由发挥 |

---

## 附录：关键代码位置速查

| 组件 | 文件 | 行号 | 核心逻辑 |
|------|------|------|---------|
| 信号并行评估入口 | `classification/classifier.go` | 1245 | `EvaluateAllSignalsWithContext()` |
| Keyword 正则匹配 | `classification/keyword_classifier.go` | 259 | `ClassifyWithKeywordsAndCount()` |
| Preference Contrastive 推理 | `classification/contrastive_preference_classifier.go` | 151 | `Classify()` → 余弦相似度 |
| Decision 优先级排序 | `decision/engine.go` | 321 | `selectBestDecision()` |
| 信号是否启用判断 | `classification/classifier.go` | 1275 | `isSignalTypeUsed()` |
| Preference 外部 LLM 调用 | `classification/preference_classifier.go` | 106 | `Classify()` → VLLMClient |
| 路由评估总入口 | `extproc/req_filter_classification.go` | 22 | `performDecisionEvaluation()` |
| 指标记录 | `extproc/req_filter_classification.go` | 106 | `metrics.RecordSignalExtraction()` |

---
