# vLLM Semantic Router 安装与配置指南

## 一、安装

有两种运行方式：**Docker 方式（推荐）** 和 **裸机方式**。

### 方式 A：Docker 方式（推荐）

只需安装 Python CLI，由 Docker 托管路由器、Envoy、Grafana、Prometheus、Jaeger 等全部组件。

#### 1. 安装 Python CLI

```bash
python3 -m venv vsr
source vsr/bin/activate
cd src/vllm-sr
pip install -e .
```

#### 2. 编辑配置文件

编辑项目根目录的 `config.yaml`（详见下方「配置说明」章节）：

```yaml
version: v0.1

listeners:
  - name: "http-8899"
    address: "0.0.0.0"
    port: 8899
    timeout: "300s"

providers:
  models:
    - name: "minimax/minimax-m2.5"
      access_key: "<你的 OpenRouter API Key>"
      endpoints:
        - name: "openrouter"
          weight: 1
          endpoint: "openrouter.ai:443/api"
          protocol: "https"
  default_model: "minimax/minimax-m2.5"

decisions:
  - name: "default-route"
    description: "Catch-all route"
    priority: 100
    rules:
      operator: "AND"
      conditions: []
    modelRefs:
      - model: "minimax/minimax-m2.5"
        use_reasoning: false
```

#### 3. 启动

```bash
source vsr/bin/activate
vllm-sr serve --image semantic-router/vllm-sr:v0.2.0-0310 --image-pull-policy ifnotpresent
```

常用启动参数：

| 参数 | 说明 |
|---|---|
| `--image <image>` | 指定 Docker 镜像（默认从远程拉取最新） |
| `--image-pull-policy ifnotpresent` | 本地有镜像就不拉取，节省启动时间 |

启动后会自动：
- 合并 `config.yaml` + `.vllm-sr/router-defaults.yaml` → `.vllm-sr/router-config.yaml`
- 渲染 `.vllm-sr/envoy.template.yaml` → Envoy 配置
- 拉起 Docker 容器（路由器 + Envoy + Grafana + Prometheus + Jaeger）

#### Docker 方式涉及的端口

| 端口 | 服务 |
|---|---|
| 8899 | Envoy 监听入口（客户端请求入口） |
| 8080 | Classification API / Dashboard 后端 |
| 8700 | Dashboard 前端 |
| 50051 | ExtProc gRPC 服务（内部） |
| 9190 | Prometheus metrics（内部） |
| 3000 | Grafana |
| 9090 | Prometheus |
| 16686 | Jaeger UI |

---

### 方式 B：裸机方式

直接编译运行 Go 二进制，不经过 Docker 和 Envoy。适合开发调试。

#### 1. 安装 Go (需要 1.24.1+)

```bash
brew upgrade go
go version
# go version go1.24.x darwin/arm64
```

#### 2. 安装 Rust

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source "$HOME/.cargo/env"
rustc --version   # rustc 1.8x+
cargo --version
```

#### 3. 创建 Python 虚拟环境并安装依赖

```bash
python3 -m venv vsr
source vsr/bin/activate
pip install "huggingface_hub[cli]" "httpx[socks]"
```

#### 4. 构建 Rust 库 (CPU-only, macOS 无 CUDA)

```bash
make rust-ci
# 编译 candle-binding + ml-binding
```

#### 5. 构建 Go Router 二进制

```bash
make build-router
# 产出 bin/router
```

#### 6. 下载 ML 模型

```bash
source vsr/bin/activate
export LD_LIBRARY_PATH=$PWD/candle-binding/target/release:$PWD/ml-binding/target/release
./bin/router -config=config/config.yaml --download-only
# 从 HuggingFace 下载模型
```

#### 7. 编辑配置文件

裸机方式直接使用 `router-config.yaml` 的平铺格式（不经过 Python CLI 转换）。
编辑你通过 `-config=` 指定的配置文件：

```yaml
vllm_endpoints:
  - name: "openrouter"
    address: "openrouter.ai"
    port: 443
    weight: 1
    protocol: "https"
    model: "minimax/minimax-m2.5"
    path: "/api"

model_config:
  "minimax/minimax-m2.5":
    access_key: "<你的 OpenRouter API Key>"

default_model: "minimax/minimax-m2.5"

decisions:
  - name: "default-route"
    description: "Catch-all route"
    priority: 100
    rules:
      operator: "AND"
      conditions: []
    modelRefs:
      - model: "minimax/minimax-m2.5"
        use_reasoning: false
```

> 注意：裸机方式的配置格式和 Docker 方式的 `config.yaml` **不同**，
> 裸机方式直接写平铺的 `vllm_endpoints` / `model_config`，
> Docker 方式写 `providers.models[]` 结构由 CLI 自动转换。

#### 8. 启动 Router

```bash
source vsr/bin/activate
export LD_LIBRARY_PATH=$PWD/candle-binding/target/release:$PWD/ml-binding/target/release
./bin/router -config=config/config.yaml --enable-system-prompt-api=true
```

#### 裸机方式涉及的端口

| 端口 | 服务 |
|---|---|
| 50051 | ExtProc gRPC 服务 |
| 8080 | Classification API 服务 |
| 9190 | Prometheus metrics |

---

## 二、配置说明

### 配置文件关系

```
config.yaml (用户编辑)                     ← Docker 方式的入口
      │
      ▼
  vllm-sr serve (Python CLI)
      │
      ├──► .vllm-sr/router-config.yaml    ← router-defaults.yaml + config.yaml 合并结果（自动生成，勿手动编辑）
      └──► Envoy 配置                     ← 从 envoy.template.yaml 渲染
```

- 要改路由/模型/endpoint → 编辑根目录的 **`config.yaml`**，重启 `vllm-sr serve`
- 要改运行时高级参数 → 编辑 **`.vllm-sr/router-defaults.yaml`**
- **不要直接编辑** `.vllm-sr/router-config.yaml`，它会在下次启动时被覆盖

### .vllm-sr/ 目录结构

```
.vllm-sr/
├── router-config.yaml      # 自动生成的合并配置（勿直接编辑）
├── router-defaults.yaml     # 运行时默认值（高级调优用）
├── router-runtime.json      # 运行时状态（自动生成）
├── envoy.template.yaml      # Envoy 配置模板
├── tools_db.json             # 工具数据库
├── grafana/                  # Grafana dashboard
└── prometheus-config/        # Prometheus 配置
```

---

### config.yaml 详解（Docker 方式）

#### 完整结构

```yaml
version: v0.1

listeners:                        # Envoy 监听器
  - name: "http-8899"
    address: "0.0.0.0"
    port: 8899
    timeout: "300s"

providers:                        # 模型供应商配置
  models:
    - name: "模型名称"            # 需与 decisions 中的 modelRefs.model 一致
      access_key: "api-key"       # API 密钥，路由器自动加 Authorization: Bearer <key>
      endpoints:
        - name: "endpoint名称"
          weight: 1               # 权重，多 endpoint 时用于负载均衡
          endpoint: "host:port/path"  # 后端地址
          protocol: "https"       # http 或 https
  default_model: "模型名称"       # 默认路由模型

decisions:                        # 路由决策规则
  - name: "决策名"
    description: "描述"
    priority: 100                 # 数字越大越优先
    rules:
      operator: "AND"
      conditions: []              # 空 = 匹配所有请求
    modelRefs:
      - model: "模型名称"
        use_reasoning: false
```

#### endpoint 格式与路径拼接

`endpoint` 字段格式为 `host:port/path`，Python CLI 自动拆分为：
- `address` = host
- `port` = port（省略时 http 默认 80，https 默认 443）
- `path` = /path（作为 Envoy 的路径前缀）

**路径拼接规则**：Envoy 将 path 前缀拼接到原始请求路径前面：

```
最终请求路径 = path_prefix + 原始请求路径
```

例如客户端发送 `POST /v1/chat/completions`：

| endpoint 值 | path_prefix | 最终路径 | 适用场景 |
|---|---|---|---|
| `openrouter.ai:443/api` | `/api` | `/api/v1/chat/completions` | OpenRouter |
| `api.openai.com:443` | (无) | `/v1/chat/completions` | OpenAI |
| `localhost:8000` | (无) | `/v1/chat/completions` | 本地 vLLM |
| `dashscope.aliyuncs.com:443/compatible-mode` | `/compatible-mode` | `/compatible-mode/v1/chat/completions` | 阿里云 DashScope |

> **常见陷阱**：OpenRouter 写成 `/api/v1` 会导致最终路径变成 `/api/v1/v1/chat/completions`，返回 404。

#### 外部供应商配置示例

**OpenRouter**

```yaml
providers:
  models:
    - name: "minimax/minimax-m2.5"
      access_key: "sk-or-v1-xxx"
      endpoints:
        - name: "openrouter"
          weight: 1
          endpoint: "openrouter.ai:443/api"     # /api 不是 /api/v1
          protocol: "https"
  default_model: "minimax/minimax-m2.5"
```

**OpenAI**

```yaml
providers:
  models:
    - name: "gpt-4o"
      access_key: "sk-proj-xxx"
      endpoints:
        - name: "openai"
          weight: 1
          endpoint: "api.openai.com:443"          # 无需 path 前缀
          protocol: "https"
  default_model: "gpt-4o"
```

**本地 vLLM**

```yaml
providers:
  models:
    - name: "Qwen/Qwen2.5-14B-Instruct"
      endpoints:
        - name: "local"
          weight: 1
          endpoint: "host.docker.internal:8000"   # Docker 内访问宿主机
          protocol: "http"
  default_model: "Qwen/Qwen2.5-14B-Instruct"
```

**多模型 + 多供应商 + 领域路由**

```yaml
providers:
  models:
    - name: "gpt-4o"
      access_key: "sk-proj-xxx"
      endpoints:
        - name: "openai"
          weight: 1
          endpoint: "api.openai.com:443"
          protocol: "https"
    - name: "minimax/minimax-m2.5"
      access_key: "sk-or-v1-xxx"
      endpoints:
        - name: "openrouter"
          weight: 1
          endpoint: "openrouter.ai:443/api"
          protocol: "https"
  default_model: "gpt-4o"

decisions:
  - name: "coding-route"
    description: "编程问题用 GPT-4o"
    priority: 200
    rules:
      operator: "AND"
      conditions:
        - type: "domain"
          name: "computer_science"
    modelRefs:
      - model: "gpt-4o"
        use_reasoning: false
  - name: "default-route"
    description: "其他请求用 minimax"
    priority: 100
    rules:
      operator: "AND"
      conditions: []
    modelRefs:
      - model: "minimax/minimax-m2.5"
        use_reasoning: false
```

#### config.yaml → router-config.yaml 字段映射

| config.yaml | router-config.yaml | 说明 |
|---|---|---|
| `providers.models[].name` | `vllm_endpoints[].model` | 模型名称 |
| `providers.models[].access_key` | `model_config[name].access_key` | API 密钥 |
| `providers.models[].endpoints[].name` | `vllm_endpoints[].name`（格式：`{模型名}_{endpoint名}`） | endpoint 标识 |
| `providers.models[].endpoints[].endpoint` | 拆分为 `address` + `port` + `path` | 后端地址 |
| `providers.models[].endpoints[].protocol` | `vllm_endpoints[].protocol` | 协议 |
| `providers.models[].endpoints[].weight` | `vllm_endpoints[].weight` | 权重 |
| `providers.default_model` | `default_model` | 默认模型 |
| `decisions` | `decisions` | 直接透传 |

---

### router-defaults.yaml 详解

此文件定义路由器内部功能的默认参数，大多数用户不需要修改。

#### response_api — OpenAI Response API

```yaml
response_api:
  enabled: true                  # 启用 Response API（会话链式调用）
  store_backend: "memory"        # memory / milvus / redis
  ttl_seconds: 86400             # 过期时间（默认 24 小时）
  max_responses: 1000            # 最大存储数量
```

#### router_replay — 路由回放

```yaml
router_replay:
  store_backend: "memory"        # memory / redis / postgres / milvus
  ttl_seconds: 2592000           # 保留时间（默认 30 天）
  async_writes: false            # 异步写入（提升性能，可能丢数据）
```

#### memory — 跨会话记忆

```yaml
memory:
  enabled: false                 # 默认关闭，需要 Milvus
  auto_store: false              # 自动存储提取的事实
  milvus:
    address: ""                  # 如 "localhost:19530"
    collection: "agentic_memory"
    dimension: 384
  embedding:
    model: "all-MiniLM-L6-v2"
    dimension: 384
  default_retrieval_limit: 5
  default_similarity_threshold: 0.70
  extraction_batch_size: 10      # 每 N 轮对话提取一次事实
```

启用步骤：设置 `enabled: true` → 配置 `milvus.address` → (可选) 添加 `external_models`。

#### semantic_cache — 语义缓存

```yaml
semantic_cache:
  enabled: true
  backend_type: "memory"         # memory / milvus / hybrid
  similarity_threshold: 0.8      # 命中阈值（越高越严格）
  max_entries: 1000              # 最大条目数（仅 memory 后端）
  ttl_seconds: 3600              # 过期时间
  eviction_policy: "fifo"        # 淘汰策略
  use_hnsw: true                 # HNSW 索引加速
  hnsw_m: 16                     # 双向链接数
  hnsw_ef_construction: 200      # 构建参数
```

#### tools — 工具自动选择

```yaml
tools:
  enabled: false
  top_k: 3                       # 返回 top-k 匹配
  similarity_threshold: 0.2
  tools_db_path: "config/tools_db.json"
  fallback_to_empty: true        # 无匹配时返回空列表
```

#### prompt_guard — 越狱检测

```yaml
prompt_guard:
  enabled: true                  # 全局开关
  threshold: 0.7                 # 检测阈值
  use_cpu: true
```

#### classifier — 意图分类与 PII 检测

```yaml
classifier:
  category_model:                # 意图/领域分类
    threshold: 0.5
    use_cpu: true
  pii_model:                     # PII 检测
    threshold: 0.9               # 较高阈值，减少误报
    use_cpu: true
```

#### hallucination_mitigation — 幻觉检测

```yaml
hallucination_mitigation:
  enabled: false                 # 默认关闭
  fact_check_model:              # 事实检查分类器
    threshold: 0.6
  hallucination_model:           # 幻觉检测器
    threshold: 0.8
    min_span_length: 2           # 最小 token 跨度
    enable_nli_filtering: true   # NLI 过滤误报
  nli_model:                     # NLI 解释模型
    threshold: 0.9
```

#### feedback_detector — 用户反馈检测

```yaml
feedback_detector:
  enabled: true
  threshold: 0.7
```

检测 4 种反馈：satisfied / need_clarification / wrong_answer / want_different。

#### external_models — 外部 LLM（高级功能）

用于路由器内部高级功能（非用户请求路由），默认未启用。

```yaml
# external_models:
#   - llm_provider: "vllm"
#     model_role: "memory_rewrite"
#     llm_endpoint:
#       address: "localhost"
#       port: 8000
#     llm_model_name: "qwen3"
#     llm_timeout_seconds: 30
#     max_tokens: 100
#     temperature: 0.1
```

| model_role | 用途 |
|---|---|
| `preference` | 基于偏好的路由决策 |
| `memory_rewrite` | 记忆搜索的查询改写 |
| `memory_extraction` | 从对话中提取事实 |
| `guardrail` | 外部安全护栏 |

#### embedding_models — 嵌入模型

```yaml
embedding_models:
  mmbert_model_path: "models/mom-embedding-ultra"
  # qwen3_model_path: "models/mom-embedding-pro"     # 1024-dim, 32K ctx
  # gemma_model_path: "models/mom-embedding-flash"    # 768-dim, 8K ctx
  use_cpu: true
  hnsw_config:
    model_type: "mmbert"            # qwen3 / gemma / mmbert / multimodal
    preload_embeddings: true
    target_dimension: 768           # qwen3=1024, gemma/mmbert=768, multimodal=384
    target_layer: 22                # mmbert 层级早退 (3/6/11/22)
```

被语义缓存、工具选择、嵌入信号、复杂度信号等功能共享。

#### observability — 可观测性

```yaml
observability:
  metrics:
    enabled: true                   # Prometheus /metrics
  tracing:
    enabled: true                   # 分布式追踪
    provider: "opentelemetry"       # opentelemetry / openinference / openllmetry
    exporter:
      type: "otlp"
      endpoint: "vllm-sr-jaeger:4317"
      insecure: true
    sampling:
      type: "always_on"            # always_on / always_off / probabilistic
      rate: 1.0
```

#### looper — 多模型编排

```yaml
looper:
  enabled: true
  endpoint: "http://localhost:8899/v1/chat/completions"   # 指向 Envoy
  timeout_seconds: 1200
```

decision 有多个 modelRefs 时，looper 负责多模型编排（如 confidence routing）。

#### model_selection — ML 模型选择

```yaml
model_selection:
  enabled: true
  default_algorithm: "knn"          # knn / kmeans / svm
  knn:
    k: 5
    weights: "distance"             # uniform / distance
    metric: "cosine"                # cosine / euclidean / manhattan
  kmeans:
    num_clusters: 8
    efficiency_weight: 0.3          # 0=纯质量, 1=纯效率
  svm:
    kernel: "rbf"                   # linear / rbf / poly
    c: 1.0
    gamma: "auto"
```

---

## 三、Keyword Routing 实践

参考教程：[Keyword Based Routing](https://vllm-semantic-router.com/docs/tutorials/intelligent-route/keyword-routing)

Keyword Routing 通过关键词匹配实现透明、可审计的路由决策，无需 ML 推理，延迟亚毫秒级。

### config.yaml 示例

```yaml
version: v0.1

listeners:
  - name: "http-8899"
    address: "0.0.0.0"
    port: 8899
    timeout: "300s"

providers:
  models:
    - name: "minimax/minimax-m2.5"
      access_key: "<你的 OpenRouter API Key>"
      endpoints:
        - name: "openrouter"
          weight: 1
          endpoint: "openrouter.ai:443/api"
          protocol: "https"
  default_model: "minimax/minimax-m2.5"

# 关键词信号定义
signals:
  keywords:
    - name: "urgent_keywords"
      operator: "OR"                  # 匹配任一关键词
      keywords: ["urgent", "immediate", "asap", "emergency"]
      case_sensitive: false

    - name: "sensitive_data_keywords"
      operator: "OR"
      keywords: ["SSN", "social security", "credit card", "password"]
      case_sensitive: false

    - name: "spam_keywords"
      operator: "OR"
      keywords: ["buy now", "free money", "click here"]
      case_sensitive: false

decisions:
  - name: "urgent_request"
    description: "Route urgent requests to priority handling"
    priority: 200                     # 高优先级，优先匹配
    rules:
      operator: "OR"
      conditions:
        - type: "keyword"            # 使用 keyword 信号
          name: "urgent_keywords"    # 引用上面定义的信号名
    modelRefs:
      - model: "minimax/minimax-m2.5"
        use_reasoning: false
    plugins:
      - type: "system_prompt"
        configuration:
          system_prompt: "You are a highly responsive assistant specialized in handling urgent requests."

  - name: "sensitive_data"
    description: "Route sensitive data queries with extra security"
    priority: 190
    rules:
      operator: "OR"
      conditions:
        - type: "keyword"
          name: "sensitive_data_keywords"
    modelRefs:
      - model: "minimax/minimax-m2.5"
        use_reasoning: false
    plugins:
      - type: "system_prompt"
        configuration:
          system_prompt: "You are a security-conscious assistant. Exercise extreme caution with personal information."

  - name: "filter_spam"
    description: "Handle spam-like queries"
    priority: 180
    rules:
      operator: "OR"
      conditions:
        - type: "keyword"
          name: "spam_keywords"
    modelRefs:
      - model: "minimax/minimax-m2.5"
        use_reasoning: false
    plugins:
      - type: "system_prompt"
        configuration:
          system_prompt: "This query appears to be spam. Please provide a polite response declining the request."

  - name: "default-route"
    description: "Catch-all route for general requests"
    priority: 50                      # 最低优先级，兜底
    rules:
      operator: "AND"
      conditions: []
    modelRefs:
      - model: "minimax/minimax-m2.5"
        use_reasoning: false
```

### 关键词匹配方法

| 方法 | 适用场景 | 容错拼写 | 额外配置 |
|---|---|---|---|
| `regex`（默认） | 精确匹配、正则表达式 | 否 | — |
| `bm25` | 大关键词列表的主题检测（TF-IDF） | 否 | `bm25_threshold` |
| `ngram` | 模糊匹配，容忍拼写错误 | 是 | `ngram_threshold`, `ngram_arity` |

```yaml
signals:
  keywords:
    # BM25 方法
    - name: "code_keywords"
      operator: "OR"
      method: "bm25"
      keywords: ["code", "function", "debug", "algorithm", "refactor"]
      bm25_threshold: 0.1

    # N-gram 方法（容忍拼写错误，如 "urgnt" → "urgent"）
    - name: "urgent_keywords"
      operator: "OR"
      method: "ngram"
      keywords: ["urgent", "immediate", "asap", "emergency"]
      ngram_threshold: 0.4
      ngram_arity: 3

    # Regex 方法（默认）
    - name: "sensitive_data_keywords"
      operator: "OR"
      keywords: ["SSN", "credit card"]
```

### 逻辑运算符

| 运算符 | 含义 |
|---|---|
| `OR` | 匹配任一关键词即触发 |
| `AND` | 必须匹配所有关键词 |
| `NOR` | 关键词都不出现时触发（排除规则） |

### 启动与测试

```bash
vllm-sr serve \
--config new-config/keyword-config.yaml \
--image ghcr.io/vllm-project/semantic-router/vllm-sr:latest \
--image-pull-policy ifnotpresent
```

等待 Router 启动后（约 50 秒），测试关键词路由：

```bash
# 测试 1: 紧急请求（匹配 "urgent" → urgent_request decision）
curl -X POST http://localhost:8899/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "MoM",
    "messages": [{"role": "user", "content": "I need urgent help with my account"}]
  }'     

# 测试 2: 敏感数据（匹配 "SSN" + "credit card" → sensitive_data decision）
curl -X POST http://localhost:8801/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "MoM",
    "messages": [{"role": "user", "content": "My SSN and credit card were stolen"}]
  }'

# 测试 3: 垃圾信息（匹配 "free money" + "click here" → filter_spam decision）
curl -X POST http://localhost:8801/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "MoM",
    "messages": [{"role": "user", "content": "Click here to get free money now!"}]
  }'

# 测试 4: 普通请求（无关键词匹配 → default-route）
curl -X POST http://localhost:8801/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "MoM",
    "messages": [{"role": "user", "content": "What is the capital of France?"}]
  }'
```

### 验证路由日志

```bash
# 查看关键词匹配和路由决策
docker logs vllm-sr-container 2>&1 | grep -E "keyword|routing_decision|Decision.*matched"
```

示例日志输出：

```
Signal evaluation results: keyword=[sensitive_data_keywords], ...
Decision evaluation result: decision=sensitive_data, confidence=1.000, matched_rules=[keyword:sensitive_data_keywords], matched_keywords=[SSN credit card]
routing_decision: selected_model=minimax/minimax-m2.5, decision=sensitive_data
```

### 注意事项

- `model` 字段使用 `"MoM"`（Mixture-of-Models），由路由器根据 decision 自动选择实际模型
- keyword decision 的 `priority` 应高于 default-route，确保关键词规则优先匹配
- `config.yaml` 的 `decisions.plugins` 支持的类型有限：`system_prompt`、`semantic-cache`、`header_mutation`、`hallucination`、`router_replay`、`memory`、`rag`、`fast_response`。**不支持 `pii` 和 `jailbreak`**（这两个由 `router-defaults.yaml` 的全局配置自动处理）

---

## 四、Preference Routing 实践

参考教程：<https://vllm-semantic-router.com/docs/tutorials/intelligent-route/preference-routing>

Preference routing 根据用户意图语义匹配路由规则，支持两种模式：

| 模式 | 配置 | 原理 | 适用场景 |
|------|------|------|---------|
| **External LLM** | `external_models` + `parser_type: json` | 调用外部 LLM 分析意图，返回 `{"route":"..."}` | 有本地 vLLM 服务（`/v1/chat/completions`） |
| **Contrastive (Embedding)** | `classifier.preference_model.use_contrastive: true` | 用本地 embedding 模型做语义相似度匹配 | 无本地 LLM，或使用 OpenRouter 等第三方 API |

> **注意**：External LLM 模式要求目标端点接受 `/v1/chat/completions` 路径（硬编码）。OpenRouter 的 API 路径是 `/api/v1/chat/completions`，而 `ClassifierVLLMEndpoint` 没有 `path` 字段，无法添加 `/api` 前缀。通过 Envoy (`localhost:8899`) 中转会导致 ExtProc 递归处理导致 gRPC 崩溃。因此使用 OpenRouter 时只能用 Contrastive 模式。

### Contrastive 模式配置

#### 1. router-defaults.yaml 添加 preference_model

在 `.vllm-sr/router-defaults.yaml` 的 `classifier` 节添加：

```yaml
classifier:
  category_model:
    # ... 保持不变
  pii_model:
    # ... 保持不变
  preference_model:
    use_contrastive: true
    embedding_model: "mmbert"    # 使用已加载的 mmbert embedding 模型
```

#### 2. config.yaml 定义 preference signals 和 decisions

```yaml
# config.yaml
version: v0.1

listeners:
  - name: "http-8899"
    address: "0.0.0.0"
    port: 8899
    timeout: "300s"

providers:
  models:
    - name: "minimax/minimax-m2.5"
      access_key: "sk-or-v1-your-key-here"
      endpoints:
        - name: "openrouter"
          weight: 1
          endpoint: "openrouter.ai:443/api"
          protocol: "https"
  default_model: "minimax/minimax-m2.5"
  # Contrastive 模式不需要 external_models

signals:
  keywords:
    - name: "urgent_keywords"
      operator: "OR"
      keywords: ["urgent", "immediate", "asap", "emergency"]
      case_sensitive: false

  preferences:
    - name: "code_generation"
      description: "Generating new code snippets, writing functions, creating classes, implementing algorithms"
    - name: "bug_fixing"
      description: "Identifying and fixing errors, debugging issues, troubleshooting code problems"
    - name: "code_review"
      description: "Reviewing code quality, suggesting improvements, analyzing best practices"

decisions:
  # Keyword decisions (priority 300) > Preference decisions (priority 200) > Default (priority 50)
  - name: "urgent_request"
    description: "Route urgent requests to priority handling"
    priority: 300
    rules:
      operator: "OR"
      conditions:
        - type: "keyword"
          name: "urgent_keywords"
    modelRefs:
      - model: "minimax/minimax-m2.5"
    plugins:
      - type: "system_prompt"
        configuration:
          system_prompt: "You are a highly responsive assistant. Prioritize speed."

  - name: "preference_code_generation"
    description: "Route code generation requests"
    priority: 200
    rules:
      operator: "AND"
      conditions:
        - type: "preference"
          name: "code_generation"
    modelRefs:
      - model: "minimax/minimax-m2.5"
    plugins:
      - type: "system_prompt"
        configuration:
          system_prompt: "You are an expert code generator."

  - name: "preference_bug_fixing"
    description: "Route bug fixing requests"
    priority: 200
    rules:
      operator: "AND"
      conditions:
        - type: "preference"
          name: "bug_fixing"
    modelRefs:
      - model: "minimax/minimax-m2.5"
    plugins:
      - type: "system_prompt"
        configuration:
          system_prompt: "You are an expert debugger."

  - name: "preference_code_review"
    description: "Route code review requests"
    priority: 200
    rules:
      operator: "AND"
      conditions:
        - type: "preference"
          name: "code_review"
    modelRefs:
      - model: "minimax/minimax-m2.5"
    plugins:
      - type: "system_prompt"
        configuration:
          system_prompt: "You are an expert code reviewer."

  - name: "default-route"
    description: "Catch-all route for general requests"
    priority: 50
    rules:
      operator: "AND"
      conditions: []
    modelRefs:
      - model: "minimax/minimax-m2.5"
```

#### 3. 启动

```bash
source vsr/bin/activate
vllm-sr serve --image semantic-router/vllm-sr:v0.2.0-0310 --image-pull-policy ifnotpresent
```

启动日志中确认 contrastive 模式初始化成功：

```
[Preference Contrastive] preloaded 3/3 example embeddings using model=mmbert in 162ms
Preference classifier initialized successfully with 3 routes
```

#### 4. 测试验证

**代码生成请求** -> 匹配 `preference_code_generation`：

```bash
curl -s http://localhost:8801/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"MoM","messages":[{"role":"user","content":"Write a Python function to calculate fibonacci numbers"}]}'
```

路由日志：
```
Preference contrastive classification: preference=code_generation, latency=0.062s
Signal evaluation results: keyword=[], preference=[code_generation]
decision=preference_code_generation, selected=minimax/minimax-m2.5
```

**Bug 修复请求** -> 匹配 `preference_bug_fixing`：

```bash
curl -s http://localhost:8899/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"MoM","messages":[{"role":"user","content":"My Python script crashes with IndexError when processing the list"}]}'
```

路由日志：
```
decision=preference_bug_fixing, confidence=1.000, matched_rules=[preference:bug_fixing]
```

**Keyword 优先级高于 Preference** -> 匹配 `urgent_request`（priority 300 > 200）：

```bash
curl -s http://localhost:8899/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"MoM","messages":[{"role":"user","content":"This is urgent! Fix the crash in production immediately"}]}'
```

路由日志：
```
Signal evaluation results: keyword=[urgent_keywords], preference=[code_review]
decision=urgent_request, confidence=1.000, matched_rules=[keyword:urgent_keywords], matched_keywords=[urgent]
```

> keyword（priority 300）和 preference（priority 200）同时触发时，高优先级的 keyword decision 胜出。

### Contrastive 模式的局限性

- **精度依赖 description**：contrastive 模式用 preference 的 `description` 文本做 embedding 匹配，description 越详细、区分度越高
- **没有「不匹配」**：与 External LLM 模式不同（LLM 可以返回 `{"route":"other"}`），contrastive 总会选择最相似的一个 preference
- **适合少量、差异明显的路由类别**：如"代码生成"vs"调试"vs"代码审查"这类差异较大的场景效果好

### External LLM 模式（需要本地 vLLM）

如果有本地运行的 vLLM 服务（监听 `/v1/chat/completions`），可以用 External LLM 模式获得更精确的意图分类：

```yaml
# config.yaml
providers:
  external_models:
    - role: "preference"
      provider: "vllm"
      endpoint: "localhost:8000"       # 本地 vLLM 服务地址
      model_name: "your-model-name"
      timeout_seconds: 30
      parser_type: "json"
      # access_key: ""                 # 如需鉴权
```

此时不需要在 `router-defaults.yaml` 中配置 `preference_model.use_contrastive`。

---

## 五、Preference Routing 遇到的问题与安装问题汇总

| # | 问题 | 原因 | 解决 |
|---|---|---|---|
| 1 | Go 版本不满足 (1.20 vs 要求 1.24.1) | Homebrew 安装的是旧版 go@1.20 | `brew upgrade go` |
| 2 | pip install 报 externally-managed-environment | macOS Homebrew Python 禁止全局安装 (PEP 668) | 创建 venv: `python3 -m venv vsr` |
| 3 | 模型下载报 huggingface-cli not found | router 启动时调用 huggingface-cli 下载模型 | `pip install "huggingface_hub[cli]"` |
| 4 | ImportError: socksio package not installed | 系统配了 SOCKS 代理，httpx 需要 socksio | `pip install "httpx[socks]"` |
| 5 | model.safetensors: No such file or directory | HF 下载的是软链接而非实际文件 | 确认模型目录下有实际的 .safetensors 文件 |
| 6 | address already in use (端口 50051/9190) | 残余 router 进程未退出 | `lsof -ti:50051 -ti:9190 -ti:8080 \| xargs kill -9` |
| 7 | ld: warning: search path not found | Go 编译时链接路径警告，不影响运行 | 运行时设置正确的 LD_LIBRARY_PATH |
| 8 | 改了 router-config.yaml 不生效 | Docker 方式下该文件是自动生成的 | 改 `config.yaml` 后重启 `vllm-sr serve` |
| 9 | OpenRouter 返回 404 | endpoint 路径写成 `/api/v1` 导致双重前缀 | 改为 `/api`（Envoy 会自动拼接 `/v1/...`） |
| 10 | endpoint 的 api_key/type 字段不生效 | Python CLI 的 Endpoint model 不支持这些字段 | `access_key` 放 model 级别，不要用 `type: openrouter` |
| 11 | decisions plugins 报 `pii` type 不支持 | v0.2.0 镜像的 config.yaml plugins 不支持 `pii`/`jailbreak` 类型 | 去掉 `pii` plugin，PII 检测由 `router-defaults.yaml` 全局配置自动处理 |
| 12 | Preference external LLM 调 OpenRouter 返回 HTML | `VLLMClient` 硬编码 `/v1/chat/completions`，OpenRouter 需要 `/api/v1/chat/completions`，`ClassifierVLLMEndpoint` 无 `path` 字段 | 改用 contrastive 模式（`classifier.preference_model.use_contrastive: true`），或搭本地 vLLM 服务 |
| 13 | Preference external LLM 通过 Envoy (localhost:8899) 返回 500 | preference 请求重入 ExtProc 导致 gRPC 递归崩溃 | 同上，不要通过 Envoy 中转 external_models 请求 |

---

## 六、调试技巧

```bash
# 查看 Envoy access log（确认请求是否到达后端、返回码）
docker exec vllm-sr-container tail -f /var/log/envoy_access.log

# 查看路由器日志（路由决策、模型选择、分类结果）
docker logs -f vllm-sr-container

# 查看实际生成的 Envoy 配置
docker exec vllm-sr-container cat /etc/envoy/envoy.yaml

# 查看生成的 router-config.yaml
cat .vllm-sr/router-config.yaml

# 杀掉残余进程（裸机方式）
lsof -ti:50051 -ti:9190 -ti:8080 | xargs kill -9
```
