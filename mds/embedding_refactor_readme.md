# Embedding Refactor Plan

## 1. Current Bottleneck Analysis

The current request path is only superficially concurrent. `EvaluateAllSignalsWithContext` launches signal evaluators in parallel, but every embedding-backed signal still performs a blocking in-process FFI call per text.

Current hot-path behavior:

- `EvaluateAllSignalsWithContext` starts one goroutine per enabled signal.
- `evaluateEmbeddingSignal` calls `EmbeddingClassifier.ClassifyAll`, which performs one blocking query embedding call.
- `evaluatePreferenceSignal` calls `PreferenceClassifier.Classify`, which in contrastive mode calls `ContrastivePreferenceClassifier.Classify`, which performs another blocking query embedding call for the same request text.
- Startup preload paths (`preloadCandidateEmbeddings`, `preloadRuleEmbeddings`, contrastive jailbreak preload, complexity preload) create many goroutines, but each goroutine still issues one blocking FFI embedding call.

Why this serializes in practice:

- Each embedding request is synchronous at the call site.
- There is no shared request queue across router requests or across signal goroutines.
- A single user request can trigger duplicate embeddings for the same text in different classifiers.
- Preload code uses CPU-worker fanout, not true embedding batching.

Net effect:

- Before: `N requests * M embedding-backed features -> N*M blocking embedding invocations`
- After refactor target: `N requests * M feature lookups -> enqueued into shared batchers -> fewer HTTP batch calls to the embedding service`

## 2. Embedding Call Sites

### 2.1 Primary routing and signal path

| File | Function | Current behavior | Replacement |
| --- | --- | --- | --- |
| `src/semantic-router/pkg/classification/classifier_signal_context.go` | `evaluateEmbeddingSignal` | Calls `c.keywordEmbeddingClassifier.ClassifyAll(text)` from the signal fanout path. | Keep this orchestration layer thin; inject an HTTP-backed embedder into `EmbeddingClassifier` so this call path stops using FFI indirectly. |
| `src/semantic-router/pkg/classification/embedding_classifier.go` | `preloadCandidateEmbeddings` | Preloads every unique candidate via worker pool + `getEmbeddingWithModelType`. | Replace worker-pool FFI loop with `EmbedBatch(ctx, candidates)` against the shared embedding client/batcher. |
| `src/semantic-router/pkg/classification/embedding_classifier.go` | `ClassifyAll` | Computes one query embedding via `getEmbeddingWithModelType`, then scores against cached candidate embeddings. | Replace query embedding call with `Embed(ctx, text)` on injected client. Keep similarity logic unchanged. |
| `src/semantic-router/pkg/classification/classifier_signal_context.go` | `evaluatePreferenceSignal` | Calls `c.preferenceClassifier.Classify(conversationJSON)`. In contrastive mode this becomes an embedding lookup. | Keep orchestration unchanged; inject the same client into `PreferenceClassifier` / `ContrastivePreferenceClassifier`. |
| `src/semantic-router/pkg/classification/contrastive_preference_classifier.go` | `preloadRuleEmbeddings` | Preloads rule descriptions/examples via worker pool + `getEmbeddingWithModelType`. | Replace with batched HTTP embedding preload. |
| `src/semantic-router/pkg/classification/contrastive_preference_classifier.go` | `Classify` | Computes query embedding via `getEmbeddingWithModelType`, then runs cosine similarity against preloaded rule embeddings. | Replace with `Embed(ctx, text)` on injected client. |
| `src/semantic-router/pkg/classification/complexity_classifier.go` | preload path for text candidates | Uses `getEmbeddingWithModelType` for text candidates and FFI multimodal calls for image branches. | Replace text candidate embedding generation with shared HTTP client. Keep multimodal image/text branches unchanged unless a separate multimodal HTTP service exists. |
| `src/semantic-router/pkg/classification/complexity_classifier.go` | `ClassifyWithImage` | Uses `getEmbeddingWithModelType` for query text, plus multimodal FFI for image-aware scoring. | Replace only the text query embedding with shared HTTP client. |
| `src/semantic-router/pkg/classification/contrastive_jailbreak_classifier.go` | `preloadKBEmbeddings` | Preloads jailbreak/benign KB embeddings via worker pool + `getEmbeddingWithModelType`. | Replace with `EmbedBatch`. |
| `src/semantic-router/pkg/classification/contrastive_jailbreak_classifier.go` | `AnalyzeMessages` | Computes one embedding per message via `getEmbeddingWithModelType`. | Replace with `Embed(ctx, msg)`; optionally batch multiple messages when `include_history` is enabled. |
| `src/semantic-router/pkg/classification/classifier_composers.go` | `GetQueryEmbedding` | Calls `candle_binding.GetEmbedding` directly for model-selection feature generation. | Replace with shared client or an adapter over it, then convert `[]float32` to `[]float64` as today. |
| `src/semantic-router/pkg/extproc/router_selection.go` | `createModelSelectorRegistry` | Injects `candle_binding.GetEmbeddingBatched` into selection factory. | Inject shared HTTP client instead, so RouterDC / ML selectors use the same embedding path as signal routing. |

### 2.2 Adjacent direct embedding generators under `src/semantic-router/pkg`

These are not the immediate signal bottleneck, but they are still direct FFI embedding users and should converge on the same client abstraction.

| File | Function | Current behavior | Replacement |
| --- | --- | --- | --- |
| `src/semantic-router/pkg/extproc/req_filter_rag_external.go` | `buildPineconeRequest`, `buildWeaviateRequest` | Calls `candle_binding.GetEmbedding` for query vectors before external vector search. | Replace with shared HTTP client. |
| `src/semantic-router/pkg/extproc/req_filter_rag_milvus.go` | `retrieveFromMilvus` | Calls `candle_binding.GetEmbedding` for query vectors. | Replace with shared HTTP client. |
| `src/semantic-router/pkg/vectorstore/candle_embedder.go` | `Embed` | Wrapper over multiple FFI embedding functions. | Replace `CandleEmbedder` with HTTP-backed embedder implementation behind the existing `Embedder` interface. |
| `src/semantic-router/pkg/memory/embedding.go` | `GenerateEmbedding` | Direct model-switching over FFI embedding functions. | Replace with model-bound HTTP client instances. |
| `src/semantic-router/pkg/cache/inmemory_cache.go` | `generateEmbedding` | Direct FFI embedding generation for semantic cache keys. | Replace with shared client. |
| `src/semantic-router/pkg/cache/redis_cache.go` | `getEmbedding` | Same as above for Redis cache. | Replace with shared client. |
| `src/semantic-router/pkg/cache/milvus_cache.go` | `getEmbedding` | Same as above for Milvus cache. | Replace with shared client. |
| `src/semantic-router/pkg/tools/tools.go` | `LoadToolsFromFile`, `AddTool`, `FindSimilarToolsWithScores` | Uses `GetEmbeddingWithModelType` for tool description/query embeddings. | Replace with shared client; preload path should use batch API. |
| `src/semantic-router/pkg/modelselection/trainer.go` | `generateCandleEmbedding` / `GetEmbedding` | Training/inference-time direct FFI embeddings. | Replace with shared client if training should match router inference; otherwise leave explicitly out of the first refactor. |
| `src/semantic-router/pkg/apiserver/route_embeddings.go` | `handleEmbeddings` | API endpoint directly proxies to FFI embedding functions. | Either route through the new HTTP client or leave as an explicit compatibility endpoint. If kept, it should not remain the only direct FFI island. |

### 2.3 Test seam already present

- `src/semantic-router/pkg/classification/embedder_hook.go`
  - Current role: swaps `getEmbeddingWithModelType` in tests.
  - Refactor role: replace with an interface-based test stub for the new embedding client.

## 3. Refactor Design

### 3.1 Embedding Client

Add a shared package, not more logic inside `classification`:

- `src/semantic-router/pkg/embedding/client.go`
- `src/semantic-router/pkg/embedding/http_client.go`
- `src/semantic-router/pkg/embedding/batcher.go`

Recommended interface:

```go
type Client interface {
    Embed(ctx context.Context, text string) ([]float32, error)
    EmbedBatch(ctx context.Context, texts []string) ([][]float32, error)
}
```

Design notes:

- One client instance should be bound to one embedding service endpoint and one model/dimension profile.
- Model selection should happen at construction time, not per request.
- For qwen3/gemma/mmbert, create separate configured clients as needed.
- Keep multimodal image encoding out of this first TEI refactor unless there is a real HTTP multimodal embedding service available.

HTTP behavior:

- Request body: one text or a text array, depending on endpoint capability.
- Response contract: preserve input order so batch result `i` maps to request text `i`.
- Context: every call must accept caller `context.Context`; use `http.NewRequestWithContext`.
- Timeout: client should also enforce a hard upper bound from config even if caller forgets to set one.

Error handling strategy:

- Transport failure / non-2xx / invalid JSON: return a wrapped client error and fail the whole batch.
- Timeout / context cancellation: return `context.DeadlineExceeded` or `context.Canceled` wrapped with endpoint/model metadata.
- Empty embedding or wrong dimension: treat as hard error.
- For the initial minimal refactor, assume batch responses are all-or-nothing. Do not invent partial-success semantics unless the HTTP service really supports them.

Initialization and dependency injection:

- Add a new embedding-service config under `EmbeddingModels`.
- Initialize the HTTP client and its batching wrapper once during router/classifier bootstrap.
- Inject the same shared client into:
  - `EmbeddingClassifier`
  - `ContrastivePreferenceClassifier`
  - `ContrastiveJailbreakClassifier`
  - `ComplexityClassifier` for text embeddings
  - selection factory in `extproc/router_selection.go`
- Keep `EvaluateAllSignalsWithContext` as orchestration; do not let it own HTTP client creation.

Suggested config addition:

- File: `src/semantic-router/pkg/config/model_config_types.go`
- Add `EmbeddingServiceConfig` under `EmbeddingModels`
- Fields:
  - `endpoint`
  - `timeout_seconds`
  - `max_batch_size`
  - `max_wait_ms`
  - `queue_capacity`
  - `max_concurrent_flushes`

### 3.2 Batching Mechanism

#### Correct integration point

Primary integration point: classifier layer, not `EvaluateAllSignalsWithContext`.

Reason:

- `EvaluateAllSignalsWithContext` only sees one request’s signal fanout.
- The real batching gain comes from aggregating embedding work across many requests and across multiple embedding-backed classifiers.
- The classifier layer owns the embedding semantics already (`ClassifyAll`, contrastive preference, contrastive jailbreak, complexity preload/query).
- The same batcher can then also be reused by selection, RAG, cache, tools, and vector-store code.

`EvaluateAllSignalsWithContext` should stay mostly unchanged, except that its embedding-backed downstream calls should now use context-aware client methods.

#### Queue structure

Use one batcher per model profile:

```go
type embedJob struct {
    ctx    context.Context
    text   string
    respCh chan embedResult
}

type embedResult struct {
    embedding []float32
    err       error
}
```

Implementation model:

- Producers: request goroutines call `Client.Embed(ctx, text)`; the batching wrapper enqueues an `embedJob`.
- Aggregator: one long-lived goroutine collects jobs into a pending slice.
- Flush worker: sends one HTTP batch request for the current slice.
- Concurrency control: allow a small bounded number of concurrent flushes with a semaphore; do not allow unbounded parallel HTTP bursts.

Flush conditions:

- Flush immediately when `len(pending) >= max_batch_size`
- Flush on timer when oldest pending item has waited `>= max_wait_ms`
- Flush on shutdown / context cancellation for remaining jobs

Mapping results back:

- Preserve the queue order in the outgoing batch payload.
- When the HTTP response returns `[][]float32`, map result `i` back to pending job `i`.
- If the batch call fails, send the same error to every job in that flush.

Why this is optimal:

- It captures same-request duplication automatically when embedding and preference signals race concurrently.
- It captures cross-request traffic, which is where throughput gains come from.
- It avoids pushing batching logic into every classifier separately.
- It keeps similarity scoring local and unchanged; only embedding generation is refactored.

## 4. Code Change Plan

### New files

File: `src/semantic-router/pkg/embedding/client.go`  
Change Type: ADD

Details:
- Add the shared `Client` interface.
- Add small config/request helpers if needed.
- Add exported constructor signatures for direct HTTP and batched wrappers.

File: `src/semantic-router/pkg/embedding/http_client.go`  
Change Type: ADD

Details:
- Implement HTTP calls to the embedding service.
- Decode ordered batch responses.
- Centralize timeout, status-code, and payload validation.

File: `src/semantic-router/pkg/embedding/batcher.go`  
Change Type: ADD

Details:
- Implement queue, timer-based flush, bounded backlog, and result fanout.
- Wrap the base HTTP client so existing callers can continue to ask for single embeddings.

### Existing files to modify

File: `src/semantic-router/pkg/config/model_config_types.go`  
Change Type: MODIFY

Details:
- Add embedding-service config for endpoint and batching knobs.
- Keep existing `HNSWConfig` for matching/scoring behavior; do not overload it with transport details.

File: `src/semantic-router/pkg/classification/classifier.go`  
Change Type: MODIFY

Details:
- Add an embedding client dependency field to `Classifier`.
- Update option wiring so embedding-backed classifiers receive the shared client.

File: `src/semantic-router/pkg/classification/classifier_construction.go`  
Change Type: MODIFY

Details:
- Initialize embedding-backed classifiers with the injected client instead of relying on package-level FFI helpers.
- Keep multimodal initialization only for multimodal/image paths that remain on FFI.

File: `src/semantic-router/pkg/classification/embedding_classifier.go`  
Change Type: MODIFY

Details:
- Delete direct use of `getEmbeddingWithModelType` for candidate preload and query embedding.
- Add an `embedding.Client` field.
- Refactor `preloadCandidateEmbeddings` to use `EmbedBatch`.
- Refactor `ClassifyAll` to use `Embed`.
- Keep similarity scoring and rule aggregation unchanged.

File: `src/semantic-router/pkg/classification/contrastive_preference_classifier.go`  
Change Type: MODIFY

Details:
- Delete direct `getEmbeddingWithModelType` usage.
- Add injected client.
- Refactor rule preload to batch.
- Refactor query embedding to single `Embed`.

File: `src/semantic-router/pkg/classification/preference_classifier.go`  
Change Type: MODIFY

Details:
- Pass the injected client into `NewContrastivePreferenceClassifier`.
- Keep external LLM preference mode unchanged.

File: `src/semantic-router/pkg/classification/contrastive_jailbreak_classifier.go`  
Change Type: MODIFY

Details:
- Delete direct text embedding FFI calls.
- Use shared client for preload and per-message analysis.

File: `src/semantic-router/pkg/classification/complexity_classifier.go`  
Change Type: MODIFY

Details:
- Replace only text embedding calls with shared client.
- Keep `getMultiModalTextEmbedding` / `getMultiModalImageEmbedding` intact unless a multimodal HTTP endpoint exists.

File: `src/semantic-router/pkg/classification/classifier_signal_context.go`  
Change Type: MODIFY

Details:
- Minimal change only.
- If context threading is added, pass request-scoped contexts to embedding-backed classifier calls.
- Do not add batching logic here.

File: `src/semantic-router/pkg/classification/classifier_composers.go`  
Change Type: MODIFY

Details:
- Replace `GetQueryEmbedding` direct FFI usage with shared client.

File: `src/semantic-router/pkg/classification/embedder_hook.go`  
Change Type: MODIFY

Details:
- Remove package-level FFI override pattern for text embeddings.
- Replace with client-interface test injection.

File: `src/semantic-router/pkg/extproc/router_components.go`  
Change Type: MODIFY

Details:
- Construct the shared HTTP embedding client + batcher once.
- Inject it into classifier creation and any other runtime component that needs embeddings.

File: `src/semantic-router/pkg/extproc/router_selection.go`  
Change Type: MODIFY

Details:
- Replace inline `candle_binding.GetEmbeddingBatched` closure with the shared client.

### Follow-on consumers to convert after the hot path

File: `src/semantic-router/pkg/extproc/req_filter_rag_external.go`  
Change Type: MODIFY

Details:
- Replace direct `GetEmbedding` calls for Pinecone/Weaviate query vectors.

File: `src/semantic-router/pkg/extproc/req_filter_rag_milvus.go`  
Change Type: MODIFY

Details:
- Replace direct `GetEmbedding` call for Milvus query vectors.

File: `src/semantic-router/pkg/vectorstore/candle_embedder.go`  
Change Type: MODIFY or REMOVE

Details:
- Replace Candle-backed implementation with HTTP-backed implementation behind the same `Embedder` interface.

File: `src/semantic-router/pkg/cache/inmemory_cache.go`  
Change Type: MODIFY

Details:
- Replace direct embedding generation with shared client.

File: `src/semantic-router/pkg/cache/redis_cache.go`  
Change Type: MODIFY

Details:
- Replace direct embedding generation with shared client.

File: `src/semantic-router/pkg/cache/milvus_cache.go`  
Change Type: MODIFY

Details:
- Replace direct embedding generation with shared client.

File: `src/semantic-router/pkg/memory/embedding.go`  
Change Type: MODIFY

Details:
- Replace direct model-switching FFI implementation with HTTP-backed clients.

File: `src/semantic-router/pkg/tools/tools.go`  
Change Type: MODIFY

Details:
- Replace description/query embedding generation with shared client.
- Convert preload path to batch mode.

## 5. Concurrency Model Before vs After

### Before

- Router request enters `EvaluateAllSignalsWithContext`.
- Signal goroutines start in parallel.
- Each embedding-backed classifier issues its own blocking FFI embedding call.
- Startup preload loops use many goroutines, but each goroutine still performs one embedding call at a time.
- There is no shared queue, no cross-request aggregation, and no reuse across embedding consumers.

### After

- Router request enters `EvaluateAllSignalsWithContext`.
- Signal goroutines still start in parallel.
- Any embedding-backed component calls `Embed(ctx, text)` on the shared batching client.
- The batching client aggregates pending jobs from many goroutines and many requests.
- Flushes become fewer, larger HTTP batch calls to the embedding service.
- Similarity search and decision logic stay local and unchanged.

Impact:

- Parallelism improves because request goroutines stop performing direct in-process embedding work.
- Throughput improves because the embedding service receives batches instead of singles.
- Per-request latency usually improves under load because queueing plus batching amortizes model execution cost.
- Startup preload also improves because static candidate/example lists can be embedded in larger batches instead of `runtime.NumCPU()*2` single calls.

## 6. Risks and Mitigations

### Timeout handling

- Risk: a stuck embedding service stalls signal evaluation.
- Mitigation: require `context.Context` on all client calls and enforce client-side hard timeouts from config.

### Partial batch failures

- Risk: one bad batch response breaks many pending requests.
- Mitigation: for the first version, treat batch failures as all-or-nothing and fail every job in that flush explicitly; add per-item handling only if the service supports it.

### Service unavailable

- Risk: HTTP embedding endpoint outage disables embedding-backed routing.
- Mitigation: clear error propagation, metrics around batch failures, and optional fallback policy only where behavior is acceptable. Do not silently fall back to unrelated embeddings.

### Backpressure

- Risk: request spikes create an unbounded queue and memory growth.
- Mitigation: bounded queue capacity; if full, return a fast overload error instead of buffering indefinitely.

### Duplicate same-request work

- Risk: embedding and contrastive preference still request the same text separately.
- Mitigation: the shared batcher reduces the cost immediately; request-scoped dedup can be a later optimization, not a prerequisite for this refactor.

### Multimodal branches

- Risk: complexity/image paths still depend on FFI.
- Mitigation: keep multimodal text/image encoding explicitly out of the TEI text-embedding refactor unless there is a real HTTP replacement. Replace only text embedding calls in phase 1.

### Bootstrap/config drift

- Risk: some components keep using direct FFI while others move to HTTP.
- Mitigation: establish `pkg/embedding` as the single text-embedding abstraction and convert all direct text embedding call sites listed above to it.

## Notes

- The repository harness command `make agent-report ENV=cpu CHANGED_FILES="mds/embedding_refactor_readme.md"` could not be completed in this environment because the local Python dependency `yaml` is missing. This plan is based on direct source inspection instead.
