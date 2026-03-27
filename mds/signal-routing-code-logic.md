# 信号路由关键代码提取与逻辑整理

## 1) 代码位置映射

| 功能 | 你给出的定位 | 当前定位（仓库实际） |
|---|---|---|
| 信号并行评估入口 | `classification/classifier.go:1245` `EvaluateAllSignalsWithContext()` | `src/semantic-router/pkg/classification/classifier_signal_context.go` `EvaluateAllSignalsWithContext()` |
| Keyword 正则匹配 | `classification/keyword_classifier.go:259` `ClassifyWithKeywordsAndCount()` | `src/semantic-router/pkg/classification/keyword_classifier.go` `ClassifyWithKeywordsAndCount()` |
| Preference Contrastive 推理 | `classification/contrastive_preference_classifier.go:151` `Classify()` | `src/semantic-router/pkg/classification/contrastive_preference_classifier.go` `Classify()` |
| Decision 优先级排序 | `decision/engine.go:321` `selectBestDecision()` | `src/semantic-router/pkg/decision/engine.go` `selectBestDecision()` |
| 信号是否启用判断 | `classification/classifier.go:1275` `isSignalTypeUsed()` | `src/semantic-router/pkg/classification/classifier_signal_eval.go` `isSignalTypeUsed()` |
| Preference 外部 LLM 调用 | `classification/preference_classifier.go:106` `Classify()` | `src/semantic-router/pkg/classification/preference_classifier.go` `Classify()`（`VLLMClient.Generate`） |
| 路由评估总入口 | `extproc/req_filter_classification.go:22` `performDecisionEvaluation()` | `src/semantic-router/pkg/extproc/req_filter_classification.go` `performDecisionEvaluation()` |
| 指标记录 | `extproc/req_filter_classification.go:106` `metrics.RecordSignalExtraction()` | 调用在 `src/semantic-router/pkg/classification/classifier_signal_context.go`；定义在 `src/semantic-router/pkg/observability/metrics/signal_decision_plugin_metrics.go` |

---

## 2) 关键代码提取

### A. 信号并行评估入口：`EvaluateAllSignalsWithContext`

```go
func (c *Classifier) EvaluateAllSignalsWithContext(text string, contextText string, nonUserMessages []string, forceEvaluateAll bool, uncompressedText string, skipCompressionSignals map[string]bool, imageURL ...string) *SignalResults {
	defer c.enterSignalEvaluationLoadGate()()
	var usedSignals map[string]bool
	if forceEvaluateAll {
		usedSignals = c.getAllSignalTypes()
		logging.Debugf("[Signal Computation] Force evaluate all signals mode enabled")
	} else {
		usedSignals = c.getUsedSignals()
	}

	textForSignal := textForSignalFunc(text, uncompressedText, skipCompressionSignals)
	ready := c.signalReadiness()

	results := &SignalResults{
		Metrics:           &SignalMetricsCollection{},
		SignalConfidences: make(map[string]float64),
		SignalValues:      make(map[string]float64),
	}

	var wg sync.WaitGroup
	var mu sync.Mutex
	imgArg := ""
	if len(imageURL) > 0 {
		imgArg = imageURL[0]
	}

	dispatchers := c.buildSignalDispatchers(results, &mu, textForSignal, contextText, nonUserMessages, imgArg)
	runSignalDispatchers(dispatchers, usedSignals, ready, &wg)

	wg.Wait()
	results = c.applySignalGroups(results)
	results = c.applySignalComposers(results)
	results = c.applyProjections(results)
	return results
}
```

### B. Keyword 正则匹配主流程：`ClassifyWithKeywordsAndCount`

```go
func (c *KeywordClassifier) ClassifyWithKeywordsAndCount(text string) (string, []string, int, int, error) {
	regexIdx := 0

	for _, ref := range c.ruleOrder {
		switch ref.method {
		case "bm25":
			result := c.bm25Classifier.Classify(text)
			if result.Matched && result.RuleName == ref.name {
				return result.RuleName, result.MatchedKeywords, result.MatchCount, result.TotalKeywords, nil
			}

		case "ngram":
			result := c.ngramClassifier.Classify(text)
			if result.Matched && result.RuleName == ref.name {
				return result.RuleName, result.MatchedKeywords, result.MatchCount, result.TotalKeywords, nil
			}

		case "regex":
			if regexIdx < len(c.regexRules) {
				rule := c.regexRules[regexIdx]
				regexIdx++
				matched, keywords, matchCount, err := c.matchesWithCount(text, rule)
				if err != nil {
					return "", nil, 0, 0, err
				}
				if matched {
					totalKeywords := len(rule.OriginalKeywords)
					return rule.Name, keywords, matchCount, totalKeywords, nil
				}
			}
		}
	}
	return "", nil, 0, 0, nil
}
```

### C. Preference Contrastive 推理（余弦相似度）：`ContrastivePreferenceClassifier.Classify`

```go
func (c *ContrastivePreferenceClassifier) Classify(text string) (*PreferenceResult, error) {
	if strings.TrimSpace(text) == "" {
		return nil, fmt.Errorf("text is empty")
	}

	c.mu.RLock()
	if len(c.ruleEmbeddings) == 0 {
		c.mu.RUnlock()
		return nil, fmt.Errorf("no embeddings loaded for contrastive preference classifier")
	}
	c.mu.RUnlock()

	out, err := getEmbeddingWithModelType(text, c.modelType, 0)
	if err != nil {
		return nil, fmt.Errorf("failed to embed query: %w", err)
	}
	queryEmbedding := out.Embedding

	var (
		bestRule  string
		bestScore float32 = -1
	)

	c.mu.RLock()
	defer c.mu.RUnlock()

	for ruleName, embeddings := range c.ruleEmbeddings {
		if len(embeddings) == 0 {
			continue
		}
		var maxSim float32
		for _, emb := range embeddings {
			sim := cosineSimilarity(queryEmbedding, emb)
			if sim > maxSim {
				maxSim = sim
			}
		}
		if maxSim > bestScore {
			bestScore = maxSim
			bestRule = ruleName
		}
	}

	if bestRule == "" {
		return nil, fmt.Errorf("no preference matched by contrastive classifier")
	}
	threshold := c.ruleThresholds[bestRule]
	if threshold > 0 && bestScore < threshold {
		return nil, fmt.Errorf("preference similarity %.3f below threshold %.3f", bestScore, threshold)
	}
	return &PreferenceResult{Preference: bestRule, Confidence: bestScore}, nil
}
```

### D. Decision 优先级排序：`selectBestDecision`

```go
func (e *DecisionEngine) selectBestDecision(results []DecisionResult) *DecisionResult {
	if len(results) == 0 {
		return nil
	}
	if len(results) == 1 {
		return &results[0]
	}

	useTieredSelection := e.useTieredSelection(results)
	sort.Slice(results, func(i, j int) bool {
		return e.decisionResultLess(results[i], results[j], useTieredSelection)
	})
	return &results[0]
}
```

`decisionResultLess()` 的关键排序规则（简化）：

```go
if useTieredSelection {
	// Tier 小者优先；同 Tier 下 confidence 高者优先；再按 priority
}
if e.strategy == "confidence" {
	// confidence 优先，再 priority
}
// 默认 priority 优先，再 confidence
```

### E. 信号是否启用判断：`isSignalTypeUsed`

```go
func isSignalTypeUsed(usedSignals map[string]bool, signalType string) bool {
	normalizedType := strings.ToLower(strings.TrimSpace(signalType))
	prefix := normalizedType + ":"

	for key := range usedSignals {
		if strings.HasPrefix(strings.ToLower(strings.TrimSpace(key)), prefix) {
			return true
		}
	}
	return false
}
```

### F. Preference 外部 LLM 调用：`PreferenceClassifier.Classify`（VLLMClient）

```go
type PreferenceClassifier struct {
	client *VLLMClient
	...
}

func (p *PreferenceClassifier) Classify(conversationJSON string) (*PreferenceResult, error) {
	ctx, cancel := context.WithTimeout(context.Background(), p.timeout)
	defer cancel()

	if p.useContrastive {
		return p.classifyContrastive(conversationJSON)
	}

	routesJSON, err := p.buildRoutesJSON()
	if err != nil { return nil, err }

	userPrompt := fmt.Sprintf(p.userPromptTemplate, routesJSON, conversationJSON)

	resp, err := p.client.Generate(ctx, p.modelName, userPrompt, &GenerationOptions{
		MaxTokens:   1000,
		Temperature: 0.0,
	})
	if err != nil {
		return nil, fmt.Errorf("external LLM API call failed: %w", err)
	}

	if len(resp.Choices) == 0 {
		return nil, fmt.Errorf("no choices in LLM response")
	}

	output := resp.Choices[0].Message.Content
	return p.parsePreferenceOutput(output)
}
```

### G. 路由评估总入口：`performDecisionEvaluation`

```go
func (r *OpenAIRouter) performDecisionEvaluation(originalModel string, userContent string, nonUserMessages []string, ctx *RequestContext) (string, float64, entropy.ReasoningDecision, string, error) {
	if len(nonUserMessages) == 0 && userContent == "" {
		return "", 0.0, entropy.ReasoningDecision{}, "", nil
	}

	if len(r.Config.Decisions) == 0 {
		if r.Config.IsAutoModelName(originalModel) {
			return "", 0.0, entropy.ReasoningDecision{}, r.Config.DefaultModel, nil
		}
		return "", 0.0, entropy.ReasoningDecision{}, "", nil
	}

	signalInput := r.prepareSignalEvaluationInput(userContent, nonUserMessages)
	if signalInput.evaluationText == "" {
		return "", 0.0, entropy.ReasoningDecision{}, "", nil
	}

	signals, authzErr := r.evaluateSignalsForDecision(originalModel, signalInput, nonUserMessages, ctx)
	if authzErr != nil {
		return "", 0, entropy.ReasoningDecision{}, "", authzErr
	}

	result, fallbackModel := r.runDecisionEngine(originalModel, ctx, signals)
	if result == nil {
		return "", 0.0, entropy.ReasoningDecision{}, fallbackModel, nil
	}

	decisionName, evaluationConfidence, reasoningDecision, selectedModel := r.finalizeDecisionEvaluation(result, originalModel, userContent, ctx)
	return decisionName, evaluationConfidence, reasoningDecision, selectedModel, nil
}
```

### H. 指标记录：`metrics.RecordSignalExtraction`

> 你给出的路径 `extproc/req_filter_classification.go:106` 在当前代码中不存在该调用。
> 当前调用主要出现在 signal 分类执行处（如 keyword/domain/preference 等），定义在 metrics 包。

`RecordSignalExtraction` 定义：

```go
func RecordSignalExtraction(signalType, signalName string, latencySeconds float64) {
	if signalType == "" {
		signalType = consts.UnknownLabel
	}
	if signalName == "" {
		signalName = consts.UnknownLabel
	}
	SignalExtractionTotal.WithLabelValues(signalType, signalName).Inc()
	SignalExtractionLatency.WithLabelValues(signalType).Observe(latencySeconds)
}
```

示例调用（keyword 信号）：

```go
func (c *Classifier) evaluateKeywordSignal(results *SignalResults, mu *sync.Mutex, text string) {
	start := time.Now()
	category, keywords, err := c.keywordClassifier.ClassifyWithKeywords(text)
	elapsed := time.Since(start)
	latencySeconds := elapsed.Seconds()

	metrics.RecordSignalExtraction(config.SignalTypeKeyword, category, latencySeconds)
	...
}
```

---

## 3) 逻辑总链路（从请求到路由）

1. `extproc.performDecisionEvaluation()` 作为总入口，先做空输入/无决策配置的快速返回。  
2. 进入 `evaluateSignalsForDecision()` 后，`Classifier.EvaluateAllSignalsWithContext()` 并发计算各类信号。  
3. 其中关键词信号通过 `ClassifyWithKeywordsAndCount()`（bm25/ngram/regex）执行 first-match。  
4. Preference 信号有两条路：  
   - contrastive：`ContrastivePreferenceClassifier.Classify()`，核心是 query embedding 与规则样本 embedding 的**最大余弦相似度**比较；  
   - external LLM：`PreferenceClassifier.Classify()` 调 `VLLMClient.Generate()` 产出 route JSON。  
5. 决策阶段 `DecisionEngine.selectBestDecision()` 对匹配结果排序（tier/confidence/priority 策略）。  
6. 全流程中通过 `metrics.RecordSignalExtraction()` 持续上报信号提取次数与时延。
