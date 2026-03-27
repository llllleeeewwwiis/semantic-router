OLLAMA_HOST=0.0.0.0:8002 ollama serve

vllm-sr serve \
  --config new-config/pa.yaml \
  --image ghcr.io/vllm-project/semantic-router/vllm-sr:3298fef2c27e27f913e54bf5277a8d9ebbb8ecd3 \
  --image-pull-policy ifnotpresent

vllm-sr serve \
  --config new-config/accuracy.yaml \
  --image ghcr.io/vllm-project/semantic-router/vllm-sr:3298fef2c27e27f913e54bf5277a8d9ebbb8ecd3 \
  --image-pull-policy ifnotpresent