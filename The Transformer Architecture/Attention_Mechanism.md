# Attention Mechanism in Transformers

## Scaled Dot-Product Attention

The core attention mechanism is defined as:

$$Attention(Q,K,V) = softmax(\frac{QK^T}{\sqrt{d_k}})V$$

### Mathematical Components

1. **Query-Key Similarity**:
   $$S = QK^T$$
   Where:
   - $Q \in \mathbb{R}^{n \times d_k}$ (Query matrix)
   - $K \in \mathbb{R}^{m \times d_k}$ (Key matrix)
   - $S \in \mathbb{R}^{n \times m}$ (Similarity matrix)

2. **Scaling Factor**:
   $$\frac{1}{\sqrt{d_k}}$$
   Prevents exploding gradients in deep networks

3. **Attention Weights**:
   $$A = softmax(\frac{S}{\sqrt{d_k}})$$
   Normalizes attention scores across keys

### Implementation Example
```python
def scaled_dot_product_attention(query, key, value, mask=None):
    matmul_qk = torch.matmul(query, key.transpose(-2, -1))
    d_k = query.size(-1)
    scaled_attention_logits = matmul_qk / math.sqrt(d_k)
    
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)
    
    attention_weights = F.softmax(scaled_attention_logits, dim=-1)
    output = torch.matmul(attention_weights, value)
    return output, attention_weights
```

![Attention Mechanism](./images/attention_mechanism.png)
