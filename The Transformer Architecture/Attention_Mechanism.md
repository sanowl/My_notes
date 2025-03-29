# Mathematical Foundations of Attention

## Core Mathematical Framework

### Attention as Matrix Operations

The attention mechanism can be decomposed into:

1. **Similarity Computation**:
   $$S(Q,K) = \frac{QK^T}{\sqrt{d_k}} \in \mathbb{R}^{n \times m}$$
   
2. **Probability Distribution Creation**:
   $$P(Q,K) = softmax(S(Q,K))$$
   Where each row $p_i$ satisfies:
   $$\sum_{j=1}^m p_{ij} = 1, \quad p_{ij} > 0$$

3. **Value Aggregation**:
   $$O = P(Q,K)V$$
   This creates weighted combinations of values:
   $$o_i = \sum_{j=1}^m p_{ij}v_j$$

### Theoretical Properties

1. **Gradient Flow**:
   For any attention weight $\alpha_{ij}$:
   $$\frac{\partial \mathcal{L}}{\partial \alpha_{ij}} = \sum_{k=1}^{d_v} \frac{\partial \mathcal{L}}{\partial o_{ik}} v_{jk}$$

2. **Information Bottleneck**:
   The mutual information satisfies:
   $$I(O;V|Q,K) \leq H(P(Q,K))$$
   where $H(P)$ is the entropy of attention distributions

3. **Stability Analysis**:
   The scaling factor $\sqrt{d_k}$ ensures:
   $$Var[q_i^T k_j] = 1$$
   preventing vanishing/exploding gradients

### Geometric Interpretation

1. **Spherical Geometry**:
   Queries and keys lie on a $d_k$-dimensional hypersphere:
   $$\|q_i\| \approx \|k_j\| \approx \sqrt{d_k}$$

2. **Distance Metrics**:
   The attention scores relate to cosine similarity:
   $$s_{ij} = \frac{q_i^T k_j}{\sqrt{d_k}} = \sqrt{d_k}\cos(\theta_{ij})$$

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

## Advanced Attention Analysis

### 1. Information Theoretic View
The attention mechanism maximizes mutual information:

$$I(Y;X|A) = H(Y) - H(Y|X,A)$$

Where:
- $H(Y)$ is output entropy
- $H(Y|X,A)$ is conditional entropy
- $A$ represents attention weights

### 2. Gradient Flow Analysis
For attention weights $\alpha_{ij}$:

$$\frac{\partial \mathcal{L}}{\partial \alpha_{ij}} = \frac{\partial \mathcal{L}}{\partial y_i} \cdot v_j$$

Full gradient computation:

$$\frac{\partial \mathcal{L}}{\partial Q} = \frac{\partial \mathcal{L}}{\partial A} \cdot \frac{\partial A}{\partial S} \cdot \frac{\partial S}{\partial Q}$$

### 3. Complexity Analysis

1. **Time Complexity Breakdown**:
   ```python
   def complexity_analysis(n, d, h):
       # n: sequence length
       # d: dimension
       # h: number of heads
       computation_steps = {
           'QK multiplication': n**2 * d,
           'Softmax': n**2,
           'Value multiplication': n**2 * d
       }
       return computation_steps
   ```

2. **Memory Usage**:
   $$M_{total} = M_{QK} + M_{attention} + M_{output}$$
   $$= (2nd + n^2 + nd)$$

### 4. Advanced Implementation

```python
class TransformerAttention(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, q, k, v, mask=None):
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_model)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        weights = F.softmax(scores, dim=-1)
        weights = self.dropout(weights)
        return torch.matmul(weights, v), weights
```