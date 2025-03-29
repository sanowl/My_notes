# Multi-Head Attention

## Architecture

Multi-head attention allows parallel attention operations:

$$MultiHead(Q,K,V) = Concat(head_1,...,head_h)W^O$$
$$head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)$$

### Components

1. **Linear Projections**:
   - $W_i^Q \in \mathbb{R}^{d_{model} \times d_k}$
   - $W_i^K \in \mathbb{R}^{d_{model} \times d_k}$
   - $W_i^V \in \mathbb{R}^{d_{model} \times d_v}$
   - $W^O \in \mathbb{R}^{hd_v \times d_{model}}$

### Implementation
```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_k = d_model // num_heads
        self.h = num_heads
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        bs = q.size(0)
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k).transpose(1, 2)
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k).transpose(1, 2)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k).transpose(1, 2)

        scores = attention(q, k, v, mask, self.dropout)
        concat = scores.transpose(1, 2).contiguous().view(bs, -1, self.d_model)
        return self.out(concat)
```

![Multi-Head Attention](./images/multihead_attention.png)

# Mathematical Analysis of Multi-Head Attention

## Theoretical Framework

### Linear Transformations

Each head performs:
$$h_i = Attention(QW_i^Q, KW_i^K, VW_i^V)$$

Where:
1. **Projection Spaces**:
   $$W_i^Q: \mathbb{R}^{d_{model}} \rightarrow \mathbb{R}^{d_k}$$
   $$W_i^K: \mathbb{R}^{d_{model}} \rightarrow \mathbb{R}^{d_k}$$
   $$W_i^V: \mathbb{R}^{d_{model}} \rightarrow \mathbb{R}^{d_v}$$

2. **Subspace Learning**:
   Each head learns a different subspace projection:
   $$\text{span}(W_i^Q) \perp \text{span}(W_j^Q), \quad i \neq j$$

### Information Theory

1. **Joint Information Capture**:
   $$I(MultiHead;X) \geq \max_i I(h_i;X)$$
   
2. **Capacity Analysis**:
   Total information flow:
   $$C_{total} = \sum_{i=1}^h C_i$$
   where $C_i$ is the capacity of head $i$

### Geometric Properties

1. **Subspace Decomposition**:
   $$\mathbb{R}^{d_{model}} = \bigoplus_{i=1}^h \text{span}(W_i^O)$$

2. **Attention Distribution**:
   $$P(attention|head=i) = softmax(\frac{QW_i^Q(KW_i^K)^T}{\sqrt{d_k}})$$

# Multi-Head Attention: Mathematical Analysis

## Fundamental Principles

### 1. Parallel Feature Learning

The multi-head mechanism splits representation into h subspaces:
$$head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)$$

Each head captures different aspects:
1. **Feature Space Decomposition**:
   $$\mathbb{R}^{d_{model}} = \bigoplus_{i=1}^h \mathbb{R}^{d_k}$$
   where $d_k = d_{model}/h$

2. **Independent Transformations**:
   $$\begin{align*}
   Q_i &= QW_i^Q \in \mathbb{R}^{n \times d_k} \\
   K_i &= KW_i^K \in \mathbb{R}^{m \times d_k} \\
   V_i &= VW_i^V \in \mathbb{R}^{m \times d_v}
   \end{align*}$$

### 2. Theoretical Properties

1. **Representation Power**:
   For any fixed attention pattern $A$:
   $$\exists \{W_i^Q, W_i^K, W_i^V\}_{i=1}^h : \|A - MultiHead(Q,K,V)\| \leq \epsilon$$

2. **Information Flow**:
   Mutual information between input and output:
   $$I(X;Y) = \sum_{i=1}^h I(X;head_i) - I(head_1;...;head_h|X)$$

### 3. Geometric Analysis

1. **Subspace Relationships**:
   - Orthogonality measure between heads:
     $$\cos \theta_{ij} = \frac{\langle W_i^Q, W_j^Q \rangle}{\|W_i^Q\| \|W_j^Q\|}$$
   
   - Diversity metric:
     $$D(heads) = \frac{1}{h(h-1)}\sum_{i\neq j} (1 - |\cos \theta_{ij}|)$$

2. **Attention Coverage**:
   Combined attention space:
   $$\mathcal{A} = \bigcup_{i=1}^h \text{span}(A_i)$$
   where $A_i$ is attention matrix for head i

### 4. Statistical Properties

1. **Head Independence**:
   Correlation between heads:
   $$\rho_{ij} = \frac{Cov(head_i, head_j)}{\sigma_i\sigma_j}$$

2. **Output Distribution**:
   For normalized inputs:
   $$\mathbb{E}[\|MultiHead(Q,K,V)\|^2] = d_{model}$$

### 5. Learning Dynamics

1. **Gradient Flow**:
   For each head i:
   $$\frac{\partial \mathcal{L}}{\partial W_i^Q} = \frac{\partial \mathcal{L}}{\partial head_i} \cdot \frac{\partial head_i}{\partial W_i^Q}$$

2. **Update Rules**:
   Weight updates follow:
   $$\Delta W_i^Q = -\eta \cdot \frac{\partial \mathcal{L}}{\partial W_i^Q}$$

### 6. Complexity Analysis

1. **Computational Cost**:
   Per-head complexity:
   $$C_{head} = O(n^2d_k + nmd_k)$$
   
   Total complexity:
   $$C_{total} = O(hn^2d_k + hnmd_k)$$

2. **Memory Requirements**:
   $$M_{total} = h(3nd_k + md_k + nm)$$

### 7. Optimization Theory

1. **Loss Landscape**:
   Local minima properties:
   $$\nabla^2 \mathcal{L}(W^*) \succeq 0$$
   for optimal weights W*

2. **Convergence Rate**:
   With proper initialization:
   $$\mathbb{E}[\|\nabla \mathcal{L}(W_t)\|^2] \leq \frac{C}{\sqrt{t}}$$

### 8. Practical Considerations

1. **Head Importance**:
   Contribution measure:
   $$I_i = \|\mathbb{E}[head_i]\|_F$$

2. **Redundancy Analysis**:
   Inter-head redundancy:
   $$R_{ij} = \frac{\|head_i \cdot head_j^T\|_F}{\|head_i\|_F\|head_j\|_F}$$

3. **Stability Analysis**:
   Output variance bound:
   $$Var(MultiHead(Q,K,V)) \leq \sum_{i=1}^h Var(head_i)$$


