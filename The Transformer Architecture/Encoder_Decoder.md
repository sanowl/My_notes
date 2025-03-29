# Encoder-Decoder Architecture

## Structure

### Encoder
- N identical layers
- Each layer has:
  1. Multi-head self-attention
  2. Position-wise feed-forward network
  3. Layer normalization
  4. Residual connections

### Decoder
- N identical layers
- Each layer has:
  1. Masked multi-head self-attention
  2. Multi-head attention over encoder output
  3. Position-wise feed-forward network
  4. Layer normalization
  5. Residual connections

### Implementation
```python
class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.norm3 = LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
```

![Encoder-Decoder Architecture](./images/encoder_decoder.png)

# Mathematical Analysis of Encoder-Decoder Architecture

## Theoretical Framework

### Information Flow Analysis

1. **Encoder Representation**:
   $$h_l^e = f_l^e(h_{l-1}^e) = LN(FFN(LN(MHA(h_{l-1}^e))) + h_{l-1}^e)$$
   where $h_l^e$ is the l-th encoder layer output

2. **Decoder Representation**:
   $$h_l^d = f_l^d(h_{l-1}^d, h_L^e) = LN(FFN(LN(MHA_2(LN(MHA_1(h_{l-1}^d)), h_L^e))) + h_{l-1}^d)$$

### Compositional Properties

1. **Layer Transformation**:
   For encoder layer l:
   $$T_l: \mathbb{R}^{n \times d} \rightarrow \mathbb{R}^{n \times d}$$
   $$T_l = LN \circ (I + FFN) \circ LN \circ (I + MHA)$$

2. **Cross-Layer Interaction**:
   $$I(h_l^e; h_k^e) \geq I(h_{l+1}^e; h_{k+1}^e)$$
   for layers l, k where |l-k| > 1

### Residual Connection Analysis

1. **Signal Propagation**:
   $$\frac{\partial \mathcal{L}}{\partial h_l} = \frac{\partial \mathcal{L}}{\partial h_L}\prod_{i=l+1}^L (1 + \frac{\partial f_i}{\partial h_{i-1}})$$

2. **Gradient Magnitude Bounds**:
   $$\|\frac{\partial \mathcal{L}}{\partial h_l}\| \leq \|\frac{\partial \mathcal{L}}{\partial h_L}\|\prod_{i=l+1}^L (1 + \|J_{f_i}\|)$$

### Cross-Attention Mathematics

1. **Information Transfer**:
   $$I(Y;X) = I(h_L^d; h_L^e) \leq \min(H(h_L^d), H(h_L^e))$$

2. **Attention Distribution**:
   $$P(y_t|x,y_{<t}) = softmax(f(h_L^d, h_L^e))$$

### Theoretical Guarantees

1. **Universal Approximation**:
   For sufficient depth L:
   $$\|f_{target} - f_{transformer}\|_{\infty} \leq \epsilon$$

2. **Convergence Rate**:
   $$\mathbb{E}[\mathcal{L}(θ_t)] - \mathcal{L}(θ^*) \leq O(\frac{1}{\sqrt{t}})$$

### Architecture Complexity

1. **Parameter Count**:
   $$|θ| = L(4d^2 + 8dh + 4d)$$
   where d is model dimension, h is head dimension

2. **Computational Depth**:
   $$D(n) = O(1)$$
   compared to RNN's O(n)

# Mathematical Deep Dive: Encoder-Decoder Architecture

## Comprehensive Mathematical Analysis

### 1. Layer-wise Transformations

Each encoder layer implements a complex non-linear transformation:

1. **Self-Attention Mapping**:
   $$SA(X) = Attention(XW^Q, XW^K, XW^V)W^O$$
   where the composition creates a rich functional space:
   $$\mathcal{F}_{SA} = \{f: \mathbb{R}^{n \times d} \rightarrow \mathbb{R}^{n \times d}\}$$

2. **Feed-Forward Expansion**:
   $$FFN(X) = \sigma(XW_1 + b_1)W_2 + b_2$$
   This creates a position-wise expansion into higher dimensions:
   $$\dim(\text{im}(FFN)) = \min(d_{ff}, \text{rank}(W_1))$$

3. **Combined Layer Effect**:
   $$T_l(X) = LN(FFN(LN(SA(X) + X)) + LN(SA(X) + X))$$
   
   Properties:
   - Lipschitz continuity: $\|T_l(X) - T_l(Y)\| \leq L\|X - Y\|$
   - Bounded variance: $Var(T_l(X)) \leq \gamma^2Var(X)$

### 2. Cross-Layer Information Flow

The information bottleneck principle applies across layers:

1. **Mutual Information Chain**:
   $$I(X;h_1) \geq I(X;h_2) \geq ... \geq I(X;h_L)$$
   
   With equality iff:
   $$\frac{\partial h_{l+1}}{\partial h_l} = I_d$$

2. **Layer-wise Capacity**:
   For each layer l:
   $$C_l = \max_{p(x)} I(X;h_l)$$
   Subject to:
   $$\mathbb{E}[\|h_l\|^2] \leq P_l$$

### 3. Residual Connection Theory

1. **Path Analysis**:
   For any layer l:
   $$h_l = h_0 + \sum_{i=1}^l \Delta h_i$$
   where $\Delta h_i$ represents the residual learning

2. **Gradient Decomposition**:
   $$\frac{\partial \mathcal{L}}{\partial h_0} = \sum_{l=1}^L \frac{\partial \mathcal{L}}{\partial h_l} \cdot \prod_{i=1}^l (1 + \frac{\partial f_i}{\partial h_{i-1}})$$

3. **Path Length Analysis**:
   Expected path length:
   $$\mathbb{E}[\|path\|] = \sqrt{L} + O(1)$$

### 4. Deep Network Properties

1. **Depth-Width Trade-off**:
   For approximation error ε:
   $$L \cdot d_{model} = \Omega(\frac{1}{\epsilon}\log(\frac{1}{\epsilon}))$$

2. **Expressivity Bounds**:
   Number of distinguishable functions:
   $$|\mathcal{F}| \leq (2^{d_{model}})^L$$

### 5. Cross-Attention Dynamics

1. **Information Transfer Rate**:
   $$R = \frac{I(X;Y)}{H(X)}$$
   where optimal rate achieved at:
   $$R_{opt} = \max_{p(x)} \frac{I(X;Y)}{H(X)}$$

2. **Attention Alignment**:
   Cross-attention similarity:
   $$A_{cross} = \frac{h_L^d(h_L^e)^T}{\|h_L^d\|\|h_L^e\|}$$

### 6. Training Dynamics

1. **Loss Surface Geometry**:
   For parameters θ:
   $$\mathcal{L}(\theta) = \mathbb{E}_{x,y}[-\log p_\theta(y|x)]$$
   
   Hessian structure:
   $$H = \nabla^2\mathcal{L}(\theta) = J^TJ + \sum_i \lambda_i H_i$$

2. **Optimization Trajectory**:
   Parameter evolution:
   $$\theta_t = \theta_0 - \eta\sum_{i=1}^t g_i + \mathcal{O}(\eta^2)$$

### 7. Memory-Computation Trade-offs

1. **Space Complexity**:
   Total memory requirement:
   $$M_{total} = M_{params} + M_{activations} + M_{gradients}$$
   where:
   $$M_{activations} = \mathcal{O}(BLnd_{model})$$

2. **Computation Graph**:
   Forward pass operations:
   $$C_{forward} = \mathcal{O}(BLn^2d_{model})$$
   
   Backward pass scaling:
   $$C_{backward} = \alpha C_{forward}$$
   where α ≈ 2-3
