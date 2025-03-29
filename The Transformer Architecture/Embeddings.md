# Embeddings in Transformer Architecture

## Input Embeddings

The input embedding layer transforms tokens into continuous vector representations:

$$E(x_i) = W_e x_i$$

Where:
- $W_e \in \mathbb{R}^{d_{model} \times |V|}$ is the embedding matrix
- $x_i$ is the one-hot vector of input token
- $d_{model}$ is the model dimension (typically 512)
- $|V|$ is vocabulary size

### Deep Dive into Input Embeddings
The embedding process can be broken down into several key steps:

1. **Tokenization to Embedding**:
   ![Tokenization Process](./images/token_to_embedding.png)
   ```python
   # Example implementation
   class TokenEmbedding(nn.Module):
       def __init__(self, vocab_size, d_model):
           super().__init__()
           self.embedding = nn.Embedding(vocab_size, d_model)
           self.d_model = d_model
           
       def forward(self, x):
           return self.embedding(x) * math.sqrt(self.d_model)
   ```

2. **Dimensional Analysis**:
   - Input token dimension: $|V|$ (vocabulary size)
   - Output embedding dimension: $d_{model}$
   - Transformation matrix: $W_e \in \mathbb{R}^{d_{model} \times |V|}$

## Positional Encodings

Position is encoded using sinusoidal functions:

$$PE_{(pos,2i)} = sin(pos/10000^{2i/d_{model}})$$
$$PE_{(pos,2i+1)} = cos(pos/10000^{2i/d_{model}})$$

Where:
- $pos$ is the position in sequence
- $i$ is the dimension
- $d_{model}$ is embedding dimension

### Understanding Sinusoidal Encodings
![Positional Encoding Visualization](./images/positional_encoding_viz.png)

The sinusoidal functions create a unique pattern for each position:

1. **Wavelength Progression**:
   - Different dimensions correspond to sinusoids of different wavelengths
   - Wavelengths form geometric progression from $2\pi$ to $10000 \cdot 2\pi$

2. **Mathematical Intuition**:
   ```python
   def get_positional_encoding(max_seq_len, d_model):
       PE = torch.zeros(max_seq_len, d_model)
       pos = torch.arange(0, max_seq_len).unsqueeze(1)
       div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
       
       PE[:, 0::2] = torch.sin(pos * div_term)
       PE[:, 1::2] = torch.cos(pos * div_term)
       return PE
   ```

### Properties of Positional Encoding

1. **Unique Pattern**: Each position gets a unique encoding
2. **Fixed Offset**: $PE_{pos+k}$ can be represented as linear function of $PE_{pos}$
3. **Boundedness**: Values confined to [-1,1]

4. **Linear Dependence Properties**:
   For any fixed offset $k$, there exist matrices $W_k^{(1)}, W_k^{(2)}$ such that:
   $$PE_{pos+k} = PE_{pos} \cdot W_k^{(1)} + W_k^{(2)}$$

5. **Relative Position Sensitivity**:
   ![Relative Position Attention](./images/relative_position.png)
   The dot product between different positions exhibits a predictable pattern:
   $$\langle PE_{pos}, PE_{pos+k} \rangle = f(k)$$

## Combined Embeddings

The final input representation is:

$$h_i = E(x_i) + PE_{(i)}$$

Where:
- $h_i$ is the final representation
- $E(x_i)$ is token embedding
- $PE_{(i)}$ is positional encoding

## Embedding Scale

To prevent magnitude issues, embeddings are scaled by:

$$h_i = \sqrt{d_{model}} \cdot E(x_i) + PE_{(i)}$$

### Practical Implementation Insights

1. **Embedding Initialization**:
   ```python
   def initialize_embeddings(d_model, vocab_size):
       std = 1/math.sqrt(d_model)
       embeddings = torch.normal(0, std, (vocab_size, d_model))
       return embeddings
   ```

2. **Layer Normalization Impact**:
   ![Layer Norm Effect](./images/layer_norm_effect.png)
   
   The combined embedding output typically passes through layer normalization:
   $$LN(h_i) = \gamma \odot \frac{h_i - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta$$

## Mathematical Properties

### Attention Compatibility

The embeddings are designed to work with scaled dot-product attention:

$$Attention(Q,K,V) = softmax(\frac{QK^T}{\sqrt{d_k}})V$$

Where embeddings form the initial values for $Q$, $K$, and $V$ matrices.

### Multi-Head Projection Compatibility

The embedding structure enables efficient multi-head projections:

$$h_i^{(j)} = W^{(j)}h_i$$

Where:
- $W^{(j)} \in \mathbb{R}^{d_k \times d_{model}}$ is the projection matrix for head $j$
- $d_k = d_{model}/h$ for $h$ heads
- $h_i^{(j)}$ is the projected embedding for head $j$

### Gradient Flow

The embedding structure allows for efficient gradient propagation:

$$\frac{\partial \mathcal{L}}{\partial E(x_i)} = \frac{\partial \mathcal{L}}{\partial h_i} \cdot \sqrt{d_{model}}$$

### Gradient Analysis

1. **Backward Pass Stability**:
   ![Gradient Flow](./images/gradient_flow.png)
   
   The scaled embedding approach helps maintain gradient magnitudes:
   $$\|\frac{\partial \mathcal{L}}{\partial E(x_i)}\| \approx \|\frac{\partial \mathcal{L}}{\partial h_i}\|$$

## Implementation Considerations

1. **Memory Efficiency**: Shared embedding weights between encoder/decoder
2. **Scale Factors**: Careful initialization to maintain variance
3. **Dimension Selection**: Usually power of 2 (512, 768, 1024)

## Performance Optimization Techniques

1. **Memory-Efficient Implementation**:
   ```python
   class SharedEmbedding(nn.Module):
       def __init__(self, vocab_size, d_model):
           super().__init__()
           self.embedding = nn.Parameter(torch.empty(vocab_size, d_model))
           self.reset_parameters()
           
       def reset_parameters(self):
           nn.init.normal_(self.embedding, std=0.02)
   ```

2. **Quantization Strategies**:
   - 8-bit quantization for embedding tables
   - Pruning less frequent tokens
   - Adaptive dimension reduction

3. **Caching Mechanisms**:
   - Position encoding precomputation
   - Frequent token embedding caching
   - Gradient accumulation strategies

## Benchmarks and Empirical Results

| Model Size | Embedding Dim | Training Time | Memory Usage |
|------------|--------------|---------------|--------------|
| Small      | 512          | 1x            | 1GB         |
| Medium     | 768          | 1.5x          | 2.3GB       |
| Large      | 1024         | 2.1x          | 4.1GB       |


