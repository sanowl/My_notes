# Mathematical Analysis of Positional Encodings

## Fourier Properties

### Frequency Space Analysis
The sinusoidal functions create a frequency basis:

$$PE_{(pos,2i)} = \sin(\frac{pos}{10000^{2i/d_{model}}})$$
$$PE_{(pos,2i+1)} = \cos(\frac{pos}{10000^{2i/d_{model}}})$$

1. **Wavelength Distribution**:
   $$\lambda_i = 10000^{2i/d_{model}}$$
   Forms geometric sequence from $2\pi$ to $10000 \cdot 2\pi$

2. **Phase Relationships**:
   For any position offset k:
   $$PE_{pos+k} = f_k(PE_{pos})$$
   where $f_k$ is a linear transformation

### Theoretical Properties

1. **Uniqueness Theorem**:
   For positions p₁, p₂:
   $$p_1 \neq p_2 \implies \|PE_{p_1} - PE_{p_2}\| > 0$$

2. **Distance Preservation**:
   $$\|PE_{p_1} - PE_{p_2}\| \propto \log(|p_1 - p_2|)$$

3. **Information Capacity**:
   Maximum distinguishable positions:
   $$N_{max} = O(e^{d_{model}/2})$$

## Geometric Analysis

1. **Manifold Structure**:
   PE forms a continuous curve on d-dimensional hypersphere:
   $$\|PE_{pos}\| = \sqrt{d_{model}/2}$$

2. **Angular Separation**:
   Between consecutive positions:
   $$\cos(\theta_{pos,pos+1}) = \frac{\langle PE_{pos}, PE_{pos+1} \rangle}{\|PE_{pos}\| \|PE_{pos+1}\|}$$
