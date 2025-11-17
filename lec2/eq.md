### Key Equations for Autoregressive Transformers with Global Attention

Autoregressive transformers (e.g., decoder-only architectures like GPT) rely on masked self-attention to ensure causality (no future token peeking), while global attention allows attending to all previous positions in the sequence. Below is a list of the most relevant equations, focusing on core components like attention mechanisms, positional encodings, and sub-layers. These are derived from the standard Transformer architecture but adapted for autoregressive decoding.

1. **Scaled Dot-Product Attention** (core of global attention, with masking for autoregressivity):  
   This computes the attention weights over all positions (global), but in autoregressive models, a causal mask is applied to the logits before softmax to set future positions to \(-\infty\).  
   \[
   \text{Attention}(Q, K, V) = \softmax\left( \frac{QK^T}{\sqrt{d_k}} + M \right) V
   \]  
   where \(Q, K, V\) are query, key, and value matrices; \(d_k\) is the key dimension; and \(M\) is the causal mask matrix (\(M_{ij} = 0\) if \(i \geq j\), else \(-\infty\)).

2. **Multi-Head Attention** (parallel attention heads for richer representations, applied globally within the causal mask):  
   \[
   \text{MultiHead}(Q, K, V) = \Concat(\head_1, \dots, \head_h) W^O
   \]  
   where each head is:  
   \[
   \head_i = \text{Attention}(Q W_i^Q, K W_i^K, V W_i^V)
   \]  
   and \(h\) is the number of heads, \(W_i^Q, W_i^K, W_i^V, W^O\) are learned projection matrices.

3. **Positional Encoding** (added to input embeddings to encode sequence order, essential for global attention over positions):  
   For position \(pos\) and dimension \(i\):  
   \[
   PE_{(pos, 2i)} = \sin\left( \frac{pos}{10000^{2i / d_{\text{model}}}} \right)
   \]  
   \[
   PE_{(pos, 2i+1)} = \cos\left( \frac{pos}{10000^{2i / d_{\text{model}}}} \right)
   \]  
   The input to the transformer is \(X + PE\), where \(X\) is the token embedding matrix and \(d_{\text{model}}\) is the model dimension.

4. **Position-wise Feed-Forward Network** (applied after attention, pointwise across positions):  
   \[
   \text{FFN}(x) = \max(0, x W_1 + b_1) W_2 + b_2
   \]  
   where \(W_1, b_1, W_2, b_2\) are parameters, and this is computed independently for each position in the sequence.

5. **Layer Normalization** (used in residual connections around attention and FFN sub-layers):  
   \[
   \LayerNorm(x) = \gamma \cdot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta
   \]  
   where \(\mu\) and \(\sigma^2\) are the mean and variance over the feature dimension, \(\epsilon\) is a small constant, and \(\gamma, \beta\) are learned parameters.

6. **Residual Connection** (around each sub-layer, e.g., attention or FFN):  
   \[
   x' = \LayerNorm(x + \text{SubLayer}(x))
   \]  
   where \(\text{SubLayer}\) is either Multi-Head Attention or FFN.

7. **Autoregressive Decoding Probability** (output logits to softmax for next-token prediction):  
   During inference, the probability of the next token \(y_t\) given previous tokens \(y_{<t}\):  
   \[
   P(y_t \mid y_{<t}) = \softmax(W_o \cdot h_t + b_o)
   \]  
   where \(h_t\) is the hidden state from the final transformer layer at position \(t\), and \(W_o, b_o\) are output projection parameters.

These equations form the backbone of models like GPT, where global attention enables full context utilization within the causal constraint. For variations (e.g., relative positional encodings), additional equations may apply, but these are the fundamentals.