# Neural Networks Quick Reference Cheat Sheet

## 🧠 Core Concepts

### Single Neuron
```
z = w₁x₁ + w₂x₂ + ... + wₙxₙ + b = wᵀx + b
a = f(z)
```
- `x`: inputs
- `w`: weights  
- `b`: bias
- `f`: activation function
- `z`: weighted sum (pre-activation)
- `a`: output (post-activation)



### Network Layer
```
Z[l] = W[l] × A[l-1] + b[l]
A[l] = f(Z[l])
```
- `[l]`: layer number
- `A[0]`: input layer (X)

## 🎯 Activation Functions

| Function | Formula | Derivative | Range | Use Case |
|----------|---------|------------|-------|----------|
| **Sigmoid** | σ(z) = 1/(1+e⁻ᶻ) | σ(z)(1-σ(z)) | (0, 1) | Binary output |
| **Tanh** | (eᶻ - e⁻ᶻ)/(eᶻ + e⁻ᶻ) | 1 - tanh²(z) | (-1, 1) | Hidden layers |
| **ReLU** | max(0, z) | 1 if z>0 else 0 | [0, ∞) | Hidden layers (default) |
| **Softmax** | eᶻⁱ / Σⱼ eᶻʲ | (depends on class) | (0, 1), Σ=1 | Multi-class output |

### When to Use What?
- **Hidden layers**: ReLU (most common)
- **Binary classification output**: Sigmoid
- **Multi-class classification output**: Softmax
- **Regression output**: Linear (no activation)

## 📊 Loss Functions

### For Regression
**Mean Squared Error (MSE)**
```
L = (1/m) Σ(y - ŷ)²
```

### For Classification
**Binary Cross-Entropy**
```
L = -(1/m) Σ[y log(ŷ) + (1-y) log(1-ŷ)]
```

**Categorical Cross-Entropy**
```
L = -(1/m) Σᵢ Σⱼ yⱼ log(ŷⱼ)
```

## 🔄 Forward Propagation

**Step-by-step:**
```python
# For each layer l = 1, 2, ..., L:
Z[l] = W[l] @ A[l-1] + b[l]  # Weighted sum
A[l] = activation(Z[l])       # Apply activation

# Final output
Y_pred = A[L]
```

**Matrix Dimensions:**
```
X:      (n[0], m)    # n[0] = # features, m = # samples
W[l]:   (n[l], n[l-1])
b[l]:   (n[l], 1)
Z[l]:   (n[l], m)
A[l]:   (n[l], m)
```

## 🔙 Backpropagation

**Step-by-step (categorical cross-entropy + softmax):**
```python
# Output layer
dZ[L] = A[L] - Y

# Hidden layers (l = L-1, L-2, ..., 1)
dA[l] = W[l+1].T @ dZ[l+1]
dZ[l] = dA[l] * activation_derivative(Z[l])

# Gradients
dW[l] = (1/m) * dZ[l] @ A[l-1].T
db[l] = (1/m) * sum(dZ[l], axis=1, keepdims=True)
```

## 📉 Gradient Descent

**Update Rule:**
```python
W[l] = W[l] - α * dW[l]
b[l] = b[l] - α * db[l]
```
- `α` (alpha): learning rate

### Choosing Learning Rate
- Too small: Slow convergence
- Too large: Divergence/oscillation
- Typical values: 0.001, 0.01, 0.1

### Variants
- **Batch GD**: Use all training data
- **Mini-batch GD**: Use batches (common: 32, 64, 128, 256)
- **Stochastic GD**: Use one sample at a time

## 🎓 Training Algorithm

```python
for epoch in range(epochs):
    # 1. Shuffle data
    # 2. For each batch:
    #    a. Forward propagation
    Y_pred = forward(X)
    #    b. Compute loss
    loss = compute_loss(Y, Y_pred)
    #    c. Backward propagation
    grads = backward(Y, Y_pred)
    #    d. Update parameters
    update(grads, learning_rate)
```

## 🔧 Weight Initialization

**Common Methods:**
```python
# Random small values
W = np.random.randn(n_out, n_in) * 0.01

# Xavier/Glorot (for sigmoid/tanh)
W = np.random.randn(n_out, n_in) * sqrt(1/n_in)

# He initialization (for ReLU)
W = np.random.randn(n_out, n_in) * sqrt(2/n_in)

# Bias
b = np.zeros((n_out, 1))
```

## 📈 Hyperparameters

| Parameter | Typical Values | What It Does |
|-----------|----------------|--------------|
| Learning rate | 0.001 - 0.1 | Step size for updates |
| Batch size | 32, 64, 128, 256 | Samples per update |
| Epochs | 10 - 1000 | Full passes through data |
| Hidden layers | 1 - 5 | Network depth |
| Neurons per layer | 16 - 512 | Layer width |

## 🐛 Common Issues & Solutions

### High Training Loss
- ❌ Learning rate too low → Increase it
- ❌ Too few epochs → Train longer
- ❌ Network too small → Add layers/neurons
- ❌ Bad initialization → Use He/Xavier init

### Training Loss Good, Test Loss Bad (Overfitting)
- ❌ Too complex network → Reduce size
- ❌ Too few training samples → Get more data
- ✅ Use regularization (L2, dropout)
- ✅ Use data augmentation

### Loss Exploding/NaN
- ❌ Learning rate too high → Decrease it
- ❌ Poor initialization → Use proper init
- ❌ Numerical instability → Clip gradients, normalize inputs

### Slow Training
- ✅ Use mini-batches
- ✅ Use ReLU instead of sigmoid/tanh
- ✅ Normalize inputs
- ✅ Use GPU if available

## 💡 Best Practices

### Data Preprocessing
```python
# 1. Normalize inputs to [0, 1] or standardize
X = X / 255.0  # For images
X = (X - mean) / std  # Standardization

# 2. One-hot encode labels for classification
# [2] → [0, 0, 1, 0, 0, ...] for class 2

# 3. Shuffle training data each epoch
```

### Model Development
1. **Start simple**: Begin with small network
2. **Validate**: Use validation set to tune hyperparameters
3. **Regularize**: Add regularization if overfitting
4. **Iterate**: Gradually increase complexity if needed

### Debugging
```python
# Check shapes
print(f"X: {X.shape}, Y: {Y.shape}")
print(f"W1: {W1.shape}, b1: {b1.shape}")

# Monitor training
if epoch % 10 == 0:
    print(f"Epoch {epoch}: Loss = {loss:.4f}")

# Visualize
plt.plot(loss_history)
```

## 📐 MNIST-Specific

### Data Format
- **Images**: 28×28 pixels = 784 features
- **Labels**: 0-9 (10 classes)
- **Training**: 60,000 images
- **Test**: 10,000 images

### Preprocessing
```python
# Flatten
X = X.reshape(X.shape[0], -1).T  # (784, m)

# Normalize
X = X / 255.0

# One-hot encode
Y_onehot = np.zeros((10, m))
for i in range(m):
    Y_onehot[Y[i], i] = 1
```

### Good Starting Architecture
```python
layers = [784, 128, 64, 10]
learning_rate = 0.1
batch_size = 128
epochs = 50
```
Expected accuracy: ~92-95% on test set

## 🎯 Quick Troubleshooting

| Problem | Check |
|---------|-------|
| Code doesn't run | Did you run all previous cells? |
| Wrong shapes | Print shapes, check transpose |
| Always predicts same class | Check weight initialization |
| Accuracy stuck at 10% (MNIST) | Network is guessing randomly |
| Loss is NaN | Learning rate too high |
| Loss not decreasing | Learning rate too low or wrong implementation |

## 📚 Key Equations Summary

```
Forward:     Z = WA + b,  A = f(Z)
Loss:        L = -(1/m) Σ y log(ŷ)
Backward:    dZ = A - Y  (for output)
             dW = (1/m) dZ Aᵀ
             db = (1/m) Σ dZ
Update:      W = W - α dW
```

## 🚀 Quick NumPy Reference

```python
# Matrix multiplication
np.dot(A, B) or A @ B

# Element-wise operations
A * B  # element-wise multiply
A + B  # element-wise add

# Axis operations
np.sum(A, axis=1, keepdims=True)  # sum along axis
np.max(A, axis=0)  # max along axis

# Shape manipulation
A.T  # transpose
A.reshape(m, n)  # reshape

# Useful functions
np.zeros((m, n))  # zero matrix
np.random.randn(m, n)  # random normal
np.clip(A, min, max)  # clip values
```

---