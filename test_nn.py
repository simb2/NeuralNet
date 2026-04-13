from mlp import MLP
import numpy as np

# Fix seed for reproducibility
np.random.seed(42)

# Simulate a simple dataset:
# Class 0: centered at (-1, -1), Class 1: centered at (1, 1)
n = 200
X0 = np.random.randn(n // 2, 2) - 1
X1 = np.random.randn(n // 2, 2) + 1
X = np.vstack([X0, X1])
y = np.array([0] * (n // 2) + [1] * (n // 2), dtype=float)

# Shuffle
idx = np.random.permutation(n)
X, y = X[idx], y[idx]

# Initialize MLP — 2 inputs, hidden layers of size 4 and 3, then output neuron
mlp = MLP(list_num_neurons=[4, 3], input=X[0])

# Train
mlp.train(X, y, learning_rate=0.01, epochs=20)

# Evaluate accuracy on training set
correct = 0
for i in range(len(X)):
    prob = mlp.forward(X[i])
    pred = 1 if prob >= 0.5 else 0
    if pred == y[i]:
        correct += 1

print(f"\nTraining accuracy: {correct / len(X) * 100:.1f}%")