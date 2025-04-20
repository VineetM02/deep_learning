import numpy as np
import matplotlib.pyplot as plt

# Generate some predictions and true values
y_true = np.array([0, 1, 1, 0, 1])
y_pred = np.array([0.1, 0.8, 0.7, 0.3, 0.9])

# Mean Squared Error (MSE)
def mse_loss(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

# Binary Cross-Entropy Loss
def binary_cross_entropy(y_true, y_pred):
    epsilon = 1e-15  # To avoid log(0)
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

# Hinge Loss (used in SVM)
def hinge_loss(y_true, y_pred):
    # Convert 0 labels to -1 for hinge loss
    y_true = 2 * y_true - 1
    return np.mean(np.maximum(0, 1 - y_true * y_pred))

# Calculate and print losses
print(f"MSE Loss: {mse_loss(y_true, y_pred):.4f}")
print(f"Binary Cross-Entropy Loss: {binary_cross_entropy(y_true, y_pred):.4f}")
print(f"Hinge Loss: {hinge_loss(y_true, y_pred):.4f}")

# Visualize different loss functions
x = np.linspace(-2, 2, 1000)
y_true_plot = 1

# Plot different loss functions
plt.figure(figsize=(12, 4))

plt.subplot(131)
mse = (y_true_plot - x) ** 2
plt.plot(x, mse)
plt.title('MSE Loss')
plt.xlabel('Prediction')
plt.ylabel('Loss')

plt.subplot(132)
bce = -(y_true_plot * np.log(1 / (1 + np.exp(-x))) + 
        (1 - y_true_plot) * np.log(1 - 1 / (1 + np.exp(-x))))
plt.plot(x, bce)
plt.title('Binary Cross-Entropy')
plt.xlabel('Prediction')

plt.subplot(133)
hinge = np.maximum(0, 1 - y_true_plot * x)
plt.plot(x, hinge)
plt.title('Hinge Loss')
plt.xlabel('Prediction')

plt.tight_layout()
plt.show() 