import numpy as np
import matplotlib.pyplot as plt

# Generate sample data
np.random.seed(42)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

# Gradient Descent implementation
def gradient_descent(X, y, learning_rate=0.01, n_iterations=1000):
    m = len(y)
    theta = np.random.randn(2, 1)  # Random initialization
    
    # Add bias term to X
    X_b = np.c_[np.ones((m, 1)), X]
    
    for iteration in range(n_iterations):
        # Compute predictions
        y_pred = X_b.dot(theta)
        
        # Compute gradients
        gradients = 2/m * X_b.T.dot(y_pred - y)
        
        # Update parameters
        theta = theta - learning_rate * gradients
        
    return theta

# Train the model
theta = gradient_descent(X, y)

# Plot results
plt.scatter(X, y, label='Data points')
plt.plot(X, theta[0] + theta[1] * X, 'r-', label='Fitted line')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Gradient Descent: Linear Regression')
plt.legend()
plt.show()

print(f"Learned parameters: bias = {theta[0][0]:.2f}, slope = {theta[1][0]:.2f}") 