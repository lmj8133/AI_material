import numpy as np
import matplotlib.pyplot as plt

# Data preparation
x = np.array([[1, 1.0, 1.0],
              [1, 1.2, 1.0],
              [1, 1.0, 1.2],
              [1, 1.3, 1.1],
              [1, 2.5, 2.4],
              [1, 2.6, 2.6],
              [1, 2.4, 2.5],
              [1, 2.7, 2.5]])

t = np.array([1, 1, 1, 1, -1, -1, -1, -1])

# Calculate the optimal weights using the closed-form solution
# w = (X^T X)^(-1) X^T t
X_transpose = np.transpose(x)
XTX = np.dot(X_transpose, x)
XTX_inv = np.linalg.inv(XTX)
XTt = np.dot(X_transpose, t)
w = np.dot(XTX_inv, XTt)

print("Optimal weights (w):", w)

# Calculate predictions
y_pred = np.dot(x, w)

# Calculate Mean Squared Error
mse = np.mean((t - y_pred) ** 2)
print("Mean Squared Error:", mse)

# Visualize the data and decision boundary
plt.figure(figsize=(10, 6))

# Plot data points
class1_indices = t == 1
class2_indices = t == -1
plt.scatter(x[class1_indices, 1], x[class1_indices, 2], color='blue', label='Class 1 (t=1)')
plt.scatter(x[class2_indices, 1], x[class2_indices, 2], color='red', label='Class 2 (t=-1)')

# Plot decision boundary (where w0 + w1*x1 + w2*x2 = 0)
x1_min, x1_max = x[:, 1].min() - 0.1, x[:, 1].max() + 0.1
x2_min, x2_max = x[:, 2].min() - 0.1, x[:, 2].max() + 0.1
x1_grid, x2_grid = np.meshgrid(np.linspace(x1_min, x1_max, 100), np.linspace(x2_min, x2_max, 100))

# Decision boundary equation: w0 + w1*x1 + w2*x2 = 0 => x2 = (-w0 - w1*x1) / w2
x2_boundary = (-w[0] - w[1] * x1_grid) / w[2]

plt.plot(x1_grid, x2_boundary, 'g-', label='Decision Boundary')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Linear Regression for Binary Classification')
plt.legend()
plt.grid(True)
plt.axis('equal')

# Show the equation of the regression line
equation = f'y = {w[0]:.4f} + {w[1]:.4f}x₁ + {w[2]:.4f}x₂'
plt.annotate(equation, xy=(0.05, 0.95), xycoords='axes fraction', fontsize=12, 
             bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))

plt.show()

print("Regression Line Equation:")
print(f"y = {w[0]:.4f} + {w[1]:.4f}x₁ + {w[2]:.4f}x₂")

# Calculate accuracy
y_pred_class = np.sign(y_pred)  # Convert predictions to class labels (-1 or 1)
accuracy = np.mean(y_pred_class == t)
print(f"Accuracy: {accuracy:.2%}")
