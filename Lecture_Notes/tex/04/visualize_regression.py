import numpy as np
import matplotlib.pyplot as plt

# Data preparation - using the same data from mse.py
x_1 = np.array([
    [1, 1.0, 1.0],
    [1, 1.2, 1.0],
    [1, 1.0, 1.2],
    [1, 1.3, 1.1],
    [1, 1.5, 1.4],
    [1, 1.6, 1.6],
    [1, 1.4, 1.5],
    [1, 1.7, 1.5],
])

t_1 = np.array([1, 1, 1, 1, -1, -1, -1, -1])

x_2 = np.array([
    [1, 1.0, 1.0],
    [1, 1.2, 1.0],
    [1, 1.0, 1.2],
    [1, 1.3, 1.1],
    [1, 1.5, 1.4],
    [1, 1.6, 1.6],
    [1, 1.4, 1.5],
    [1, 1.7, 1.5],
    ####
    [1, 3.2, 3.3],
    [1, 3.4, 3.5]
])

t_2 = np.array([1, 1, 1, 1, -1, -1, -1, -1, -1, -1])

def regression(x, t):
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

    # Create a more detailed visualization of the regression function in x₂-x₁ space
    plt.figure(figsize=(10, 8))

    # Set up the plot area with a bit more margin
    x1_min, x1_max = 0.5, 4.0
    x2_min, x2_max = 0.5, 4.0

    # Create a meshgrid for the feature space
    x1_grid, x2_grid = np.meshgrid(np.linspace(x1_min, x1_max, 100), np.linspace(x2_min, x2_max, 100))

    # Calculate the regression function values over the grid
    X_grid = np.column_stack([np.ones(x1_grid.flatten().shape), x1_grid.flatten(), x2_grid.flatten()])
    y_grid = np.dot(X_grid, w).reshape(x1_grid.shape)

    # Create a filled contour plot to visualize the regression function
    contour = plt.contourf(x1_grid, x2_grid, y_grid, levels=20, cmap='coolwarm', alpha=0.7)
    plt.colorbar(contour, label='Regression Function Value')

    # Plot the decision boundary (where the regression function equals 0)
    plt.contour(x1_grid, x2_grid, y_grid, levels=[0], colors='green', linewidths=2, linestyles='solid')

    # Plot data points
    class1_indices = t == 1
    class2_indices = t == -1
    plt.scatter(x[class1_indices, 1], x[class1_indices, 2], color='blue', marker='o', s=100, 
               edgecolor='black', label='Class 1 (t=1)')
    plt.scatter(x[class2_indices, 1], x[class2_indices, 2], color='red', marker='s', s=100, 
               edgecolor='black', label='Class 2 (t=-1)')

    # Add arrows to show the gradient direction of the regression function
    plt.quiver(x1_grid[::10, ::10], x2_grid[::10, ::10], 
              w[1] * np.ones_like(x1_grid[::10, ::10]), w[2] * np.ones_like(x2_grid[::10, ::10]),
              scale=20, color='black', alpha=0.3)

    # Enhance the plot with labels and title
    plt.xlabel('x₁', fontsize=14)
    plt.ylabel('x₂', fontsize=14)
    #plt.title('Visualization of Regression Function in x₂-x₁ Space', fontsize=16)
    plt.legend(loc='lower right', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)

    # Show the equation of the regression function
    equation = f'f(x₁,x₂) = {w[0]:.4f} + {w[1]:.4f}x₁ + {w[2]:.4f}x₂'
    plt.annotate(equation, xy=(0.05, 0.95), xycoords='axes fraction', fontsize=14, 
                 bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="gray", alpha=0.8))

    # Add explanation text
    #plt.figtext(0.5, 0.01, 
    #           "The green line shows the decision boundary where the regression function equals zero.\n"
    #           "The color gradient represents the value of the regression function across the feature space.\n"
    #           "Points above the boundary are classified as negative, points below as positive.", 
    #           ha="center", fontsize=12, bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="gray", alpha=0.8))

    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.savefig('regression_visualization.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Print additional information
    print("\nRegression Function:")
    print(f"f(x₁,x₂) = {w[0]:.4f} + {w[1]:.4f}x₁ + {w[2]:.4f}x₂")

    # Calculate accuracy
    y_pred_class = np.sign(y_pred)  # Convert predictions to class labels (-1 or 1)
    accuracy = np.mean(y_pred_class == t)
    print(f"Classification Accuracy: {accuracy:.2%}")

regression(x_1, t_1)
regression(x_2, t_2)