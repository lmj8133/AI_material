import numpy as np
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms

# Function to plot confidence ellipses to visualize covariance matrices
def plot_cov_ellipse(ax, mean, cov, color, alpha=0.3, n_std=2.0):
    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensional dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2, facecolor=color, alpha=alpha)
    
    # Calculating the standard deviation of x from the matrix
    scale_x = np.sqrt(cov[0, 0]) * n_std
    # Calculating the standard deviation of y from the matrix
    scale_y = np.sqrt(cov[1, 1]) * n_std
    
    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean[0], mean[1])
    
    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)

# 1. Generate synthetic 2D data with controlled covariance matrices
# For the equal covariance case
np.random.seed(42)  # For reproducibility

# Class means
mean1 = np.array([0, 2])
mean2 = np.array([3, 0])

# Case 1: Equal covariance matrices (will produce linear boundary)
cov_equal = np.array([[2.0, 0.5], [0.5, 1.0]])  # Same covariance for both classes

# Generate samples from each class with equal covariance
n_samples = 200
class1_equal = np.random.multivariate_normal(mean1, cov_equal, n_samples // 2)
class2_equal = np.random.multivariate_normal(mean2, cov_equal, n_samples // 2)

# Combine data for equal covariance case
X_equal = np.vstack([class1_equal, class2_equal])
y_equal = np.hstack([np.zeros(n_samples // 2), np.ones(n_samples // 2)])

# Case 2: Different covariance matrices (will produce curved boundary)
cov1_diff = np.array([[2.0, 0.5], [0.5, 1.0]])  # Covariance for class 1
cov2_diff = np.array([[1.0, -0.8], [-0.8, 2.0]])  # Different covariance for class 2

# Generate samples with different covariances
class1_diff = np.random.multivariate_normal(mean1, cov1_diff, n_samples // 2)
class2_diff = np.random.multivariate_normal(mean2, cov2_diff, n_samples // 2)

# Combine data for different covariance case
X_diff = np.vstack([class1_diff, class2_diff])
y_diff = np.hstack([np.zeros(n_samples // 2), np.ones(n_samples // 2)])

# 2. Initialize classifiers
lda = LinearDiscriminantAnalysis()   # Assumes equal covariances
qda = QuadraticDiscriminantAnalysis()  # Allows different covariances

# 3. Create a figure with two rows for our two scenarios
plt.figure(figsize=(15, 10))

# ----- Equal Covariance Case -----
# Fit classifiers to the equal covariance data
lda.fit(X_equal, y_equal)
qda.fit(X_equal, y_equal)

# Create a mesh for the equal covariance case
x_min, x_max = X_equal[:, 0].min() - 1, X_equal[:, 0].max() + 1
y_min, y_max = X_equal[:, 1].min() - 1, X_equal[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                     np.linspace(y_min, y_max, 200))

# Predict class labels for the mesh points
Z_lda_equal = lda.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
Z_qda_equal = qda.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

# Plot LDA with equal covariance
plt.subplot(2, 2, 1)
plt.contourf(xx, yy, Z_lda_equal, alpha=0.3, cmap=plt.cm.coolwarm)
plt.scatter(X_equal[y_equal == 0, 0], X_equal[y_equal == 0, 1], c='blue', edgecolor='k', label='Class 0')
plt.scatter(X_equal[y_equal == 1, 0], X_equal[y_equal == 1, 1], c='red', edgecolor='k', label='Class 1')

# Plot covariance ellipses
plot_cov_ellipse(plt.gca(), mean1, cov_equal, 'blue')
plot_cov_ellipse(plt.gca(), mean2, cov_equal, 'red')

plt.title('LDA with Equal Covariances: Linear Boundary')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()

# Plot QDA with equal covariance
plt.subplot(2, 2, 2)
plt.contourf(xx, yy, Z_qda_equal, alpha=0.3, cmap=plt.cm.coolwarm)
plt.scatter(X_equal[y_equal == 0, 0], X_equal[y_equal == 0, 1], c='blue', edgecolor='k', label='Class 0')
plt.scatter(X_equal[y_equal == 1, 0], X_equal[y_equal == 1, 1], c='red', edgecolor='k', label='Class 1')

# Plot covariance ellipses
plot_cov_ellipse(plt.gca(), mean1, cov_equal, 'blue')
plot_cov_ellipse(plt.gca(), mean2, cov_equal, 'red')

plt.title('QDA with Equal Covariances: Nearly Linear Boundary')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()

# ----- Different Covariance Case -----
# Fit classifiers to the different covariance data
lda.fit(X_diff, y_diff)
qda.fit(X_diff, y_diff)

# Create a mesh for the different covariance case
x_min, x_max = X_diff[:, 0].min() - 1, X_diff[:, 0].max() + 1
y_min, y_max = X_diff[:, 1].min() - 1, X_diff[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                     np.linspace(y_min, y_max, 200))

# Predict class labels for the mesh points
Z_lda_diff = lda.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
Z_qda_diff = qda.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

# Plot LDA with different covariances
plt.subplot(2, 2, 3)
plt.contourf(xx, yy, Z_lda_diff, alpha=0.3, cmap=plt.cm.coolwarm)
plt.scatter(X_diff[y_diff == 0, 0], X_diff[y_diff == 0, 1], c='blue', edgecolor='k', label='Class 0')
plt.scatter(X_diff[y_diff == 1, 0], X_diff[y_diff == 1, 1], c='red', edgecolor='k', label='Class 1')

# Plot covariance ellipses
plot_cov_ellipse(plt.gca(), mean1, cov1_diff, 'blue')
plot_cov_ellipse(plt.gca(), mean2, cov2_diff, 'red')

plt.title('LDA with Different Covariances: Linear Boundary (Suboptimal)')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()

# Plot QDA with different covariances
plt.subplot(2, 2, 4)
plt.contourf(xx, yy, Z_qda_diff, alpha=0.3, cmap=plt.cm.coolwarm)
plt.scatter(X_diff[y_diff == 0, 0], X_diff[y_diff == 0, 1], c='blue', edgecolor='k', label='Class 0')
plt.scatter(X_diff[y_diff == 1, 0], X_diff[y_diff == 1, 1], c='red', edgecolor='k', label='Class 1')

# Plot covariance ellipses
plot_cov_ellipse(plt.gca(), mean1, cov1_diff, 'blue')
plot_cov_ellipse(plt.gca(), mean2, cov2_diff, 'red')

plt.title('QDA with Different Covariances: Curved Boundary (Optimal)')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()

# Add an overall title explaining the demonstration
#plt.suptitle('Effect of Covariance Matrices on Decision Boundaries', fontsize=16)
#plt.figtext(0.5, 0.01, 
#           "Top row: When both classes have identical covariance matrices, LDA produces an optimal linear boundary.\n"
#           "Bottom row: When classes have different covariance matrices, QDA's curved boundary is more appropriate.", 
#           ha="center", fontsize=12, bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="gray", alpha=0.8))

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('covariance_decision_boundaries.png', dpi=300, bbox_inches='tight')
plt.show()
