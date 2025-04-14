# Lecture 1: Regression - Case Study

---

### 1. Purpose of Machine Learning

Machine learning aims to find the **best possible function (model)** from a **hypothesis space (possibly infinite)** to fit the dataset.

---

### 2. How to Evaluate Goodness of a Function $y = b + w \times x$?

#### 2.1 Loss Function $L$

- **Input**: a function $f(x)$
- **Output**: how bad it is
- **Goal**: find the best $f(x)$ by minimizing $L$

The loss function for linear regression:

\[
L(w, b) = \sum_{n=1}^{N}\left(\hat{y}^{(n)} - (b + w \times x^{(n)})\right)^2
\]

where $\hat{y}^{(n)}$ is the true value and superscript $(n)$ indicates the $n^{th}$ data point.

---

#### 2.2 Gradient Descent

In order to find **the best $f(x)$**, we need to find **the minimum of $L(w, b)$**.

- Start with some initial values $w^{(0)}, b^{(0)}$
- Repeat until convergence:

\[
w^{(t+1)} = w^{(t)} - \eta \frac{\partial L}{\partial w}\Big|_{w=w^{(t)}, b=b^{(t)}}
\]
\[
b^{(t+1)} = b^{(t)} - \eta \frac{\partial L}{\partial b}\Big|_{w=w^{(t)}, b=b^{(t)}}
\]

where $\eta$ is the **learning rate**, and $t$ is epochs.

---

#### 2.3 Increasing Model Complexity (Polynomial Regression)

We can modify the model to better fit data by increasing complexity:

- $y = b + w_1 \times x$
- $y = b + w_1 \times x + w_2 \times x^2$
- $y = b + w_1 \times x + w_2 \times x^2 + w_3 \times x^3$
- $y = b + w_1 \times x + w_2 \times x^2 + w_3 \times x^3 + w_4 \times x^4$
- and so on...

---

#### 2.4 Overfitting

If we choose a more complex model, it fits the **training data** better, but **generalizes worse** (i.e., it does not perform well on unseen test data).

---

#### 2.5 Regularization

- **Goal**: Redesign the **loss function** to reduce sensitivity of the model's predictions to input noise by constraining weights.
- **Method**: Add a penalty term to the loss function.
- **Penalty Term**: $w_1^2 + w_2^2 + \cdots + w_d^2$

The **regularized loss function** for polynomial regression:

\[
L(w, b) = \sum_{n = 1}^{N}\left(\hat{y}^{(n)} - \left(b + \sum_{j=1}^{d} w_j x^{(n)^j}\right)\right)^2 + \lambda \sum_{j = 1}^{d}(w_j)^2
\]

where $\lambda$ is the **regularization coefficient**.

- **Why prefer smaller weights?**
Smaller weights make the model **less sensitive** to variations or noise in the input, thus yielding **smoother** and more stable predictions.

- **Why is a smoother model better?**
A smoother model is less impacted by input noise, making it more robust and reliable when predicting unseen (testing) data.

- **Impact**: Regularization slightly increases the training error but usually improves performance on testing data by preventing overfitting.

