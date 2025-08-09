# Machine Learning Cheat Sheet — Andrew Ng Specialization

---

## 1. Linear Regression

### Core Idea
Predicts a continuous target by fitting a straight line (or hyperplane) to minimize squared error.

### Challenges
- Underfitting with too simple models
- Sensitivity to outliers
- Feature scaling importance

### Cost Function
$$
J(\theta) = \frac{1}{2m} \sum_{i=1}^m (h_\theta(x^{(i)}) - y^{(i)})^2
$$
where $h_\theta(x) = \theta^T x$.

### Formulas
Gradient Descent update:
$$
\theta_j := \theta_j - \alpha \frac{1}{m} \sum_{i=1}^m (h_\theta(x^{(i)}) - y^{(i)})x_j^{(i)}
$$

### Code Example
```python
import numpy as np
X = np.c_[np.ones(3), [1, 2, 3]]
y = np.array([1, 2, 3])
theta = np.zeros(2)
alpha = 0.1
for _ in range(1000):
    theta -= alpha / len(y) * X.T @ (X @ theta - y)
print(theta)
```

---

## 2. Logistic Regression

### Core Idea
Binary classification using the logistic (sigmoid) function to model probabilities.

### Challenges
- Non-linear boundaries require feature engineering
- Imbalanced data handling

### Cost Function
$$
J(\theta) = -\frac{1}{m} \sum_{i=1}^m \left[ y^{(i)} \log h_\theta(x^{(i)}) + (1 - y^{(i)}) \log(1 - h_\theta(x^{(i)})) \right]
$$
where $h_\theta(x) = \frac{1}{1+e^{-\theta^T x}}$.

### Formulas
Gradient:
$$
\theta_j := \theta_j - \alpha \frac{1}{m} \sum_{i=1}^m (h_\theta(x^{(i)}) - y^{(i)})x_j^{(i)}
$$

### Code Example
```python
import numpy as np
sigmoid = lambda z: 1 / (1 + np.exp(-z))
X = np.c_[np.ones(3), [1, 2, 3]]
y = np.array([0, 1, 1])
theta = np.zeros(2)
alpha = 0.1
for _ in range(1000):
    theta -= alpha / len(y) * X.T @ (sigmoid(X @ theta) - y)
print(theta)
```

---

## 3. Neural Networks

### Core Idea
Stacked layers of neurons transform inputs through weighted sums and non-linear activations; trained via forward pass and backpropagation.

### Challenges
- Vanishing/exploding gradients
- Overfitting
- Choosing architecture/hyperparameters

### Cost Function
Binary classification example:
$$
J(W, b) = -\frac{1}{m} \sum_{i=1}^m \left[ y^{(i)} \log \hat{y}^{(i)} + (1 - y^{(i)}) \log(1 - \hat{y}^{(i)}) \right]
$$

### Formulas
Backpropagation updates:
$$
W^{[l]} := W^{[l]} - \alpha \frac{\partial J}{\partial W^{[l]}}
$$
(similar for $b^{[l]}$).

### Code Example
```python
import numpy as np
sigmoid = lambda z: 1 / (1 + np.exp(-z))
X = np.random.randn(3,2); y = np.array([[1],[0],[1]])
W = np.random.randn(2,1); b = 0; alpha = 0.1
for _ in range(1000):
    Z = X @ W + b
    A = sigmoid(Z)
    dW = X.T @ (A - y) / len(y)
    db = np.mean(A - y)
    W -= alpha * dW; b -= alpha * db
```

---

## 4. Applying Best Practices in ML

### Core Idea
Diagnose and fix bias/variance issues; use regularization; monitor with learning curves; perform error analysis.

### Challenges
- Misinterpreting bias/variance
- Over-regularization
- Incorrect metrics

### Key Points
- **Bias**: High training error
- **Variance**: Low training error, high dev/test error
- L2 regularization: $\lambda \sum \theta_j^2$

### Code Example
```python
from sklearn.linear_model import Ridge
model = Ridge(alpha=1.0).fit(X, y)
```

---

## 5. Decision Trees

### Core Idea
Recursively split features to maximize purity (e.g., information gain or Gini reduction).

### Challenges
- Overfitting with deep trees
- Sensitive to small changes in data

### Cost Function
Classification: maximize information gain  
$$
IG = H_{\text{parent}} - \sum_k \frac{n_k}{n} H_k
$$

### Code Example
```python
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(max_depth=3).fit(X, y)
```

---

## 6. Random Forest

### Core Idea
Ensemble of decision trees trained on bootstrap samples with feature randomness; reduces variance.

### Challenges
- Less interpretable
- Slower inference with many trees

### Code Example
```python
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=100).fit(X, y)
```

---

## 7. XGBoost

### Core Idea
Sequentially builds trees to correct previous errors; uses gradient boosting with regularization.

### Challenges
- Sensitive to hyperparameters
- Risk of overfitting with too many trees

### Code Example
```python
from xgboost import XGBClassifier
clf = XGBClassifier().fit(X, y)
```

---

## 8. K-means Clustering

### Core Idea
Unsupervised algorithm that groups data into $k$ clusters by minimizing within-cluster variance.

### Challenges
- Choosing $k$
- Sensitivity to initialization

### Cost Function
$$
J = \sum_{i=1}^m \| x^{(i)} - \mu_{c^{(i)}} \|^2
$$

### Code Example
```python
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=3).fit(X)
```

---

## 9. Anomaly Detection

### Core Idea
Model normal data distribution (e.g., Gaussian) and flag low-probability points as anomalies.

### Challenges
- Assumes distribution form
- Feature scaling critical

### Formula
Gaussian probability:
$$
p(x) = \frac{1}{\sqrt{2\pi\sigma^2}} e^{-\frac{(x - \mu)^2}{2\sigma^2}}
$$

### Code Example
```python
from scipy.stats import norm
mu, sigma = np.mean(X), np.std(X)
p = norm.pdf(X, mu, sigma)
anomalies = X[p < 0.01]
```

---

## 10. Reinforcement Learning

### Core Idea
Agent interacts with environment to maximize cumulative reward; learns optimal policy via trial and error.

### Challenges
- Exploration vs exploitation
- Delayed rewards

### Q-learning Update
$$
Q(s,a) \leftarrow Q(s,a) + \alpha \left[ r + \gamma \max_{a'} Q(s',a') - Q(s,a) \right]
$$

### Code Example
```python
import numpy as np
Q = np.zeros((5,2))
alpha, gamma = 0.1, 0.9
s, a, r, s_next = 0, 1, 1, 2
Q[s,a] += alpha * (r + gamma * np.max(Q[s_next]) - Q[s,a])
```
