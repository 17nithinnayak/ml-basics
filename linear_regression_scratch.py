
import numpy as np

class LinearRegressionScratch:
    def __init__(self, learning_rate=0.01, num_iterations=1000, verbose=False):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.verbose = verbose
        self.w = None
        self.b = None

    def initialize_params(self, n_features):
        self.w = np.zeros((n_features, 1))  # shape: (n, 1)
        self.b = 0

    def fit(self, X, y):
        m, n = X.shape
        y = y.reshape(m, 1)  # ensure column vector
        self.initialize_params(n)

        for i in range(self.num_iterations):
            # Forward pass
            y_hat = np.dot(X, self.w) + self.b

            # Compute loss (MSE)
            cost = (1/m) * np.sum((y_hat - y)**2)

            # Backward pass (gradients)
            dw = (2/m) * np.dot(X.T, (y_hat - y))
            db = (2/m) * np.sum(y_hat - y)

            # Parameter update
            self.w -= self.learning_rate * dw
            self.b -= self.learning_rate * db

            if self.verbose and i % 100 == 0:
                print(f"Iteration {i}, MSE: {cost:.4f}")

    def predict(self, X):
        return np.dot(X, self.w) + self.b

    def mse(self, X, y):
        y_pred = self.predict(X)
        return np.mean((y_pred - y.reshape(-1, 1))**2)

    def rmse(self, X, y):
        return np.sqrt(self.mse(X, y))

    def r2_score(self, X, y):
        y = y.reshape(-1, 1)
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred)**2)
        ss_tot = np.sum((y - np.mean(y))**2)
        return 1 - ss_res / ss_tot
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Generate synthetic data for regression
X, y = make_regression(n_samples=1000, n_features=5, noise=10, random_state=42)

# Normalize features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train linear regression model
model = LinearRegressionScratch(learning_rate=0.01, num_iterations=1000, verbose=True)
model.fit(X_train, y_train)

# Evaluate model
print("Train MSE :", model.mse(X_train, y_train))
print("Test MSE  :", model.mse(X_test, y_test))
print("R2 Score  :", model.r2_score(X_test, y_test))


