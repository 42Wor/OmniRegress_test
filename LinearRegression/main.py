from omniregress import (__version__, LinearRegression,
                         PolynomialRegression,
                         LogisticRegression,
                         RidgeRegression)
import numpy as np

print("Omniregress version:", __version__)


def test_linear_regression():
    print("\n--- Simple Linear Regression Test ---")
    # X should be a 2D array of shape (n_samples, n_features)
    # Even with one feature, it needs to be 2D, so we reshape it.
    X = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
    y = np.array([3, 5, 7, 9, 11]) # y is a 1D array of targets

    model = LinearRegression()
    model.fit(X, y)
    
    print(f"Intercept: {model.intercept:.2f}")
    print(f"Coefficients: {model.coefficients[0]:.2f}")

    # The data for prediction must also be a 2D array
    X_new = np.array([6, 7]).reshape(-1, 1)
    predictions = model.predict(X_new)
    print(f"Predictions for {X_new.flatten()}: {np.round(predictions, 2)}")
    # Expected output for this data (y = 2x + 1) is [13, 15]


def linear_regression_long_test():
    print("\n--- Large Single Feature Test (with a real pattern) ---")
    
    # Create sample data with a clear linear relationship: y = 2x + 1 + noise
    # This is a much better test than using two completely random vectors.
    X = np.random.rand(100_000, 1) * 10 # 100,000 samples, 1 feature
    noise = np.random.randn(100_000) * 0.5 # Add some random noise
    y = 2 * X.flatten() + 1 + noise # .flatten() makes X 1D for the formula

    # Initialize and fit model
    model = LinearRegression()
    model.fit(X, y)

    # Show parameters. They should be close to 1 (intercept) and 2 (coefficient).
    print(f"Learned Intercept: {model.intercept:.2f} (Expected ~1.0)")
    print(f"Learned Coefficient: {model.coefficients[0]:.2f} (Expected ~2.0)")

    # Calculate score. It should be high since there is a clear pattern.
    r2 = model.score(X, y)
    print(f"R² score: {r2:.4f}")


def multi_linear_regression_long_test():
    print("\n--- Multiple Features Test (with a real pattern) ---")

    # The shape should be (n_samples, n_features).
    # Let's use 1000 samples and 2 features.
    X_multi = np.random.rand(1000, 2)
    
    # y should be a 1D array of length n_samples.
    # Let's create a relationship: y = 3*x1 + 5*x2 + 2 + noise
    noise = np.random.randn(1000) * 0.2
    y_multi = 3 * X_multi[:, 0] + 5 * X_multi[:, 1] + 2 + noise

    model_multi = LinearRegression()
    model_multi.fit(X_multi, y_multi)

    print(f"Learned Intercept: {model_multi.intercept:.2f} (Expected ~2.0)")
    print(f"Learned Coefficients: {np.round(model_multi.coefficients, 2)} (Expected ~[3.0, 5.0])")

    # Predict on new data. Shape must be (n_new_samples, n_features)
    X_test_multi = np.array([
        [1, 3],  # 1 sample with 2 features
        [4, 2]   # 1 sample with 2 features
    ])
    predictions_multi = model_multi.predict(X_test_multi)
    print(f"Predictions for new data: {np.round(predictions_multi, 2)}")


if __name__ == "__main__":
    test_linear_regression()
    linear_regression_long_test()
    multi_linear_regression_long_test()