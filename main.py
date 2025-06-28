from omniregress import LinearRegression, PolynomialRegression, LogisticRegression

def test_linear_regression():
    X = [1, 2, 3, 4, 5]
    y = [3, 5, 7, 9, 11]
    model = LinearRegression()
    model.fit(X, y)
    print("Linear Regression:", model.coefficients, model.intercept)
    print("Predictions:", model.predict([6, 7]))

def test_polynomial_regression():
    X = [-1.0, -0.5, 0.0, 0.5, 1.0]
    y = [x * x for x in X]
    model = PolynomialRegression(degree=2)
    model.fit(X, y)
    print("Polynomial Regression:", model.coefficients, model.intercept)
    print("Predictions:", model.predict([1.5, -1.5]))

def test_logistic_regression():
    X = [[0.5], [1.0], [1.5], [2.0], [2.5], [3.0]]
    y = [0, 0, 0, 1, 1, 1]
    model = LogisticRegression(learning_rate=0.1, max_iter=1000)
    model.fit(X, y)
    print("Logistic Regression:", model.coefficients, model.intercept)
    print("Probabilities:", model.predict_proba([[1.0], [2.0], [3.5]]))
    print("Predictions:", model.predict([[1.0], [2.0], [3.5]]))

if __name__ == "__main__":
    test_linear_regression()
    test_polynomial_regression()
    test_logistic_regression()