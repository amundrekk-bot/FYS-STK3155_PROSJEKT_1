import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error as mse, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

rs = 2026
np.random.seed(rs)
std = 0.1

n = 100
x_0 = -1
x_1 = 1

x = np.linspace(x_0, x_1, n)
y = (1 / (1 + (25 * (x ** 2)))) + np.random.normal(0, std, len(x))

def design_matrix(data, degree):
    """
    Input data and degree of polynomial. Return design matrix
    """
    X = np.column_stack([data ** j for j in range(degree + 1)])
    return X

def OLS(dsgn_mtrx, target):
    """
    Input data and target variable. Return the polynomial coefficients.
    Uses Moore-Penrose pseudoinverse.
    """
    theta = np.linalg.pinv(dsgn_mtrx) @ target
    return theta

def ridge(X, target, lmbda = 0.1):
    """
    Input data, lmbda, and target vriable. return polynomial coefficients.
    Uses ridge regression with np.linalg.solve().
    """
    theta = np.linalg.solve(X.T @ X + n * np.eye(len(X[0, :])), X.T @ target)
    return theta


def plot_score(data, target, ts, degree_max, method):
    """
    Input data, target variable, training size, and maximum degree of polnomial.
    Plots mse, r2, and coefficients.
    """
    powers = np.arange(0, degree_max + 1)
    mses = np.zeros_like(powers, dtype = float)
    R2s = np.zeros_like(powers, dtype = float)

    for p in range(degree_max + 1):
        X = design_matrix(data, p)
        X_train, X_test, y_train, y_test = train_test_split(X, target, test_size = ts, random_state = rs)

        # making sure to avoid data leakage:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        # setting intercept columns to constant ones
        X_train_scaled[:, 0] = np.ones(len(X_train[:, 0]))
        X_test_scaled[:, 0] = np.ones(len(X_test[:, 0]))

        y_train_centered = y_train - np.mean(y_train)
        y_test_centered = y_test - np.mean(y_train)

        theta = method(X_train_scaled, y_train_centered)
        y_tilde = X_test_scaled @ theta
        mse_score, R2_score = mse(y_test_centered, y_tilde), r2_score(y_test_centered, y_tilde)

        mses[p] = mse_score
        R2s[p] = R2_score


    plt.plot(powers, mses, "o-", label = "MSE score")
    plt.xlabel("Powers")
    plt.ylabel("MSE")
    plt.legend()
    plt.show()
    plt.plot(powers, R2s, "o-", label = "$R^2$ score")
    plt.xlabel("Powers")
    plt.ylabel("$R^2$")
    plt.legend()
    plt.show()
    



plot_score(x, y, ts = 0.2, degree_max = 15, method = OLS)