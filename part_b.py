from part_a import *


def ridge(X, target, lmbda = 0.1):
    """
    Input data, lmbda, and target vriable. return polynomial coefficients.
    Uses ridge regression with np.linalg.solve().
    """
    penalty = np.eye(len(X[0, :]))
    penalty[0, 0] = 0 #avoid punishing bias

    theta = np.linalg.solve(X.T @ X + n * lmbda * penalty, X.T @ target)
    return theta


def plot_score_lmbdas(data, target, pow,  ts, lmbda_min = -8, lmbda_max = 2, nlmbdas = 20):
    """
    Input data, target variable, power, training size, and lmbda values.
    Plots mse, r2, and coefficients.
    """
    lmbdas = np.logspace(lmbda_min, lmbda_max, nlmbdas)
    mses = np.zeros_like(lmbdas, dtype = float)
    R2s = np.zeros_like(lmbdas, dtype = float)

    for p in range(nlmbdas):
        X = design_matrix(data, pow)
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

        theta = ridge(X_train_scaled, y_train_centered, lmbda = lmbdas[p])
        y_tilde = X_test_scaled @ theta
        mse_score, R2_score = mse(y_test_centered, y_tilde), r2_score(y_test_centered, y_tilde)

        mses[p] = mse_score
        R2s[p] = R2_score

    plt.plot(lmbdas, mses, "o-", label = "MSE score")
    plt.xlabel("$\\lambda$")
    plt.ylabel("MSE")
    plt.xscale("log")
    #plt.yscale("log")
    plt.legend()
    plt.show()
    plt.plot(lmbdas, R2s, "o-", label = "$R^2$ score")
    plt.xlabel("$\\lambda $")
    plt.ylabel("$R^2$")
    plt.xscale("log")
    plt.legend()
    plt.show() 

if __name__ == "__main__":

    rs = 2026
    np.random.seed(rs)
    std = 0.1

    n = 100
    x_0 = -1
    x_1 = 1

    x = np.linspace(x_0, x_1, n)
    y = (1 / (1 + (25 * (x ** 2)))) + np.random.normal(0, std, len(x))

    pow = 15
    ts = 0.2
    plot_score_lmbdas(x, y, pow, ts)