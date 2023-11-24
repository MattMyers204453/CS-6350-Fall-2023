import numpy as np
from scipy import optimize

def sgn(v):
    return 1 if v > 0 else -1

def predict(w, x_vector):
    return sgn(np.dot(w, x_vector))

# Get feature vector (with folded bias) at i
# drop label, add constant value of 1 (for bias), and convert to numpy array for matrix operations.
def x_at_i(i, df):
    return np.array(df.iloc[i][:-1].tolist() + [1])

# Objective function in vector/matrix form
def objective_func(alphas, X, Y):
    XX = np.dot(X, X.T) 
    Y = np.diag(Y)
    step1 = np.dot(Y, XX)
    step2 = np.dot(step1, Y)
    step3 = np.dot(alphas.T, step2)
    step4 = np.dot(step3, alphas)
    return 0.5 * step4 - np.sum(alphas)


def sum_to_zero_constraint(alphas, Y):
    return np.dot(alphas, Y)

def train(df, C):
    N = len(df.index)
    X = np.array(df.drop('genuine', axis=1).values)
    Y = np.array(df['genuine'].values)
    constraints = ({'type': 'eq', 'fun': sum_to_zero_constraint, 'args': (Y,), 'jac': lambda alphas, Y: Y})
    bounds = [(0, C) for _ in range(N)]
    print("Training...")
    alphas_result = optimize.minimize(fun=objective_func, args=(X, Y), x0=np.zeros(N), method="SLSQP", constraints=constraints, bounds=bounds)
    print(alphas_result)