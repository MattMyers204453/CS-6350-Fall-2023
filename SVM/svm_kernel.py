import numpy as np
from scipy import optimize

def sgn(v):
    return 1 if v > 0 else -1

def predict(X_to_predict, model):
    X, Y_vector, support_vector_indices, alphas, b, gamma = model
    sum= 0.0
    for i in support_vector_indices:
        kernel_value = gaussian_kernel(X[i], X_to_predict, gamma)
        sum += alphas[i] * Y_vector[i] * kernel_value
    return sgn(sum + b)

# Get feature vector (WITHOUT folded bias) at i
# drop label, add constant value of 1 (for bias), and convert to numpy array for matrix operations.
def x_at_i(i, df):
    return np.array(df.iloc[i][:-1])

# Objective function in vector/matrix form
def objective_func(alphas, K):
    step1 = np.dot(alphas.T, K)
    step2 = np.dot(step1, alphas)
    return 0.5 * step2 - np.sum(alphas)

def gaussian_kernel(x_i, x_j, gamma):
    return np.exp(-np.linalg.norm(x_i - x_j)**2 / gamma)

def get_gram_matrix(X, gamma):
    N = len(X)
    K = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            K[i, j] = gaussian_kernel(X[i], X[j], gamma)
    return K

def sum_to_zero_constraint(alphas, Y):
    return np.dot(alphas, Y)

# find average b
def recover_b_and_get_support_vector_locations(X, Y_vector, alphas, C, gamma):
    close_to_zero_threshold = 0.00001
    indices = np.where((alphas > close_to_zero_threshold) & (alphas < C))[0]
    if len(indices) == 0:
        return 0
    sum = 0.0
    for i in indices:
        sum_w_using_kernal = 0.0
        for j in range(len(X)):
            sum_w_using_kernal += alphas[j] * Y_vector[j] * gaussian_kernel(X[j], X[i], gamma)
        sum += (Y_vector[i] - sum_w_using_kernal)
    return (sum, indices)

def train(df, C, gamma):
    N = len(df.index)
    X = np.array(df.drop('genuine', axis=1).values)
    Y_vector = np.array(df['genuine'].values)
    Y = np.diag(Y_vector)

    # These products are constant, so we calculate them once and pass them to objective function to speed things up
    # Calculate gram matrix
    K = get_gram_matrix(X, gamma)
    YK = np.dot(Y, K)
    YKY = np.dot(YK, Y)

    constraints = ({'type': 'eq', 'fun': sum_to_zero_constraint, 'args': (Y_vector,), 'jac': lambda alphas, Y: Y})
    bounds = [(0, C) for _ in range(N)]
    result = optimize.minimize(fun=objective_func, args=(YKY,), x0=np.zeros(N), method="SLSQP", constraints=constraints, bounds=bounds)
    alphas = result.x
    b, support_vector_indices = recover_b_and_get_support_vector_locations(X, Y_vector, alphas, C, gamma)
    return (X, Y_vector, support_vector_indices, alphas, b, gamma)

def test_and_print_results(df, model):
    errors = 0
    for i in range(len(df.index)):
        feature_vector = x_at_i(i, df)
        ground_truth_label = df.iloc[i].get("genuine")
        prediction = predict(feature_vector, model) 
        if prediction != ground_truth_label:
            errors += 1
    print(f"Total examples misclassified: {errors}")
    print(f"Accuracy: {(((float(len(df.index)) - errors) / float(len(df.index))) * 100)}%")