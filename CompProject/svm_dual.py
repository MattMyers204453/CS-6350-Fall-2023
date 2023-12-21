import numpy as np
from scipy import optimize

def sgn(v):
    return 1 if v > 0 else -1

def predict(w, x_i):
    return sgn(np.dot(w, x_i))

# Get feature vector (with folded bias) at i
# drop label, add constant value of 1 (for bias), and convert to numpy array for matrix operations.
def x_at_i(i, df):
    return np.array(df.iloc[i][:-1].tolist() + [1])

def x_at_i_no_label(i, df):
    return np.array(df.iloc[i].tolist() + [1])

# Objective function in vector/matrix form
def objective_func(alphas, YXXY):
    step1 = np.dot(alphas.T, YXXY)
    step2 = np.dot(step1, alphas)
    return 0.5 * step2 - np.sum(alphas)

def sum_to_zero_constraint(alphas, Y):
    return np.dot(alphas, Y)
 
def recover_w(X, Y_vector, alphas):
    num_attributes = X.shape[1]
    w = np.zeros(num_attributes)
    for i in range(len(X)):
        w += alphas[i] * Y_vector[i] * X[i]
    return w

# find average b
def recover_b(w, X, Y_vector, alphas, C):
    close_to_zero_threshold = 0.00001
    indices = np.where((alphas > close_to_zero_threshold) & (alphas < C))[0]
    if len(indices) == 0:
        return 0
    sum = 0.0
    for i in indices:
        sum += (Y_vector[i] - np.dot(w, X[i]))
    return sum / float(len(indices)) if sum != 0 else 0

def train(df, C):
    N = len(df.index)
    X = np.array(df.drop('label', axis=1).values)
    Y_vector = np.array(df['label'].values)
    Y = np.diag(Y_vector)

    # These products are constant, so we calculate them once and pass them to objective function to speed things up
    XX = np.dot(X, X.T)
    YXX = np.dot(Y, XX)
    YXXY = np.dot(YXX, Y)

    constraints = ({'type': 'eq', 'fun': sum_to_zero_constraint, 'args': (Y_vector,), 'jac': lambda alphas, Y: Y})
    bounds = [(0, C) for _ in range(N)]
    result = optimize.minimize(fun=objective_func, args=(YXXY,), x0=np.zeros(N), method="SLSQP", constraints=constraints, bounds=bounds)
    alphas = result.x
    w = recover_w(X, Y_vector, alphas)
    b = recover_b(w, X, Y_vector, alphas, C)
    w_folded_b = np.append(w, b)
    return w_folded_b

def test_and_print_results(df, w):
    errors = 0
    for i in range(len(df.index)):
        feature_vector = x_at_i(i, df)
        ground_truth_label = df.iloc[i].get("label")
        prediction = predict(w, feature_vector) 
        if prediction != ground_truth_label:
            errors += 1
    print(f"Total examples misclassified: {errors}")
    print(f"Accuracy: {(((float(len(df.index)) - errors) / float(len(df.index))) * 100)}%")

def make_predictions(df, w):
    predictions = [0] * len(df)
    for i in range(len(df.index)):
        feature_vector = x_at_i_no_label(i, df)
        predictions[i] = predict(w, feature_vector) 
    print("Predictions generated")
    return predictions