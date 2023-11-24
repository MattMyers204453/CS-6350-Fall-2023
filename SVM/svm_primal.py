import numpy as np

def sgn(v):
    return 1 if v > 0 else -1

def predict(w, x_i):
    return sgn(np.dot(w, x_i))

# Get feature vector (with folded bias) at i
# drop label, add constant value of 1 (for bias), and convert to numpy array for matrix operations.
def x_at_i(i, df):
    return np.array(df.iloc[i][:-1].tolist() + [1])

def hinge_loss(df, w, C):
    empirical_loss = 0
    for i in range(len(df.index)):
        x_i = x_at_i(i, df)
        y_i = df.iloc[i].get("genuine")
        empirical_loss += max(0, 1 - y_i * np.dot(w.T, x_i))
    return (np.dot(w.T, w) / 2) + C * empirical_loss

# Train using stochastic sub-gradient descent
def train(df, attribute_names, T=100, r=0.001, C=(100.0 / 872.0), a=0.5, conv_threshold=0.001, break_threshold=6):
    w = np.array([0] * (len(attribute_names) + 1))
    N = len(df.index)
    small_difference_consecutive_count = 0
    previous_hinge_loss = 0
    for t in range(T):
        #r = r / float(1 + t)
        r = r / float(1 + (r*t) / float(a))
        # shuffle the data before every epoch
        df = df.sample(frac=1)
        for i in range(len(df.index)):
            x_i = x_at_i(i, df)
            y_i = df.iloc[i].get("genuine")
            if (y_i * np.dot(w, x_i)) <= 1:
                gradient = np.append(w[:-1], 0)
                w = w - (r * gradient) + (r * C * N * y_i * x_i)
            else:
                gradient = np.append(w[:-1], 0)
                w = w - (r * gradient)
        # Calculate the difference in loss between this and the last epoch. Because
        # we are training stochastically, this can happen by chance, even if far away from minimum.
        # Only break if the difference is small several times in a row
        if (hinge_loss(df, w, C) - previous_hinge_loss) <= conv_threshold:
            small_difference_count += 1
            if small_difference_consecutive_count >= break_threshold:
                print(f"CONVERGED AT EPOCH: {t}")
                break
        else:
            small_difference_consecutive_count = 0

    return w

def test_and_print_results(df, w):
    errors = 0
    for i in range(len(df.index)):
        feature_vector = x_at_i(i, df)
        ground_truth_label = df.iloc[i].get("genuine")
        prediction = predict(w, feature_vector) 
        if prediction != ground_truth_label:
            errors += 1
    print(f"Total examples misclassified: {errors}")
    print(f"Accuracy: {(((float(len(df.index)) - errors) / float(len(df.index))) * 100)}%")