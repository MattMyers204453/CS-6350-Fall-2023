import numpy as np

def sgn(v):
    return 1 if v > 0 else -1

def predict(w, x_vector):
    return sgn(np.dot(w, x_vector))

# Get feature vector (with folded bias) at i
# drop label, add constant value of 1 (for bias), and convert to numpy array for matrix operations.
def x_i(i, df):
    return np.array(df.iloc[i][:-1].tolist() + [1])

def update(w, x_vector, y, r):
    return np.add(w, r * y * x_vector)

def train(df, attribute_names, T, r):
    # add space for bias to be folded into weight vector
    w = np.array([0] * (len(attribute_names) + 1))
    m = len(df.index)
    w_average = [0] * (len(w))
    for _ in range(T):
        # shuffle the data before every epoch
        df = df.sample(frac=1)
        for i in range(m):
            ground_truth_label = df.iloc[i].get("genuine")
            feature_vector = x_i(i, df)
            prediction = predict(w, feature_vector)
            if prediction != ground_truth_label:
                w = update(w, feature_vector, ground_truth_label, r)
            w_average = np.add(w_average, w)
    return w_average

def test_and_print_results(test_df, w_average):
    errors = 0
    for i in range(len(test_df.index)):
        feature_vector = x_i(i, test_df)
        ground_truth_label = test_df.iloc[i].get("genuine")
        prediction = predict(w_average, feature_vector) 
        if prediction != ground_truth_label:
            errors += 1
    print(f"Total examples misclassified: {errors}")
    print(f"Accuracy: {(((float(len(test_df.index)) - errors) / float(len(test_df.index))) * 100)}%")
    print(f"Learned average weights: {w_average}")