import numpy as np

def sgn(v):
    return 1 if v > 0 else -1

def predict(w, x_vector):
    return sgn(np.dot(w, x_vector))

def sum_predict(w_and_counts, x_vector):
    return sgn(sum(c * sgn(np.dot(w, x_vector)) for w, c in w_and_counts))

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
    w_and_counts = []
    curr_count = 1
    for _ in range(T):
        # shuffle the data before every epoch
        df = df.sample(frac=1)
        for i in range(m):
            ground_truth_label = df.iloc[i].get("genuine")
            feature_vector = x_i(i, df)
            prediction = predict(w, feature_vector)
            if prediction != ground_truth_label:
                w_and_counts.append((w, curr_count))
                curr_count = 1
                w = update(w, feature_vector, ground_truth_label, r)
            else:
                curr_count += 1
    return w_and_counts

def test_and_print_results(test_df, w_and_counts):
    errors = 0
    for i in range(len(test_df.index)):
        feature_vector = x_i(i, test_df)
        ground_truth_label = test_df.iloc[i].get("genuine")
        prediction = sum_predict(w_and_counts, feature_vector) 
        if prediction != ground_truth_label:
            errors += 1
    print(f"Total examples misclassified: {errors}")
    print(f"Accuracy: {(((float(len(test_df.index)) - errors) / float(len(test_df.index))) * 100)}%")