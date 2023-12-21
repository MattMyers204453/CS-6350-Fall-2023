from threading import Thread
import ID3_forest as id3

def predict(x_i, trees, attribute_values):
    sum = 0.0
    for i in range(len(trees)):
        prediction = id3.predict(row=x_i, root=trees[i], attribute_values=attribute_values)
        sum += int(prediction)
    return 1 if sum > 0 else 0


def train_single_tree(df_sample, attribute_names, attribute_values, label_values, trees, i):
    trees[i] = id3.ID3(df_sample, attribute_names, attribute_values, label_values)

def train(df, T, attribute_names, attribute_values, label_values, sample_size):
    trees = [None] * T
    threads = []
    for i in range(T):
        df_sample = df.sample(n=sample_size, replace=True)
        threads.append(Thread(target=train_single_tree, args=(df_sample, attribute_names, attribute_values, label_values, trees, i)))
    for i, th in enumerate(threads):
        print(f"--Training tree {i} in thread {i}...")
        th.start()
        # th.join()
        # print(f"--Tree {i} in thread {i} completed training...")
    for i, th in enumerate(threads):
        th.join()
        print(f"--Tree {i} in thread {i} completed training...")
    return trees

def test_and_print_results(df, trees, attribute_values):
    errors = 0
    for i in range(len(df.index)):
        x_i = df.iloc[i]
        ground_truth_label = x_i.get("label")
        prediction = predict(x_i, trees, attribute_values)
        if (ground_truth_label != prediction):
            errors += 1
    print("Total Test Errors: ", str(errors))
    print("Test Accuracy: ", (float(len(df.index)) - float(errors)) / float(len(df.index)))

def make_predictions(df, trees, attribute_values):
    predictions = [0] * len(df)
    for i in range(len(df.index)):
        x_i = df.iloc[i]
        predictions[i] = predict(x_i, trees, attribute_values)
    print("Predictions generated")
    return predictions