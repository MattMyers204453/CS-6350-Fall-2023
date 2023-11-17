import os
import read_data as read
import standard_perceptron as s_perceptron
import voted_perceptron as v_perceptron
import average_perceptron as a_perceptron
import argparse

# Load data, preproccess data
attribute_names = ["variance", "skewness", "kurtosis", "entropy"]
label_name = "genuine"
df = read.read_data(file_name="bank-note/train.csv",attribute_names=attribute_names, label_name=label_name)
df = read.preprocess_data(df)
test_df = read.read_data(file_name="bank-note/test.csv", attribute_names=attribute_names, label_name=label_name)
test_df = read.preprocess_data(test_df)

# Get hyperparameters
parser = argparse.ArgumentParser()
parser.add_argument('--type', type=str, default="standard", help='...')
parser.add_argument('--T', type=int, default=10, help='...')
parser.add_argument('--r', type=float, default=1.0, help='...')
args = parser.parse_args()

perceptron_type = args.type
T = args.T
r = args.r

# Train, test, and print results
print(f"Epochs: {T}; Learning rate = {r}")
print(f"Training model ({perceptron_type} perceptron)...")

if perceptron_type == 'standard':
    weights = s_perceptron.train(df=df, attribute_names=attribute_names, T=T, r=r)
    print(f"Testing model ({perceptron_type} perceptron)...")
    s_perceptron.test_and_print_results(test_df=test_df, w=weights)
if perceptron_type == "voted":
    w_and_counts = v_perceptron.train(df=df, attribute_names=attribute_names, T=T, r=r)
    print(f"Testing model ({perceptron_type} perceptron)...")
    v_perceptron.test_and_print_results(test_df=test_df, w_and_counts=w_and_counts)
if perceptron_type == "average":
    w_average = a_perceptron.train(df=df, attribute_names=attribute_names, T=T, r=r)
    print(f"Testing model ({perceptron_type} perceptron)...")
    a_perceptron.test_and_print_results(test_df=test_df, w_average=w_average)