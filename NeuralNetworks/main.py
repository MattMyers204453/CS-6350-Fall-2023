import read_data as read
import argparse
import neural_network as neural_network

# Load data, preproccess data
attribute_names = ["variance", "skewness", "kurtosis", "entropy"]
label_name = "genuine"
df = read.read_data(file_name="bank-note/train.csv",attribute_names=attribute_names, label_name=label_name)
df = read.preprocess_data(df)
test_df = read.read_data(file_name="bank-note/test.csv", attribute_names=attribute_names, label_name=label_name)
test_df = read.preprocess_data(test_df)

# Get hyperparameters
parser = argparse.ArgumentParser()
parser.add_argument('--T', type=int, default=10, help='...')
parser.add_argument('--r', type=float, default=0.1, help='...')
parser.add_argument('--d', type=float, default=0.3, help='...')
parser.add_argument('--width', type=int, default=50, help='...')

args = parser.parse_args()
T = args.T
r = args.r
d = args.d
WIDTH = args.width

# Train, test, and print results
# T = 5
# r = 0.1
# d = 0.3
# WIDTH = 20
print(f"Epochs: {T}; Learning rate = {r}; d = {d}; Width = {WIDTH}")
neural_network.train(df=df, attribute_names=attribute_names, label_name=label_name, T=T, r=r, d=d, WIDTH=WIDTH)
neural_network.test_and_print_results(df=df, test_df=test_df, label_name=label_name)