import read_data as read
import sys
import svm_primal as svm_primal
import svm_dual as svm_dual
import svm_kernel as svm_kernel

# Load data, preproccess data
attribute_names = ["variance", "skewness", "kurtosis", "entropy"]
label_name = "genuine"
df = read.read_data(file_name="bank-note/train.csv",attribute_names=attribute_names, label_name=label_name)
df = read.preprocess_data(df)
test_df = read.read_data(file_name="bank-note/test.csv", attribute_names=attribute_names, label_name=label_name)
test_df = read.preprocess_data(test_df)

T = 100
r = 0.001
a = 0.5
# C = 100.0 / 872.0
C = 500.0 / 872.0
# C = 700.0 / 872.0
gamma = 0.1

# print(f"Epochs: {T}; Learning rate: {r}")
# print("Training primal SVM...")
# learned_weights = svm_primal.train(df=df, attribute_names=attribute_names, T=T, r=r, C=C, a=a)
# print("Testing primal SVM on test data...")
# svm_primal.test_and_print_results(df=test_df, w=learned_weights)
# print(f"Learned parameters: {learned_weights}")
# print("Testing primal SVM on training data...")
# svm_primal.test_and_print_results(df=df, w=learned_weights)

# print("Training dual SVM...")
# learned_weights = svm_dual.train(df=df, C=C)
# print("Testing dual SVM on test data...")
# svm_dual.test_and_print_results(df=test_df, w=learned_weights)

print("Training kernel SVM...")
model = svm_kernel.train(df=df, C=C, gamma=gamma)
print("Testing kernel SVM on test data...")
svm_kernel.test_and_print_results(df=test_df, model=model)