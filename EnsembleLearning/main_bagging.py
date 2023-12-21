import sys
import pandas as p
import bagging as bagging
import preprocess_data as preprocess
import matplotlib.pyplot as plt

# DEFINE ATTRIBUTE NAMES 
attribute_names = ["age", "job", "marital", "education", "default", "balance", "housing", "loan",
              "contact", "day", "month", "duration", "campaign", "pdays", "previous", "poutcome"]
cat_attribute_names = ["job", "marital", "education", "contact", "month", "poutcome"]
num_attribute_names = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']
bin_attribute_names = ['default', 'loan', 'housing']
label_name = "label"

# DEFINE POSSIBLE VALUES FOR EACH ATTRIBUTE (AND STORE IN DICT)
bin_attribute_names = bin_attribute_names + num_attribute_names
attribute_values = {}
for name in bin_attribute_names:
    attribute_values[name] = [0, 1]
attribute_values["job"] = ["admin.","unknown","unemployed","management","housemaid","entrepreneur","student",
                            "blue-collar","self-employed","retired","technician","services"]
attribute_values["marital"] = ["married","divorced","single"]
attribute_values["education"] = ["unknown","secondary","primary","tertiary"]
attribute_values["contact"] = ["unknown","telephone","cellular"]
attribute_values["month"] = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
attribute_values["poutcome"] = ["unknown","other","failure","success"]
label_values = [-1, 1]

# READ AND PREPROCESS DATA
df = preprocess.read_data("train.csv", attribute_names, label_name)
df = preprocess.preprocess(df, num_attribute_names, bin_attribute_names, label_name)

# READ AND PREPROCESS TEST DATA
test_df = preprocess.read_data("test.csv", attribute_names, label_name)
test_df = preprocess.preprocess(test_df, num_attribute_names, bin_attribute_names, label_name)
sys.displayhook(test_df)

# GET HYPER-PARAMETERS
T = int(sys.argv[1]) if len(sys.argv) == 3 else 20
training_sample_size = int(sys.argv[2]) if len(sys.argv) == 3 else 400
print(f"T = {T}")
print(f"Training Sample Size = {training_sample_size}")

# TRAIN
trees = bagging.train(df=df, T=T, attribute_names=attribute_names, attribute_values=attribute_values,
                       label_values=label_values, sample_size=training_sample_size)

# TEST ON TEST DATA
bagging.test_and_print_results(df=test_df, trees=trees, attribute_values=attribute_values)


# print("Training model...")
# adaboost_model = boost.adaboost(df=df, t=T, attribute_names=attribute_names, attribute_values=attribute_values, label_values=label_values)
# print("Testing model on test data...")
# error_count = 0
# for i in range(len(test_df.index)):
#     row = test_df.iloc[i]
#     actual_label = row.get("label")
#     result_label = boost.predict(row, adaboost_model, attribute_values)
#     if (actual_label != result_label):
#         error_count += 1
# print("Total Test Errors: ", str(error_count))
# print("Test Accuracy: ", (float(len(test_df.index)) - float(error_count)) / float(len(test_df.index)))

# print("Testing model on training data...")
# error_count = 0
# for i in range(len(df.index)):
#     row = df.iloc[i]
#     actual_label = row.get("label")
#     result_label = boost.predict(row, adaboost_model, attribute_values)
#     if (actual_label != result_label):
#         error_count += 1
# print("Total Training Errors: ", str(error_count))
# print("Training Accuracy: ", (float(len(df.index)) - float(error_count)) / float(len(df.index)))

