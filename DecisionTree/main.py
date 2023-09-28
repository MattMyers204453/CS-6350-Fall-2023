#from sys import displayhook
import sys
import pandas as p
from ID3 import ID3, ID3_traverse_tree
import read_data as read

# READ DATA INTO DATAFRAME
attribute_names = ["buying", "maint", "doors", "persons", "lug_boot", "safety"]
label_name = "label"
df = read.read_data("train.csv", attribute_names, label_name)
sys.displayhook(df)

# DEFINE ATTRIBUTE AND LABEL VALUES
attribute_values = {}
attribute_values["buying"] = ["vhigh", "high", "med", "low"]
attribute_values["maint"] = ["vhigh", "high", "med", "low"]
attribute_values["doors"] = ["2", "3", "4", "5more"]
attribute_values["persons"] = ["2", "4", "more"]
attribute_values["lug_boot"] = ["small", "med", "big"]
attribute_values["safety"] = ["low", "med", "high"]
label_values = ["unacc", "acc", "good", "vgood"]

# TRAIN ID3 
root = ID3(df, attribute_names, attribute_values, label_values)

# READ TEST DATA INTO DATAFRAME
test_df = read.read_data("test.csv", attribute_names, label_name)

# TEST ID3 AGAINST TRAINING DATA
error_count = 0
for i in range(len(test_df.index)):
    # get example i from dataset
    row = test_df.iloc[i]
    # get correct label
    actual_label = row.get("label")
    # get predicted label
    result_label = ID3_traverse_tree(row, root, attribute_values)
    # increment error if they don't match
    if (actual_label != result_label):
        error_count += 1
print("Total Number of Errors: ", str(error_count))
print("Accuracy: ", (float(len(test_df.index)) - float(error_count)) / float(len(test_df.index)) * 100, "%")
print("Error Rate: ", (1 - (float(len(test_df.index)) - float(error_count)) / float(len(test_df.index))) * 100, "%")
