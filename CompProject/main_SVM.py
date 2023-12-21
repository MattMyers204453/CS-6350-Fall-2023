import os
import read_data as preprocess
import svm_dual as svm_dual
import pandas as p
import sys

# DEFINE FEATURES
attribute_names = ["age","workclass","fnlwgt","education","education.num","marital.status","occupation", "relationship","race", "sex","capital.gain","capital.loss","hours.per.week","native.country"]
label_name = "income>50K"
categorical_attributes = ["workclass","education","marital.status","occupation", "relationship","race", "sex","native.country"]
numerical_attributes = ["age","fnlwgt","education.num","capital.gain","capital.loss","hours.per.week"]

# READ TRAINING DATA
print("Reading training data...")
df = preprocess.read_training_data("train_final.csv", attribute_names, label_name)
# Rename label column
label_name = "label"
df = df.rename(columns={df.columns[-1]: label_name})

# PREPROCESS TRAINING DATA
print("Preprocessing training data...")
df = preprocess.preprocess_training_data(df=df, attribute_names=attribute_names, numerical_attributes=numerical_attributes)

# READ TEST DATA
print("Reading test data...")
df_test = preprocess.read_test_data("test_final.csv", attribute_names, label_name)

# PREPROCESS TEST DATA
df_test = preprocess.preprocess_test_data(df=df_test, attribute_names=attribute_names, numerical_attributes=numerical_attributes)

# DEFINE ATTRIBUTE VALUES FOR TRAINING
attribute_values = {}
attribute_values["age"] = [0, 1]
attribute_values["workclass"] = ["Private", "Self-emp-not-inc", "Self-emp-inc", "Federal-gov", "Local-gov", "State-gov", "Without-pay", "Never-worked"]
attribute_values["fnlwgt"] = [0, 1]
attribute_values["education"] = ["Bachelors", "Some-college", "11th", "HS-grad", "Prof-school", "Assoc-acdm", "Assoc-voc", "9th", "7th-8th", "12th", "Masters", "1st-4th", "10th", "Doctorate", "5th-6th", "Preschool"]
attribute_values["education.num"] = [0, 1]
attribute_values["marital.status"] = ["Married-civ-spouse", "Divorced", "Never-married", "Separated", "Widowed", "Married-spouse-absent", "Married-AF-spouse"]
attribute_values["occupation"] = ["Tech-support", "Craft-repair", "Other-service", "Sales", "Exec-managerial", "Prof-specialty", "Handlers-cleaners", "Machine-op-inspct", "Adm-clerical", "Farming-fishing", "Transport-moving", "Priv-house-serv", "Protective-serv", "Armed-Forces"]
attribute_values["relationship"] = ["Wife", "Own-child", "Husband", "Not-in-family", "Other-relative", "Unmarried"]
attribute_values["race"] = ["White", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other", "Black"]
attribute_values["sex"] = ["Female", "Male"]
attribute_values["capital.gain"] = [0, 1]
attribute_values["capital.loss"] = [0, 1]
attribute_values["hours.per.week"] = [0, 1]
attribute_values["native.country"] = ["United-States", "Cambodia", "England", "Puerto-Rico", "Canada", "Germany", "Outlying-US(Guam-USVI-etc)", "India", "Japan", "Greece", "South", "China", "Cuba", "Iran", "Honduras", "Philippines", "Italy", "Poland", "Jamaica", "Vietnam", "Mexico", "Portugal", "Ireland", "France", "Dominican-Republic", "Laos", "Ecuador", "Taiwan", "Haiti", "Columbia", "Hungary", "Guatemala", "Nicaragua", "Scotland", "Thailand", "Yugoslavia", "El-Salvador", "Trinadad&Tobago", "Peru", "Hong", "Holand-Netherlands"]
label_values = [0, 1]

# GET SMALLER SAMPLE OF TRAINING DATA
df = df.sample(n=1200)
df = df.reset_index(drop=True)

# TRAINING DATA: SECOND PREPROCESSING STEP FOR SVM
print("Preprocessing training data...")
if os.path.exists("preprocessed_svm_training_data.csv"):
    print("TRAIN FILE EXISTS")
    df = p.read_csv("preprocessed_svm_training_data.csv")
else:
    preprocess.preprocess_training_data_SVM(df, numerical_attributes, categorical_attributes, attribute_values, label_name)
    df.to_csv("preprocessed_svm_training_data.csv", index=False)


# GET SMALLER SAMPLE OF TRAINING DATA
# df = df.sample(n=10)

# TEST DATA: SECOND PREPROCESSING STEP FOR SVM
print("Preprocessing testing data...")
# df_test = df_test.sample(n=10)
# df_test = df_test.reset_index(drop=True)
if os.path.exists("preprocessed_svm_test_data.csv"):
    print("TEST FILE EXISTS")
    df_test = p.read_csv("preprocessed_svm_test_data.csv")
else:
    preprocess.preprocess_test_data_SVM(df_test, numerical_attributes, categorical_attributes, attribute_values)
    df_test.to_csv("preprocessed_svm_test_data.csv", index=False)

# DEFINE HYPERPARAMETERS
C = 500.0 / 872.0

# TRAIN
print("Training...")
learned_weights = svm_dual.train(df=df, C=C)

# RECORD PREDICTIONS
print("Recording predictions...")
predictions = svm_dual.make_predictions(df_test, learned_weights)

ID_s = [1] * len(df_test)
for i in range(len(df_test)):
    ID_s[i] = i + 1
submission = p.DataFrame({'Id': ID_s, 'Prediction': predictions})
submission.to_csv("submission.csv", index=False)




