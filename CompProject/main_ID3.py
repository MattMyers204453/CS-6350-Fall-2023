# READ DATA INTO DATAFRAME
import sys
import read_data as preprocess
import ID3 as id3
import pandas as p

# READ TRAINING DATA
print("Reading training data...")
attribute_names = ["age","workclass","fnlwgt","education","education.num","marital.status","occupation", "relationship","race", "sex","capital.gain","capital.loss","hours.per.week","native.country"]
label_name = "income>50K"
df = preprocess.read_training_data("train_final.csv", attribute_names, label_name)
# Delete Column name row
df = df.drop(df.index[0])
# Rename label column
label_name = "label"
df = df.rename(columns={df.columns[-1]: label_name})


# PREPROCESS TRAINING DATA
print("Preprocessing training data...")
categorical_attributes = ["workclass","education","marital.status","occupation", "relationship","race", "sex","native.country"]
numerical_attributes = ["age","fnlwgt","education.num","capital.gain","capital.loss","hours.per.week"]
df = preprocess.preprocess_training_data(df=df, attribute_names=attribute_names, numerical_attributes=numerical_attributes)

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

# Take smaller sample of dataset
df = df.sample(n=1000)

# TRAIN
print("Training...")
ID3_model = id3.ID3(df, attribute_names=attribute_names, attribute_values=attribute_values, label_values=label_values)

# READ TEST DATA
print("READING TEST DATA")
df_test = preprocess.read_training_data("test_final.csv", attribute_names, label_name)
# Delete Column name row
df_test = df_test.drop(df.index[0])
# Rename label column
df_test = df_test.rename(columns={df_test.columns[-1]: 'label'})

# PREPROCESS TEST DATA
df_test = preprocess.preprocess_training_data(df=df_test, attribute_names=attribute_names, numerical_attributes=numerical_attributes)

# RUN TEST
print("Testing...")
predictions = [0] * len(df_test)
error_count = 0
for i in range(len(df_test.index)):
    row = df_test.iloc[i]
    prediction = id3.predict(row=row, root=ID3_model, attribute_values=attribute_values)
    predictions[i] = prediction

#test = p.read_csv(os.path.join(sys.path[0], 'test_final.csv'))
ID_s = [1] * len(df_test)
for i in range(len(df_test)):
    ID_s[i] = i + 1
submission = p.DataFrame({'Id': ID_s, 'Prediction': predictions})
submission.to_csv("submission.csv", index=False)
