import os
import sys
import pandas as p

def read_training_data(file_name, attribute_names, label_name):
    file_path = os.path.join(sys.path[0], file_name)
    df = p.read_csv(file_path, header=None, na_values="?")
    df.columns = attribute_names + [label_name]
    # Delete Column name row
    df = df.drop(df.index[0])
    return df

def preprocess_training_data(df, attribute_names, numerical_attributes):
    df = df.dropna()
    for na in numerical_attributes:
        df[na] = df[na].astype(int).gt(df[na].median()).astype(int)
    return df

def read_test_data(file_name, attribute_names, label_name):
    file_path = os.path.join(sys.path[0], file_name)
    df = p.read_csv(file_path, header=None)
    # Drop ID column
    df = df.drop(df.columns[0], axis=1)
    df.columns = attribute_names
    # Delete Column name row
    df = df.drop(df.index[0])
    return df

def preprocess_test_data(df, attribute_names, numerical_attributes):
    for name in attribute_names:
        df.loc[df[name] == '?', name] = df[name].mode()[0]
    for na in numerical_attributes:
        df[na] = df[na].astype(int).gt(df[na].median()).astype(int)
    return df

def preprocess_training_data_SVM(df, numerical_attributes, categorical_attributes, attribute_values, label_name):
    # for a in attributes:
    #     df.loc[df[a] == '?', a] = df[a].mode()[0]
    for na in numerical_attributes:
        standardize_col(df, na)
    for ca in categorical_attributes:
        one_hot_encode(df, ca, attribute_values)
    map = {1: 1, 0: -1}
    df[label_name] = df[label_name].astype("int").map(map)

def preprocess_test_data_SVM(df, numerical_attributes, categorical_attributes, attribute_values):
    # for a in attributes:
    #     df.loc[df[a] == '?', a] = df[a].mode()[0]
    for na in numerical_attributes:
        standardize_col(df, na)
    for ca in categorical_attributes:
        one_hot_encode(df, ca, attribute_values)


def standardize_col(df, col_name):
    df[col_name] = df[col_name].astype(float)
    df[col_name] = (df[col_name] - df[col_name].mean()) / df[col_name].std()

def one_hot_encode(df, col_name, attribute_values):
    new_categories = attribute_values[col_name]
    for new_category in new_categories:
        df[new_category] = [0] * len(df.index)
    for i in range(1, len(df.index)):
        value_one = df.at[i, col_name]
        for new_category in new_categories:
            df.at[i, new_category] = 1 if value_one == new_category else -1
    df.drop(col_name, inplace=True, axis=1)
