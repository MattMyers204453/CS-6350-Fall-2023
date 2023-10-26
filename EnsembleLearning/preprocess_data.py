import os
import sys
import pandas as p

def read_data(file_name, attribute_names, label_name):
    file_path = os.path.join(sys.path[0], file_name)
    df = p.read_csv(file_path, header=None)
    # Read only five rows
    # df = p.read_csv(file_path, header=None, nrows=5)
    df.columns = attribute_names + [label_name]
    return df

def convert_numerical_to_binary(df, numerical_feature_names):
    for name in numerical_feature_names:
        df[name] = df[name].astype('int')
        median = df[name].median();
        df[name] = df[name].gt(median).astype('int')
    return df

def convert_binary_to_integer_binary(df, binary_feature_names):
    binary = {'no': 0, 'yes': 1}
    # for name in binary_feature_names:
    #     df[name] = df[name].map(binary).astype(int)
    df["default"] = df["default"].map(binary).astype(int)
    df["loan"] = df["loan"].map(binary).astype(int)
    df["housing"] = df["housing"].map(binary).astype(int)
    return df

def convert_label_to_binary(df, label_name):
    label_binary = {'no': -1, 'yes': 1}
    df[label_name] = df[label_name].map(label_binary).astype(int)
    return df

def preprocess(df, numerical_feature_names, binary_feature_names, label_name):
    df = convert_numerical_to_binary(df, numerical_feature_names)
    df = convert_binary_to_integer_binary(df, binary_feature_names)
    df = convert_label_to_binary(df, label_name)
    return df
