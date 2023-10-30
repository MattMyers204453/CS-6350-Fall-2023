import os
import sys
import pandas as p

def read_data(file_name, attribute_names, label_name):
    file_path = os.path.join(sys.path[0], file_name)
    df = p.read_csv(file_path, header=None)
    # df = p.read_csv(file_path, header=None, nrows=5)
    df.columns = attribute_names + [label_name]
    return df

def preprocess_data(df, attribute_names, numerical_attributes):
    for name in attribute_names:
        df.loc[df[name] == '?', name] = df[name].mode()[0]
    for ca in numerical_attributes:
        df[ca] = df[ca].astype(int).gt(df[ca].median()).astype(int)
    return df
