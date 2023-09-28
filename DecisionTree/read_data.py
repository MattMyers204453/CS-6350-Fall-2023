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
