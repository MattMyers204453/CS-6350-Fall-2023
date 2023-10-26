import math


# IMPURITY = "me"
IMPURITY = "gini"
# IMPURITY = "entropy"

# Get probability/proportion of label value
def getp(df, label_value):
    total = len(df.index) 
    if (total == 0):
        return 0
    numerator = sum(df["label"] == label_value)
    return (numerator / float(total))

def get_impurity_of_dataset(df, labels):
    if IMPURITY == "gini":
        sum = 0.0
        for label in labels:
            prob_i = getp(df, label)
            sum += (prob_i * prob_i)
        gini = (1 - sum)
        return gini
    if IMPURITY == "me":
        label_counts_array = df["label"].value_counts()
        majority_feature_count = label_counts_array.max()
        me = 1 - (majority_feature_count / len(df))
        return me
    if IMPURITY == "entropy":
        label_counts_array = df["label"].value_counts()
        total_size = len(df)
        entropy = 0
        for count in label_counts_array:
            p = count / total_size
            entropy -= p * math.log2(p)
        return entropy

def get_impurity_of_feature_at_specific_value(df, attribute_name, value, labels):
    subset = df.loc[(df[attribute_name] == value)]
    if len(subset.index) == 0:
        return 0
    return get_impurity_of_dataset(subset, labels)

def get_IG(impurity_of_set, df, attribute_name, attribute_values, labels):
    impurity_for_each_value = {}
    for value in attribute_values[attribute_name]:
        impurity = get_impurity_of_feature_at_specific_value(df, attribute_name, value, labels)
        impurity_for_each_value[value] = impurity
    length_of_whole_set = len(df.index)
    sigma = 0.0
    for value, impurity_for_this_value in impurity_for_each_value.items():
        num_features_at_this_value = sum(df[attribute_name] == value)
        # weight times impurity
        term = (num_features_at_this_value / float(length_of_whole_set)) * impurity_for_this_value
        sigma += term
    return impurity_of_set - sigma

def find_highest_IG(df, attribute_names, labels, attribute_values):
    impurity_of_set = get_impurity_of_dataset(df, labels)
    IG_for_each_value = {}
    # Get the information gain for each feature
    for i in range(len(attribute_names)):
        IG = get_IG(impurity_of_set, df, attribute_names[i], attribute_values, labels)
        IG_for_each_value[attribute_names[i]] = IG
    best_feature = max(IG_for_each_value, key=IG_for_each_value.get)
    return best_feature

class Node:
    def __init__(self):
        pass
    def __init__(self, isLeafNode, feature, label):
        self.isLeafNode = isLeafNode
        self.feature = feature
        self.label = label
        self.children = {}

def ID3(df, attribute_names, attribute_values, label_values):
    # If all labels are the same, return leaf node with unique label
    if df["label"].nunique() == 1:
        return Node(isLeafNode=True, feature=None, label=df["label"].unique()[0])

    # Find feature with highest IG
    feature_with_highest_IG = find_highest_IG(df, attribute_names, label_values, attribute_values)

    # We now create a feature node. (We "split" on this feature)
    root = Node(isLeafNode=False, feature=feature_with_highest_IG, label=None)

    # For each possible value of the attribute we split on...
    i = 0
    for value in attribute_values[feature_with_highest_IG]:
        subset = df.loc[(df[feature_with_highest_IG] == value)]
        # If all examples in subset have same label, create leaf node
        if subset["label"].nunique() == 1:
            leaf = Node(isLeafNode=True, feature=None, label=df["label"].unique()[0])
            root.children[i] = leaf
        # If the subset doesn't contain anything, create leaf node with the most common label of the original array
        if (len(subset) == 0):
            most_common_label_in_original_df = df["label"].mode()[0]
            leaf = Node(isLeafNode=True, feature=None, label=most_common_label_in_original_df)
            root.children[i] = leaf
        # Otherwise, crea
        else:
            subAttributes = attribute_names[:]
            subAttributes.remove(feature_with_highest_IG)
            subAttribute_values = attribute_values.copy()
            del subAttribute_values[feature_with_highest_IG]
            # Recursively call ID3 on subset
            subtree = ID3(subset, subAttributes, subAttribute_values, label_values)
            # We are implicitly tracking which child is connected to which feature value edge. 
            # When we traverse the tree, we iterate through children in the same order
            root.children[i] = subtree
        i += 1
    return root

def predict(row, root, attribute_values):
    while root.isLeafNode == False:
        feature = root.feature
        item_value = row.get(feature)
        index = attribute_values[feature].index(item_value)
        root = root.children[index]
    return root.label
