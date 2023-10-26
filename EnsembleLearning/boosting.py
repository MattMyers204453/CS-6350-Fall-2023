import math
# from DecisionTree.ID3 import Node, find_highest_IG
import ID3 as id3


# MODIFIED ID3 ----------

# IMPURITY = "me"
IMPURITY = "gini"
# IMPURITY = "entropy"

# Get WEIGHTED probability/proportion of label value
def getp(df, label_value, weights):
    num_examples = len(df.index) 
    if (num_examples == 0):
        return 0
    # The weight sum should equal 1
    weight_sum = 0
    for i in range(num_examples):
        row = df.iloc[i]
        weight_sum += weights[row.name]
    p = 0.0
    for i in range(num_examples):
        row = df.iloc[i]
        actual_label = row.get("label")
        if (actual_label == label_value):
            p += weights[row.name]
    return p / weight_sum

def get_impurity_of_dataset(df, labels, weights):
    if IMPURITY == "gini":
        sum = 0.0
        for label in labels:
            prob_i = getp(df, label, weights)
            sum += (prob_i * prob_i)
        gini = (1 - sum)
        return gini

def get_impurity_of_feature_at_specific_value(df, attribute_name, value, labels, weights):
    subset = df.loc[(df[attribute_name] == value)]
    if len(subset.index) == 0:
        return 0
    return get_impurity_of_dataset(subset, labels, weights)

def get_IG(impurity_of_set, df, attribute_name, attribute_values, labels, weights):
    impurity_for_each_value = {}
    for value in attribute_values[attribute_name]:
        impurity = get_impurity_of_feature_at_specific_value(df, attribute_name, value, labels, weights)
        impurity_for_each_value[value] = impurity
    num_examples = len(df.index)
    sigma = 0.0
    for value, impurity_for_this_value in impurity_for_each_value.items():
        weighted_sum = get_weighted_sum_at_feature(df, attribute_name, value, weights)
        #####num_features_at_this_value = sum(df[attribute_name] == value)
        # weight times impurity
        #####term = (weighted_num_features_at_this_value / float(num_examples)) * impurity_for_this_value
        term = weighted_sum * impurity_for_this_value
        sigma += term
    return impurity_of_set - sigma

def find_highest_IG(df, attribute_names, labels, attribute_values, weights):
    impurity_of_set = get_impurity_of_dataset(df, labels, weights)
    IG_for_each_value = {}
    # Get the information gain for each feature
    for i in range(len(attribute_names)):
        IG = get_IG(impurity_of_set, df, attribute_names[i], attribute_values, labels, weights)
        IG_for_each_value[attribute_names[i]] = IG
    best_feature = max(IG_for_each_value, key=IG_for_each_value.get)
    return best_feature


# BOOSTING ----------

def get_weighted_sum_at_feature(df, feature, feature_value, weights):
    num_rows = len(df.index)
    weighted_sum = 0.0
    for i in range(num_rows):
        row = df.iloc[i]
        actual_label = row.get(feature)
        if (actual_label == feature_value):
            weighted_sum += weights[row.name]
    return weighted_sum

def test_then_get_alpha_and_agreement_vector(stump, df, weights, num_test_examples, attribute_values):
    agreement_vector = [1] * num_test_examples
    error = 0.0
    for i in range(len(df.index)):
        row = df.iloc[i]
        actual_label = row.get("label")
        result_label = id3.predict(row, stump, attribute_values)
        if (actual_label != result_label):
            error += weights[i]
            agreement_vector[i] = -1
    if error == 0:
        error = 0.00001
    alpha = (1 / 2) * math.log((1 - error) / error)
    return (alpha, agreement_vector)

def update_weights(old_weights, alpha, agreement_vector):
    new_weights_unormalized = [0] * len(old_weights)
    for i in range(len(new_weights_unormalized)):
        new_weights_unormalized[i] = old_weights[i] * math.exp(-1 * alpha * agreement_vector[i])
    normalization_constant = sum(new_weights_unormalized)
    new_weights = [0] * len(old_weights)
    for i in range(len(new_weights)):
        new_weights[i] = new_weights_unormalized[i] / normalization_constant
    return new_weights

def weighted_mode(df, weights):
    num_examples = len(df.index)
    pos_count = 0.0
    neg_count = 0.0
    for i in range(num_examples):
        row = df.iloc[i]
        actual_label = row.get("label")
        if (actual_label == 1):
            pos_count += weights[row.name]
        else:
            neg_count += weights[row.name]
    if (pos_count > neg_count):
        return 1
    return -1

def get_stump(df, attribute_names, attribute_values, label_values, weights):
    feature_with_highest_IG = find_highest_IG(df, attribute_names, label_values, attribute_values, weights)
    root = id3.Node(isLeafNode=False, feature=feature_with_highest_IG, label=None)
    i = 0
    for value in attribute_values[feature_with_highest_IG]:
        subset = df.loc[(df[feature_with_highest_IG] == value)]
        weighted_prediction = None
        # If subset is empty, find weighted mode in original set
        if (len(subset.index) == 0):
            weighted_prediction = weighted_mode(df, weights)
        # Otherwise, find weighted mode in subset
        else:
            weighted_prediction = weighted_mode(subset, weights)
        leaf = id3.Node(isLeafNode=True, feature=None, label=weighted_prediction)
        root.children[i] = leaf
        i += 1
    return root

def adaboost(df, t, attribute_names, attribute_values, label_values):
    alphas = [0] * t
    weak_classifiers = [None] * t
    num_examples = len(df.index)
    weights = [1 / num_examples] * num_examples 
    for i in range(t):
        # Train weak classifier 
        stump = get_stump(df, attribute_names, attribute_values, label_values, weights)

        # Cache weak classifier 
        weak_classifiers[i] = stump

        # Get vote and "agreement vector"
        alpha_and_agreement_vector = test_then_get_alpha_and_agreement_vector(stump, df, weights, num_examples, attribute_values)
        alpha = alpha_and_agreement_vector[0]
        agreement_vector = alpha_and_agreement_vector[1]

        # Record alpha value (vote) for this round
        alphas[i] = alpha

        # Update weights
        weights = update_weights(weights, alpha, agreement_vector)
    return (alphas, weak_classifiers)

def predict(row, trained_adaboost_alphas_classifiers, attribute_values):
    alpha_values = trained_adaboost_alphas_classifiers[0]
    weak_classifiers = trained_adaboost_alphas_classifiers[1]
    t = len(alpha_values)
    final_sum = 0.0
    for i in range(t):
        h_x = id3.predict(row, weak_classifiers[i], attribute_values)
        a = alpha_values[i]
        final_sum += (a * h_x)
    if (final_sum > 0):
        return 1
    return -1