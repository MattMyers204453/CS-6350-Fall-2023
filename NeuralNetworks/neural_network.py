import numpy as np
import math

# CONSTANTS
TOP_SIGMOID_THRESHOLD = 23
BOTTOM_SIGMOID_THRESHOLD = -23
LAYER_ONE = 1
LAYER_TWO = 2
LAYER_THREE = 3
# GLOBAL VARIABLES
num_of_zs_layer_1 = 50
num_of_zs_layer_2 = 50
num_of_xs_layer_0 = 4 + 1
weights_map = {}
z_map = {}

# Get feature vector (with folded bias) at i
# drop label, add constant value of 1 (for bias), and convert to numpy array for matrix operations.
def x_at_i(i, df):
    return np.array(df.iloc[i][:-1].tolist() + [1])

def sgn(v):
    return 1 if v > 0.0 else -1

# weight identifier
def w(start, to, layer):
    return f"w_{start}{to}^{layer}"

# neuron identifier
def z(num, layer):
    return f"z_{num}^{layer}"

def sigmoid(x):
    if (x > TOP_SIGMOID_THRESHOLD ):
        return 1
    if (x < BOTTOM_SIGMOID_THRESHOLD):
        return 0
    return 1.0 / (1.0 + math.exp(-1 * x))

def sigmoid_deriv(s):
    return sigmoid(s) * (1 - sigmoid(s))

def get_g_layer1_z(z_val, x_i):
    g = 0
    for i in range(num_of_xs_layer_0):
        g += (weights_map[w(i, z_val, LAYER_ONE)] * x_i[i])
    return g

def get_g_layer2_z(z_val):
    g = 0
    for i in range(num_of_zs_layer_1):
        g += (weights_map[w(i, z_val, LAYER_TWO)] * z_map[z(i, LAYER_ONE)])
    return g

def loss_deriv(y_i, y_output):
    return y_i - y_output

# Use chain rule directly
def get_layer_1_gradient(y_i, y_ouput, x_i):
    L_deriv = loss_deriv(y_i, y_ouput)
    gradient_weights_layer_1 = {}
    for i in range(1, num_of_zs_layer_1):
        path_sum = 0
        for j in range(1, num_of_zs_layer_2):
            g = get_g_layer2_z(j)
            sig_deriv = sigmoid_deriv(g)
            path_sum += (L_deriv * weights_map[w(j, LAYER_ONE, LAYER_THREE)] * sig_deriv * weights_map[w(i, j, LAYER_TWO)])
        rightmost_term = get_g_layer1_z(i, x_i)
        middle_term = sigmoid_deriv(rightmost_term)
        for k in range(len(x_i)):
            gradient_weights_layer_1[w(k, i, LAYER_ONE)] = path_sum * middle_term * x_i[0]
    return gradient_weights_layer_1

# Use chain rule directly
def get_layer_2_gradient(y_i, y_ouput):
    L_deriv = loss_deriv(y_i, y_ouput)
    gradient_weights_layer_2 = {}
    for i in range(1, num_of_zs_layer_2):
        g = get_g_layer2_z(i)
        sig_partial = sigmoid_deriv(g)
        for j in range(num_of_zs_layer_1):
            gradient_weights_layer_2[w(j, i, LAYER_TWO)] = L_deriv * weights_map[w(i, LAYER_ONE, LAYER_THREE)] * sig_partial * z_map[z(j, LAYER_ONE)]
    return gradient_weights_layer_2

# Use chain rule directly
def get_layer_3_gradient(y_i, y_output):
    L_deriv = loss_deriv(y_i, y_output)
    gradient_weights_layer_3 = {}
    for i in range(num_of_zs_layer_2):
        gradient_weights_layer_3[w(i, LAYER_ONE, LAYER_THREE)] =  L_deriv * z_map[z(i, 2)]
    return gradient_weights_layer_3

def forward_pass(x_input):
    z_map[z(0, LAYER_ONE)] = 1
    for i in range(1, num_of_zs_layer_1):
        g = get_g_layer1_z(i, x_input)
        sig = sigmoid(g)
        z_map[z(i, LAYER_ONE)] = sig
    z_map[z(0, LAYER_TWO)] = 1
    for i in range(1, num_of_zs_layer_2):
        g= get_g_layer2_z(i)
        sig = sigmoid(g)
        z_map[z(i, LAYER_TWO)] = sig
    y_output = 0
    for i in range(num_of_zs_layer_2):
        y_output += (weights_map[w(i, LAYER_ONE, LAYER_THREE)] * z_map[z(i, LAYER_TWO)])
    return y_output

def back_propagation(y_ouput, y_i, x_input):
    w_layer_3_partials = get_layer_3_gradient(y_ouput, y_i)
    w_layer_2_partials = get_layer_2_gradient(y_ouput, y_i)
    w_layer_1_partials = get_layer_1_gradient(y_ouput, y_i, x_input)
    return (w_layer_1_partials, w_layer_2_partials, w_layer_3_partials)

def init_weights():
    # layer 1
    for n in range(1, num_of_zs_layer_1):
        for m in range(num_of_xs_layer_0):
            weights_map[w(m, n, LAYER_ONE)] = np.random.normal(0, 1)
    # layer 2
    for n in range(1, num_of_zs_layer_2):
        for m in range(num_of_zs_layer_1):
            weights_map[w(m, n, LAYER_TWO)] = np.random.normal(0, 1)
    # layer 3
    for i in range(num_of_zs_layer_2):
        weights_map[w(i, LAYER_ONE, LAYER_THREE)] = np.random.normal(0, 1)

def update_weights(r, w_partials_gradient_tuple):
    w_layer_3_gradient = w_partials_gradient_tuple[2]
    w_layer_2_gradient = w_partials_gradient_tuple[1]
    w_layer_1_gradient = w_partials_gradient_tuple[0]
    for i in range(num_of_zs_layer_2):
        weights_map[w(i, LAYER_ONE, LAYER_THREE)] -= r * w_layer_3_gradient[w(i, LAYER_ONE, LAYER_THREE)]
    for n in range(1, num_of_zs_layer_2):
        for m in range(num_of_zs_layer_1):
            weights_map[w(m, n, LAYER_TWO)] -= r * w_layer_2_gradient[w(m, n, LAYER_TWO)]
    for n in range(1, num_of_zs_layer_1):
        for m in range(num_of_xs_layer_0):
            weights_map[w(m, n, LAYER_ONE)] -= r * w_layer_1_gradient[w(m, n, LAYER_ONE)]


def train(df, attribute_names, label_name, T, r, d, WIDTH):
    print(f"Training neural network...")
    global num_of_zs_layer_1
    num_of_zs_layer_1 = WIDTH
    global num_of_zs_layer_2 
    num_of_zs_layer_2 = WIDTH
    global num_of_xs_layer_0
    num_of_xs_layer_0 = len(attribute_names) + 1
    init_weights()
    T = 3
    r = 0.1
    d= 0.3
    for t in range(T):
        r = r / float(1 + (r * t) / float(d))
        df = df.sample(frac=1)
        for i in range(len(df.index)):
            x_i = x_at_i(i, df)
            x_input = np.insert(x_i, 0, 1)[:-1]
            y_i = df.iloc[i].get(label_name)
            prediction = sgn(forward_pass(x_input))
            if y_i != prediction:
                loss_gradient = back_propagation(y_ouput=prediction, y_i=y_i, x_input=x_input)
                update_weights(r, loss_gradient)
        

def test_and_print_results(df, test_df, label_name):        
    print("Testing model on testing data...")
    errors = 0
    for i in range(len(test_df.index)):
        feature_vector_no_folded_bias = x_at_i(i, test_df)
        feature_vector = np.insert(feature_vector_no_folded_bias, 0, 1)[:-1]
        ground_truth_label = test_df.iloc[i].get(label_name)
        prediction = sgn(forward_pass(feature_vector)) 
        if ground_truth_label != prediction:
            errors += 1
    print(f"Total examples misclassified: {errors}")
    print(f"Test Accuracy: {(((float(len(test_df.index)) - errors) / float(len(test_df.index))) * 100)}%")

    print("Testing model on training data...")
    errors = 0
    for i in range(len(df.index)):
        feature_vector_no_folded_bias = x_at_i(i, df)
        feature_vector = np.insert(feature_vector_no_folded_bias, 0, 1)[:-1]
        ground_truth_label = df.iloc[i].get(label_name)
        prediction = sgn(forward_pass(feature_vector)) 
        if ground_truth_label != prediction:
            errors += 1
    print(f"Total examples misclassified: {errors}")
    print(f"Train Accuracy: {(((float(len(df.index)) - errors) / float(len(df.index))) * 100)}%")