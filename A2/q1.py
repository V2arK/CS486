import numpy as np
from math import log2
from queue import PriorityQueue
import matplotlib.pyplot as plt

##### Variables ######
NUM_FEATURES = 0 # Define later
NUM_SAMPLE = 1500
ATHEISM_ID = 1
BOOKS_ID = 2
LABEL_STR = {ATHEISM_ID: "Atheism", BOOKS_ID: "Books" }

# We define P(x) as # belongs to atheism over total.

##### READING FILE ######

# Function to read label data, returning a dictionary mapping document ID to label
def read_label_data(file_name):
    label_data = {}
    with open(file_name, 'r') as file:
        lineNum = 1
        for line in file:
            label_data[lineNum] = int(line.strip())
            lineNum += 1
            # docId = line number
    return label_data

# Function to read word data, returning a dictionary mapping word ID to word
def read_word_data(file_name):
    word_data = {}
    with open(file_name, 'r') as file:
        lineNum = 1
        for line in file:
            word_data[int(lineNum)] = line.strip()
            lineNum += 1
            # wordId = line number
    # Define the Number of features
    global NUM_FEATURES
    NUM_FEATURES = lineNum
    return word_data

# Function to read train or test data
# document_data[n] = arrray of word_id that n+1'th doc have.
def read_document_data(file_name):
    document_data = {i: [] for i in range(1, NUM_SAMPLE + 1)} # 1 to 1500 
    with open(file_name, 'r') as file:
        for line in file:
            doc_id, word_id = line.strip().split()
            document_data[int(doc_id)].append(int(word_id))
    return document_data

##### COMPUTING #####

# functions that does log_2(x/x+y).
# the # of bits to encode x.
def entropy(x, y):
    #print(f"x = {x}, y = {y}")
    
    # corner case, for our purpose log_2(0) = 0.
    if x == 0:
        return 0
    
    val = x / (x + y)
    #print(f"val = {val}")
    #print(f"log2(val) = {log2(val)}")
    return - log2(val)

def information_content(dataset, doc_subreddit_dict):
    # calculate the information content of the dataset
    # dataset needs to be a dictionary
    
    atheism_count = 0
    books_count = 0
    
    doc_ids = [key for key, _ in dataset.items()]
    
    # Loop over all elements and calculate # belongs to atheism or books respectively
    for doc_id in doc_ids:
        if doc_subreddit_dict[doc_id] == ATHEISM_ID:
            atheism_count += 1
        elif doc_subreddit_dict[doc_id] == BOOKS_ID:
            books_count += 1
            
            
    etp1 = entropy(atheism_count, books_count)
    etp2 = entropy(books_count, atheism_count)
    
    # Corner case
    if (atheism_count + books_count) == 0:
        return 0
        
    return ((atheism_count) / (atheism_count + books_count) * etp1) + ((books_count) / (atheism_count + books_count) * etp2)


'''
def information_gain(num1, num2, method):
    etp1 = entropy(num1, num2)
    etp2 = entropy(num2, num1)
    if method == 1:
        # Method 1: Average information gain across the leaves
        return (etp1 / 2) + (etp2 / 2)
    elif method == 2:
        # Method 2: The one discussed in Class
        
        # Corner case
        if (num1 + num2) == 0:
            return 0
        
        return ((num1) / (num1 + num2) * etp1) + ((num2) / (num1 + num2) * etp2)
'''

def information_gain(num1, num2):
    etp1 = entropy(num1, num2)
    etp2 = entropy(num2, num1)
    
    # Corner case
    if (num1 + num2) == 0:
        return 0
        
    return ((num1) / (num1 + num2) * etp1) + ((num2) / (num1 + num2) * etp2)


# Function to calculate the delta information gain for a given split
def delta_information_gain(elements, doc_subreddit_dict, word_to_split, method, debug=False):
    
    atheism_count_E = 0
    books_count_E = 0
    
    doc_ids = [key for key, _ in elements.items()]
    
    # Loop over all elements and calculate # belongs to atheism or books respectively
    for doc_id in doc_ids:
        if doc_subreddit_dict[doc_id] == ATHEISM_ID:
            atheism_count_E += 1
        elif doc_subreddit_dict[doc_id] == BOOKS_ID:
            books_count_E += 1
            
    # TODO: not sure if for IE we going to use different methods.
    IE = information_gain(atheism_count_E, books_count_E)
    if debug:
        print(f"IE = {IE}, atheism_count_E = {atheism_count_E}, books_count_E = {books_count_E}")
    # We now proceed to split the elements by the word_to_split.
    
    # E1
    has_word_to_split = [key for key, value in elements.items() 
                         if word_to_split in value]
    # E2
    not_have_word_to_split = [key for key, value in elements.items() 
                              if word_to_split not in value]
    
    #if debug:
    #    print(f"len has_word_to_split = {len(has_word_to_split)}, len not_have_word_to_split = {len(not_have_word_to_split)}")
    
    # note the values in the arrays are doc_id has or has not the word to split.
    atheism_count_E1 = 0
    books_count_E1 = 0
    
    # Loop over all elements and calculate # belongs to atheism or books respectively
    for doc_id in has_word_to_split:
        if doc_subreddit_dict[doc_id] == ATHEISM_ID:
            atheism_count_E1 += 1
        elif doc_subreddit_dict[doc_id] == BOOKS_ID:
            books_count_E1 += 1
            
    # TODO: not sure if for IE we going to use different methods.
    IE1 = information_gain(atheism_count_E1, books_count_E1)
    if debug:
        print(f"IE1 = {IE1}, atheism_count_E1 = {atheism_count_E1}, books_count_E1 = {books_count_E1}")
    
    atheism_count_E2 = 0
    books_count_E2 = 0
    
    # Loop over all elements and calculate # belongs to atheism or books respectively
    for doc_id in not_have_word_to_split:
        if doc_subreddit_dict[doc_id] == ATHEISM_ID:
            atheism_count_E2 += 1
        elif doc_subreddit_dict[doc_id] == BOOKS_ID:
            books_count_E2 += 1
    
    # TODO: not sure if for IE we going to use different methods.
    IE2 = information_gain(atheism_count_E2, books_count_E2)
    if debug:
        print(f"IE2 = {IE2}, atheism_count_E2 = {atheism_count_E2}, books_count_E2 = {books_count_E2}")
    
    # Finally the calculation
    if method == 1:
        # Method 1: Average information gain across the leaves
        return IE - ((IE1 / 2) + (IE2 / 2))
    elif method == 2:
        # Method 2: The one discussed in Class
        
        sum_E1 = atheism_count_E1 + books_count_E1
        sum_E2 = atheism_count_E2 + books_count_E2
    
        # Corner case
        if sum_E1 == 0 and sum_E2 == 0:
            return IE
        
        return IE - ((sum_E1 / (sum_E1 + sum_E2)) * IE1) - ((sum_E2 / (sum_E1 + sum_E2)) * IE2)


##### Decision Tree #####

class Node:
    def __init__(self, dataset, point_estimate, feature_to_split=None, info_gain=0, splitted_feature=[]):
        self.dataset = dataset  # The subset 'E' of the dataset at this node
        self.feature_to_split = feature_to_split  # 'X_prime': The feature to split on at this node
        self.point_estimate = point_estimate # Point estimation of current node.
        self.splitted_feature = splitted_feature[:] # array of already splitted feature
        self.info_gain = info_gain  # 'delta_I': Information gain of the split
        self.left = None      # Left child (with feature)
        self.right = None     # Right child (without feature)

    # This is needed for the PriorityQueue to compare Nodes based on information gain
    def __lt__(self, other):
         # since priority queue is a min_heap, so we define lt as gt
        return self.info_gain > other.info_gain
    
    
def calculate_point_estimate(dataset, label_dict):
        # Calculate the dominant label (subreddit) in the dataset
        label_counts = {ATHEISM_ID: 0, BOOKS_ID: 0}
        for doc_id in dataset:
            label = label_dict[doc_id]
            label_counts[label] += 1
        # The point estimate is the label with the highest count
        point_estimate = max(label_counts, key=label_counts.get)
        return point_estimate
    
def find_best_to_split(dataset, label_dict, method, words=None, splitted_feature=[]):
    # Find the best feature to split next
    # by try to compute the delta_information_gain on all features 
    # appeared in the dataset but not the ones in splitted_feature, and return a tuple containing
    # the feature to split that will give the biggest delta_information_gain
    # and the delta_information_gain.
    best_feature = None
    best_info_gain = 0  # Start with 0 to ensure any gain is better, but not no split

    # Iterate through each feature in the dataset to find the best one to split on
    for feature in range(1, NUM_FEATURES + 1):
        
        # skips the feature we already split on
        if feature in splitted_feature:
            continue
        
        # Compute the information gain for splitting on the current feature
        current_info_gain = delta_information_gain(dataset, label_dict, feature, method)
        
        # If the information gain of the current feature is better than the best one so far, update the best feature and gain
        if current_info_gain > best_info_gain:
            #print(f"better word = {words[feature]}, info gain = {current_info_gain} ")
            #print(f"delta I = {delta_information_gain(dataset, label_dict, feature, method, True)}")
            #print("------------")
            best_feature = feature
            best_info_gain = current_info_gain

    '''
    if words is not None and best_feature is not None:
        print(f"spliting feature = {words[best_feature]}, info gain = {best_info_gain}")
    elif words is not None:
        print(f"spliting feature = None, info gain = {best_info_gain}")
    '''
    
    print(f"delta I = {delta_information_gain(dataset, label_dict, best_feature, method, True)}")
    return best_feature, best_info_gain

def split_dataset(dataset, feature_to_split):
    # Datasets to hold the split
    dataset_with_feature = {}
    dataset_without_feature = {}
    #print(f"Splitting on {feature_to_split}")
    # Iterate over each entry in the dataset
    for doc_id, word_ids in dataset.items():
        # Check if the feature to split on is in the document's word IDs
        if feature_to_split in word_ids:
            # Add this document to the dataset with the feature
            #print(f"with_feature: doc_id = {doc_id}, features = {word_ids}")
            dataset_with_feature[doc_id] = word_ids[:] # [:] to make a shallow copy
            
            #print(f"remove feature: {feature_to_split}")
            # remove the spliting feature
            #dataset_with_feature[doc_id].remove(feature_to_split)
        else:
            # Add this document to the dataset without the feature
            #print(f"without_feature: doc_id = {doc_id}, features = {word_ids}")
            dataset_without_feature[doc_id] = word_ids
    
    print(f"L size: {str(len(dataset_with_feature))}, R size:  {str(len(dataset_without_feature))}")
    return dataset_with_feature, dataset_without_feature

# Function to build the decision tree
def build_decision_tree(train_data, train_labels, method, subreddit_dict, max_nodes=100, words=None):
    MAX_NODES = max_nodes
    pq = PriorityQueue()
    root = Node(dataset=train_data, point_estimate=calculate_point_estimate(train_data, subreddit_dict))
    
    best_feature, best_info_gain = find_best_to_split(root.dataset, train_labels, method, words=words)
    
    # Update root node with split info
    root.feature_to_split = best_feature
    root.info_gain = best_info_gain
    
    pq.put((best_info_gain, root))

    while not pq.empty() and max_nodes > 0:
        
        info_gained, current_node = pq.get()
        
        # We only counts the number of internal nodes with max_nodes
        max_nodes -= 1
        print(f"\ndoing the {MAX_NODES - max_nodes}'th node")
        print(f"info gained = {info_gained}, split word = {words[current_node.feature_to_split]}")
        
        # Split the dataset based on the best feature
        current_node.splitted_feature.append(current_node.feature_to_split) # add splitted feature

        #print(f"splitted feature list is  {current_node.splitted_feature}")
        left_dataset, right_dataset = split_dataset(current_node.dataset, current_node.feature_to_split)
        
        best_feature_L, best_info_gain_L = find_best_to_split(left_dataset, train_labels, method, words=words, splitted_feature=current_node.splitted_feature)
        best_feature_R, best_info_gain_R = find_best_to_split(right_dataset, train_labels, method, words=words, splitted_feature=current_node.splitted_feature)
        
        
        left_node = Node(dataset=left_dataset, point_estimate=calculate_point_estimate(left_dataset, subreddit_dict), info_gain=best_info_gain_L, feature_to_split=best_feature_L, splitted_feature=current_node.splitted_feature)
        right_node = Node(dataset=right_dataset, point_estimate=calculate_point_estimate(right_dataset, subreddit_dict), info_gain=best_info_gain_R, feature_to_split=best_feature_R, splitted_feature=current_node.splitted_feature)
        
        # Update current node with split info
        current_node.left = left_node
        current_node.right = right_node
        
        # Add child nodes to the priority queue
        if best_feature_L is not None:
            pq.put((best_info_gain_L, left_node))
            #print(f"Adde L feature = {words[best_feature_L]}, info gain = {best_info_gain_L}")
        if best_feature_R is not None:
            pq.put((best_info_gain_R, right_node))
            #print(f"Added R feature = {words[best_feature_R]}, info gain = {best_info_gain_R}")
        
    
    return root

def print_tree(node, depth=0, feature_names=None):
    # Base case: if the node is a leaf, it will not have a child
    if node.left is None or node.right is None:
        print("-" * depth + "Leaf, estimate: " + LABEL_STR[node.point_estimate])
        return

    # Recursive case: print the current node's split information
    if feature_names and node.feature_to_split in feature_names:
        feature_name = feature_names[node.feature_to_split]
    else:
        feature_name = str(node.feature_to_split)

    print("-" * depth + f"Node: Split Feature = {feature_name}, Info Gain = {node.info_gain:.10f}")

    # Recursively print the left subtree
    print("-" * depth + "L (w/ feature):")
    print_tree(node.left, depth + 1, feature_names)

    # Recursively print the right subtree
    print("-" * depth + "R (wo/ feature):")
    print_tree(node.right, depth + 1, feature_names)
    
    
##### TESTING ######

# Use decision tree to predict the label for a single document
def predict_label(node, document_word_array):
    # If we have reached a leaf node, return its point estimate
    if node.left is None and node.right is None:
        return node.point_estimate
    # If the document contains the word_id at the current node, go left
    elif node.feature_to_split in document_word_array:
        return predict_label(node.left, document_word_array)
    # If the document does not contain the word_id, go right
    else:
        return predict_label(node.right, document_word_array)

# Function to calculate the accuracy of the decision tree
def calculate_accuracy(tree, data, labels):
    correct_predictions = 0
    # Iterate over all documents in the test data
    for doc_id, document_word_array in data.items():
        # Use the tree to predict the label for the current document
        predicted_label = predict_label(tree, document_word_array)
        # If the predicted label matches the actual label, increment the correct predictions count
        if predicted_label == labels[doc_id]:
            correct_predictions += 1
    # Calculate the percentage of correctly classified samples
    accuracy = (correct_predictions / len(data)) * 100
    return accuracy

##### MAIN ######

# Reading data from files
words = read_word_data('./words.txt')
train_data = read_document_data('./trainData.txt')
train_labels = read_label_data('./trainLabel.txt')
test_data = read_document_data('./testData.txt')
test_labels = read_label_data('./testLabel.txt')

##### b) #####

print("--- building tree 1 ---\n")
tree1 = build_decision_tree(train_data, train_labels, method=1, subreddit_dict=train_labels, words=words, max_nodes=10)
print("\n--- method 1 tree ---\n")
print_tree(tree1, feature_names=words)

print("\n--- building tree 2 ---\n")
tree2 = build_decision_tree(train_data, train_labels, method=2, subreddit_dict=train_labels, words=words, max_nodes=10)
print("\n--- method 2 tree ---\n")
print_tree(tree2, feature_names=words)

##### C) #####

# Validate the decision trees tree1 and tree2
accuracy_tree1 = calculate_accuracy(tree1, test_data, test_labels)
accuracy_tree2 = calculate_accuracy(tree2, test_data, test_labels)

# Print the accuracies
print(f"\nAccuracy of tree1 (Method 1): {accuracy_tree1:.2f}%\n")
print(f"\nAccuracy of tree2 (Method 2): {accuracy_tree2:.2f}%\n")