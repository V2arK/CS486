import numpy as np
from math import log2
from queue import PriorityQueue
import matplotlib.pyplot as plt

##### Variables ######
NUM_FEATURES = 0 # Define later
NUM_SAMPLE = 1500
ATHEISM_ID = 1
BOOKS_ID = 2

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
    
    if debug:
        print(f"len has_word_to_split = {len(has_word_to_split)}, len not_have_word_to_split = {len(not_have_word_to_split)}")
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
            print("case 0")
            return IE
        elif sum_E1 == 0:
            if debug:
                print("case 1")
            return IE - ((atheism_count_E2) / sum_E2 * IE2)
        elif sum_E2 == 0:
            if debug:
                print("case 2")
            return IE - ((atheism_count_E1) / sum_E1 * IE1)
        
        return IE - ((atheism_count_E1) / sum_E1 * IE1) - ((atheism_count_E2) / sum_E2 * IE2)


##### Decision Tree #####

class Node:
    def __init__(self, dataset, point_estimate, feature_to_split=None, info_gain=0):
        self.dataset = dataset  # The subset 'E' of the dataset at this node
        self.feature_to_split = feature_to_split  # 'X_prime': The feature to split on at this node
        self.point_estimate = point_estimate
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
    
def find_best_to_split(dataset, label_dict, method, words=None):
    # Find the best feature to split next
    # by try to compute the delta_information_gain on all features 
    # appeared in the dataset, and return a tuple containing
    # the feature to split that will give the biggest delta_information_gain
    # and the delta_information_gain.
    best_feature = None
    best_info_gain = 0  # Start with 0 to ensure any gain is better, but not no split

    # Iterate through each feature in the dataset to find the best one to split on
    for feature in range(1, NUM_FEATURES + 1):
        # Compute the information gain for splitting on the current feature
        current_info_gain = delta_information_gain(dataset, label_dict, feature, method)
        
        # If the information gain of the current feature is better than the best one so far, update the best feature and gain
        if current_info_gain > best_info_gain:
            #print(f"better word = {words[feature]}, info gain = {current_info_gain} ")
            #print(f"delta I = {delta_information_gain(dataset, label_dict, feature, method, True)}")
            #print("------------")
            best_feature = feature
            best_info_gain = current_info_gain

    if words is not None and best_feature is not None:
        print(f"spliting feature = {words[best_feature]}, info gain = {best_info_gain}")
    elif words is not None:
        print(f"spliting feature = None, info gain = {best_info_gain}")
        
    return best_feature, best_info_gain

def split_dataset(dataset, feature_to_split):
    # Datasets to hold the split
    dataset_with_feature = {}
    dataset_without_feature = {}
    print(f"Splitting on {feature_to_split}")
    # Iterate over each entry in the dataset
    for doc_id, word_ids in dataset.items():
        # Check if the feature to split on is in the document's word IDs
        if feature_to_split in word_ids:
            # Add this document to the dataset with the feature
            #print(f"with_feature: doc_id = {doc_id}, features = {word_ids}")
            dataset_with_feature[doc_id] = word_ids
        else:
            # Add this document to the dataset without the feature
            #print(f"without_feature: doc_id = {doc_id}, features = {word_ids}")
            dataset_without_feature[doc_id] = word_ids
    
    
    return dataset_with_feature, dataset_without_feature

# Function to build the decision tree
def build_decision_tree(train_data, train_labels, method, subreddit_dict, max_nodes=100, words=None):
    pq = PriorityQueue()
    root = Node(dataset=train_data, point_estimate=calculate_point_estimate(train_data, subreddit_dict))
    pq.put((-1, root))  # We use -1 as the priority since PriorityQueue is min-based

    while not pq.empty() and max_nodes > 0:
        max_nodes -= 1
        _, current_node = pq.get()
        print(f"doing the {100 - max_nodes}'th node")
    
        # Determine the best feature to split on and the gain from that split
        best_feature, best_info_gain = find_best_to_split(current_node.dataset, train_labels, method, words=words)
        
        # If no good split is found, this is a leaf node
        if best_feature is None:
            continue
        
        # Split the dataset based on the best feature
        left_dataset, right_dataset = split_dataset(current_node.dataset, best_feature)
        left_node = Node(dataset=left_dataset, point_estimate=calculate_point_estimate(left_dataset, subreddit_dict))
        right_node = Node(dataset=right_dataset, point_estimate=calculate_point_estimate(right_dataset, subreddit_dict))
        
        # Update current node with split info
        current_node.feature_to_split = best_feature
        current_node.info_gain = best_info_gain
        current_node.left = left_node
        current_node.right = right_node
        
        # Add child nodes to the priority queue
        pq.put((best_info_gain, left_node))
        pq.put((best_info_gain, right_node))
    
    return root

def print_tree(node, depth=0, feature_names=None):
    # Base case: if the node is a leaf, it will not have a feature to split on
    if node.feature_to_split is None:
        print(" " * depth + "Leaf, estimate: " + str(node.point_estimate))
        return

    # Recursive case: print the current node's split information
    if feature_names and node.feature_to_split in feature_names:
        feature_name = feature_names[node.feature_to_split]
    else:
        feature_name = str(node.feature_to_split)

    print(" " * depth + f"Node: Split Feature = {feature_name}, Info Gain = {node.info_gain:.4f}")

    # Recursively print the left subtree
    print(" " * depth + "L (w/ feature):")
    print_tree(node.left, depth + 1, feature_names)

    # Recursively print the right subtree
    print(" " * depth + "R (wo/ feature):")
    print_tree(node.right, depth + 1, feature_names)
    
##### MAIN ######

# Reading data from files
words = read_word_data('./words.txt')
train_data = read_document_data('./trainData.txt')
train_labels = read_label_data('./trainLabel.txt')
test_data = read_document_data('./testData.txt')
test_labels = read_label_data('./testLabel.txt')

tree1 = build_decision_tree(train_data, train_labels, method=1, subreddit_dict=train_labels, words=words)
#tree2 = build_decision_tree(train_data, train_labels, 2)

print_tree(tree1, feature_names=words)

'''
# Print the structure of the first few entries of each dictionary for verification
print("Train Data (first few entries):", list(train_data.items())[:5])
print("Train Labels (first few entries):", list(train_labels.items())[:5])
print("Test Data (first few entries):", list(test_data.items())[:5])
print("Test Labels (first few entries):", list(test_labels.items())[:5])
print("Words (first few entries):", list(words.items())[:5])
'''
