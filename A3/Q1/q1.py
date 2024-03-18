from math import log2
from queue import PriorityQueue
import matplotlib.pyplot as plt
from multiprocessing import Pool

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
    file.close()
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
    file.close()
    return word_data

# Function to read train or test data
# document_data[n] = arrray of word_id that n+1'th doc have.
def read_document_data(file_name):
    document_data = {i: [] for i in range(1, NUM_SAMPLE + 1)} # 1 to 1500 
    with open(file_name, 'r') as file:
        for line in file:
            doc_id, word_id = line.strip().split()
            document_data[int(doc_id)].append(int(word_id))
    file.close()
    return document_data

##### COMPUTING #####

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
    
    #print(f"delta I = {delta_information_gain(dataset, label_dict, best_feature, method, True)}")
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
    
    #print(f"L size: {str(len(dataset_with_feature))}, R size:  {str(len(dataset_without_feature))}")
    return dataset_with_feature, dataset_without_feature

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


if __name__ == '__main__':
    
    # Reading data from files
    words = read_word_data('./dataset/words.txt')
    train_data = read_document_data('./dataset/trainData.txt')
    train_labels = read_label_data('./dataset/trainLabel.txt')
    test_data = read_document_data('./dataset/testData.txt')
    test_labels = read_label_data('./dataset/testLabel.txt')
    
    # First calculate relative frequency of books belong to subreddit 'Atheism' or 'Books'
    
    
    

