from math import log2
from queue import PriorityQueue
import matplotlib.pyplot as plt

DEBUG = True

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
    
##### TESTING ######

# 0.5 as we are ML
def predict_class(document, Theta_i_atheism, Theta_i_books, theta_atheism=0.5, theta_books=0.5):
    # Start with the log of the priors to avoid underflow
    log_prob_atheism = log2(theta_atheism)
    log_prob_books = log2(theta_books)
    
    # Add the log likelihood of each word in the document
    for word_id in document:
        if word_id in Theta_i_atheism:
            log_prob_atheism += log2(Theta_i_atheism[word_id])
        if word_id in Theta_i_books:
            log_prob_books += log2(Theta_i_books[word_id])
    
    # Return the class with the highest posterior probability
    if log_prob_atheism > log_prob_books:
        return ATHEISM_ID
    elif log_prob_atheism < log_prob_books:
        return BOOKS_ID
    else: # TIE
        return ATHEISM_ID

##### MAIN ######

if __name__ == '__main__':
    
    # Reading data from files
    words = read_word_data('./dataset/words.txt')
    train_data = read_document_data('./dataset/trainData.txt')
    train_labels = read_label_data('./dataset/trainLabel.txt')
    test_data = read_document_data('./dataset/testData.txt')
    test_labels = read_label_data('./dataset/testLabel.txt')
    
    #### First calculate relative frequency of books belong to subreddit 'Atheism' or 'Books' ####
    total_count = 0
    atheism_count = 0
    books_count = 0

    for _, label in train_labels.items():
        total_count += 1
        if label == ATHEISM_ID:
            atheism_count += 1
        elif label == BOOKS_ID:
            books_count += 1
    
    theta_atheism = atheism_count / total_count
    theta_books = books_count / total_count
    
    if DEBUG:
        print(f"Total number of documents: {total_count}")
        print(f"Number of documents labeled as 'Atheism': {atheism_count}")
        print(f"Number of documents labeled as 'Books': {books_count}")
        print(f"Prior probability of 'Atheism' class: {theta_atheism:.4f}")
        print(f"Prior probability of 'Books' class: {theta_books:.4f}")
    
    #### Split train data for each label ####
    atheism_train_data = {}
    books_train_data = {}
    
    # Split the train_data based on the labels
    for doc_id, label in train_labels.items():
        if label == ATHEISM_ID:
            atheism_train_data[doc_id] = train_data[doc_id]
        elif label == BOOKS_ID:
            books_train_data[doc_id] = train_data[doc_id]
            
    #### calculate number of document (value) that have feature word_id (key) ####
    
    atheism_word_counts = {i: 0 for i in range(1, NUM_FEATURES + 1)} 
    for i in range(1, NUM_FEATURES + 1):
        # count the occurance of word_id = i
        for _, word_ids in atheism_train_data.items():
            if i in word_ids:
                atheism_word_counts[i] += 1

    books_word_counts = {i: 0 for i in range(1, NUM_FEATURES + 1)} 
    for i in range(1, NUM_FEATURES + 1):
        # count the occurance of word_id = i
        for _, word_ids in books_train_data.items():
            if i in word_ids:
                books_word_counts[i] += 1
                
    #### Account for Laplace correction when calculating the actual theta_i_1/0 ####
    
    Theta_i_atheism = {i: 0 for i in range(1, NUM_FEATURES + 1)}
    Theta_i_books = {i: 0 for i in range(1, NUM_FEATURES + 1)}
    
    for i in range(1, NUM_FEATURES + 1):
        Theta_i_atheism[i] = (atheism_word_counts[i] + 1) / (atheism_count + 2)
        Theta_i_books[i] = (books_word_counts[i] + 1) / (books_count + 2)
        
    ######################## PROCESSING FINISHED ########################
    
    print("")
    
    #### 10 most discriminative word features ####
    
    # Compute the log probabilities for each word
    log_prob_diffs = {}
    for word_id in range(1, NUM_FEATURES + 1):
        log_prob_atheism = log2(Theta_i_atheism[word_id])
        log_prob_books = log2(Theta_i_books[word_id])
        log_prob_diffs[word_id] = abs(log_prob_atheism - log_prob_books)

    # Sort the words by the most discriminative features
    most_discriminative = sorted(log_prob_diffs, key=log_prob_diffs.get, reverse=True)[:10]

    # Print the 10 most discriminative words
    print("10 Most Discriminative Word Features:")
    for word_id in most_discriminative:
        word = words[word_id]
        print(f"Word: {word}, Difference in Log Probabilities: {log_prob_diffs[word_id]:.4f}")
        
    print("")
    
    #### Predict the class for each document in the training set ####
    predicted_labels = {}
    for doc_id, document in train_data.items():
        predicted_labels[doc_id] = predict_class(document, Theta_i_atheism, Theta_i_books)

    # Calculate the accuracy
    correct_predictions = sum([predicted_labels[doc_id] == test_labels[doc_id] for doc_id in test_labels])
    accuracy = correct_predictions / len(test_labels)

    print(f'Training accuracy of the Naive Bayes classifier (): {accuracy * 100:.2f}%')
    
    #### Predict the class for each document in the test set ####
    predicted_labels = {}
    for doc_id, document in test_data.items():
        predicted_labels[doc_id] = predict_class(document, Theta_i_atheism, Theta_i_books)

    # Calculate the accuracy
    correct_predictions = sum([predicted_labels[doc_id] == test_labels[doc_id] for doc_id in test_labels])
    accuracy = correct_predictions / len(test_labels)

    print(f'Testing accuracy of the Naive Bayes classifier (): {accuracy * 100:.2f}%')
    
    print("")