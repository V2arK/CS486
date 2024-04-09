import math
import collections
import decimal

testData = open("dataset/testData.txt", "r")
testLabel = open("dataset/testLabel.txt", "r")
trainData = open("dataset/trainData.txt", "r")
trainLabel = open("dataset/trainLabel.txt", "r")
words = open("dataset/words.txt", "r")

# actual word as string
feature_words = words.read().splitlines()
dataset = collections.defaultdict(set, {k:set() for k in range(1,1501)})
for line in trainData:
    doc_id, word_id = map(int, line.strip().split())
    dataset[doc_id].add(word_id)

labels = [int(line.strip()) for line in trainLabel]

dataset1 = {}
dataset2 = {}
for i in range(1,len(dataset)+1):
    if labels[i-1] == 1:
        dataset1[i] = dataset[i]
    else:
        dataset2[i] = dataset[i]
        
        
n = len(feature_words)
theta_i1 = []
theta_i2 = []
theta_c = len(dataset1)/1500

for i in range(1,n+1):
    count = 0
    for word in dataset1.values():
        if i in word:
            count += 1
    theta_i1.append((count + 1)/(len(dataset1) + 2))
    count = 0
    for word in dataset2.values():
        if i in word:
            count += 1
    theta_i2.append((count + 1)/(len(dataset2) + 2))
def predict1(word_set: set):
    sol = decimal.Decimal(theta_c)
    for i in range(1,n+1):
        if i in word_set:
            sol *= decimal.Decimal(theta_i1[i-1])
        else:
            sol *= decimal.Decimal(1-theta_i1[i-1])
    return sol
def predict2(word_set: set):
    sol = decimal.Decimal(1-theta_c)
    for i in range(1,n+1):
        if i in word_set:
            sol *= decimal.Decimal(theta_i2[i-1])
        else:
            sol *= decimal.Decimal(1-theta_i2[i-1])
    return sol
# precit the label of doc given word_set
def predict(word_set):
    prob = predict1(word_set)/ (predict1(word_set) + predict2(word_set))
    if prob >=0.5:
        return 1
    else:
        return 2
    
test_dataset = collections.defaultdict(set, {k:set() for k in range(1,1501)})
for line in testData:
    doc_id, word_id = map(int, line.strip().split())
    test_dataset[doc_id].add(word_id)

test_labels = [int(line.strip()) for line in testLabel]


def find_acc(data, labels):
    correct = 0
    for doc_id, word_set in data.items():
        prediction = predict(word_set)
        if prediction == labels[doc_id - 1]:
            correct += 1
    return correct / len(labels)

train_acc = find_acc(dataset, labels)
test_acc = find_acc(test_dataset, test_labels)
print(train_acc, test_acc)

index = 1
for j in theta_i1[:100]:
    print(f"key = {index}, value = {j}")
    index += 1

abs_diff = [abs(math.log(theta_i1[i]) - math.log(theta_i2[i])) for i in range(len(theta_i1))]
top10 = sorted(zip(feature_words, abs_diff), key=lambda x: x[1], reverse=True)[:10]
print(*top10, sep = '\n')

testData.close()
testLabel.close()
trainData.close()
trainLabel.close()
words.close()