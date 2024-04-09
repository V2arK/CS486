# version 1.1

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from neural_net import NeuralNetwork
from operations import *

from sklearn.model_selection import KFold

def load_dataset(csv_path, target_feature):
    dataset = pd.read_csv(csv_path)
    t = np.expand_dims(dataset[target_feature].to_numpy().astype(float), axis=1)
    X = dataset.drop([target_feature], axis=1).to_numpy()
    return X, t


'''
X, y = load_dataset("data/wine_quality.csv", "quality")

n_features = X.shape[1]
net = NeuralNetwork(n_features, [32,32,16,1], [ReLU(), ReLU(), Sigmoid(), Identity()], MeanSquaredError(), learning_rate=0.001)
epochs = 500

test_split = 0.1
X_train = X[:int((1 - test_split) * X.shape[0])]
X_test = X[int((1 - test_split) * X.shape[0]):]
y_train = y[:int((1 - test_split) * y.shape[0])]
y_test = y[int((1 - test_split) * y.shape[0]):]

trained_W, epoch_losses = net.train(X_train, y_train, epochs)
print("Error on test set: {}".format(net.evaluate(X_test, y_test, mean_absolute_error)))

plt.plot(np.arange(0, epochs), epoch_losses)
plt.show()
'''

# Part B 2

X, y = load_dataset("data/wine_quality.csv", "quality")

n_features = X.shape[1]

# Hyperparameters and settings
n_splits = 5
epochs = 500
learning_rate = 0.001

# Initialize k-fold cross-validation
kf = KFold(n_splits=n_splits, shuffle=True, random_state=None)

# Initialize variables to store metrics
fold_losses = []
fold_maes = []
epoch_losses = np.zeros((n_splits, epochs))

# neural network architecture
layer_sizes = [32,32,16,1]
activations = [ReLU(), ReLU(), Sigmoid(), Identity()]


# K-Fold Cross Validation
for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
    print(f"Training on fold {fold+1}/{n_splits}...")
    
    # Split data into training and validation sets
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    
    # Initialize the neural network
    nn = NeuralNetwork(n_features=n_features, 
                       layer_sizes=layer_sizes, 
                       activations=activations, 
                       loss=MeanSquaredError(), 
                       learning_rate=learning_rate)
    
    # Train the neural network
    W, epoch_loss = nn.train(X_train, y_train, epochs=epochs)
    epoch_losses[fold, :] = epoch_loss
    
    # Evaluate on validation set
    mae = nn.evaluate(X_val, y_val, mean_absolute_error)
    fold_maes.append(mae)
    print(f"Fold {fold+1} MAE: {mae}")

# Average training loss over folds for each epoch
average_epoch_losses = epoch_losses.mean(axis=0)
std_deviation_maes = np.std(fold_maes)
average_mae = np.mean(fold_maes)

# Output the results
print(f"Average MAE over all folds: {average_mae}")
print(f"Standard Deviation of MAE over all folds: {std_deviation_maes}")

# Plot the results
plt.plot(range(epochs), average_epoch_losses, label='Average Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Average Loss')
plt.title('Training Loss Over Epochs')
plt.legend()
plt.show()
