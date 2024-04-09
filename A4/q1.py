from enum import Enum, auto
import numpy as np

### Definition ###

# Define enumerations for the medical conditions and gene presence
class Sloepnea(Enum):
    ABSENT = 0
    PRESENT = 1

class Foriennditis(Enum):
    ABSENT = 0
    PRESENT = 1

class DegarSpots(Enum):
    ABSENT = 0
    PRESENT = 1

class TRIMONOHTS(Enum):
    ABSENT = 0
    PRESENT = 1

class DunettsSyndrome(Enum):
    UNKNOWN = -1
    ABSENT = 0
    MILD = 1
    SEVERE = 2
    
### Prior (Guess) ###

# parents

# Trimono-HT/S Gene's probabilityof being true. 
# [false, true]
P_TG = np.array([0.9, 0.1]) 

# Dunetts Syndrome's probability of being true. 
# [not present, mild, severe]
P_DS = np.array([0.5, 0.25, 0.25]) 

# childs

# Sloepnea's probability of being true. 
# [TG = false, DR = 0,1,2], [TG = True, DR = 0,1,2]
P_S = np.array([[0.03, 0.485, 0.485], [0.01, 0.01, 0.01]]) 
 
# Degar Spots's probability of being true. 
# [DR = 0,1,2]
P_DS = np.array([0.05, 0.25, 0.70])

# Foriennditis's probability of being true. 
# [DR = 0,1,2]
P_F = np.array([0.05, 0.70, 0.25])

### Reading Data ###

# A function to read the data file and store the variables
def read_and_store_data(filepath):
    data = []
    with open(filepath, 'r') as file:
        for line in file:
            a, b, c, d, e = map(int, line.split())
            record = {
                'S': Sloepnea(a),
                'F': Foriennditis(b),
                'D': DegarSpots(c),
                'TG': TRIMONOHTS(d),
                'DS': DunettsSyndrome(e),
            }
            data.append(record)
    return data

### Expectation Maximization ###

def apply_noise(delta, p, data):
    np.random.seed(p)

#def EM(delta, p, data):
    

### MAIN ###

trainfilepath = "em-data/traindata.txt"
data = read_and_store_data(trainfilepath)

# `data` is now a list of dictionaries where each dictionary represents a record from the input file.
'''
# Function to display information of the first 12 persons
def display_first_12_persons(data):
    for i, record in enumerate(data[:12], start=1):
        print(f"Person {i}:")
        for condition, value in record.items():
            print(f"  {condition}: {value.name}")
        print()  # Add an empty line for better readability

# Display the information for the first 10 persons
display_first_12_persons(data)
'''

