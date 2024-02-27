# Sample dictionary
my_dict = {
    1: [1, 2, 3],
    2: [4, 5, 6],
    3: [7, 8, 9, 2],
    4: [10, 11, 12]
}

# Specific value to look for
specific_value = 2

# Selecting keys whose arrays contain the specific value
selected_keys = [key for key, value in my_dict.items() if specific_value in value]

print(selected_keys)