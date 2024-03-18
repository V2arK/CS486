import array

# Create an array of integer type
my_array = array.array('i', [2, 2, 3, 4, 5])
# Remove the element '3' from the array
my_array.remove(3)
print(my_array)