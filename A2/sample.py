import queue

def print_priority_queue(pq):
    temp_list = []
    # Remove all items and add them to a temporary list
    while not pq.empty():
        item = pq.get()
        print(item, end=' ')
        temp_list.append(item)
    
    # Re-insert the items into the PriorityQueue
    for item in temp_list:
        pq.put(item)

    print("\n")

# Create a priority queue
pq = queue.PriorityQueue()

# Put some items in the queue. The first element of the tuple is the priority.
pq.put((10, 'ten'))
pq.put((1, 'one'))
pq.put((5, 'five'))

print_priority_queue(pq)

# Pop an item from the queue.
item = pq.get()

print(f"Item popped from the queue: {item}\n")
print_priority_queue(pq)
# This will print: Item popped from the queue: (1, 'one')