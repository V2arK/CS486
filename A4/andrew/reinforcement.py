import random
import numpy as np
import sys
import matplotlib.pyplot as plt

class Sender:
    """
    A Q-learning agent that sends messages to a Receiver

    """
    def __init__(self, num_sym:int, grid_rows:int, grid_cols:int, alpha_i:float, alpha_f:float, num_ep:int, epsilon:float, discount:float):
        """
        Initializes this agent with a state, set of possible actions, and a means of storing Q-values

        :param num_sym: The number of arbitrary symbols available for sending
        :type num_sym: int
        :param grid_rows: The number of rows in the grid
        :type grid_rows: int
        :param grid_cols: The number of columns in the grid
        :type grid_cols: int
        :param alpha_i: The initial learning rate
        :type alpha: float
        :param alpha_f: The final learning rate
        :type alpha: float
        :param num_ep: The total number of episodes
        :type num_ep: int
        :param epsilon: The epsilon in epsilon-greedy exploration
        :type epsilon: float
        :param discount: The discount factor
        :type discount: float
        :param q_vals: Q-values table  (#cells in the grid * #actions/symbols can take)
        """
        self.actions = range(num_sym)
        self.alpha = alpha_i
        self.alpha_i = alpha_i
        self.alpha_f = alpha_f
        self.num_ep = num_ep
        self.epsilon = epsilon
        self.discount = discount
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols
        # Your code here!
        grid_size = (grid_rows * grid_cols)
        self.q_vals = np.zeros((grid_size, num_sym))
        
    def state_to_index(self, state):
        """
        Convert a state (x, y) to the index used in self.q_vals.
        This method assumes the grid is linearized row-wise.
        """
        x, y = state
        return (x * self.grid_cols) + y  
    
    def select_action(self, state):
        """
        This function is called every time the agent must act. It produces the action that the agent will take
        based on its current state

        :param state: the state the agent is acting from, in the form (x,y), which are the coordinates of the prize
        :type state: (int, int)
        :return: The symbol to be transmitted (must be an int < N)
        :rtype: int
        """
        # Your code here!
        state_index = self.state_to_index(state)
        if np.random.rand() < self.epsilon:  
            # Explore
            return np.random.choice(self.actions)
        else:  
            # Exploit
            return np.argmax(self.q_vals[state_index])

    def update_q(self, old_state, action, reward):
        """
        This function is called after an action is resolved so that the agent can update its Q-values

        :param old_state: the state the agent was in when it acted, in the form (x,y), which are the coordinates
                          of the prize
        :type old_state: (int, int)
        :param action: the action that was taken
        :type action: int
        :param reward: the reward that was received
        :type reward: float
        """
        # Your code here!
        old_state_index = self.state_to_index(old_state)
        # sender doesn't consider future states(drop 𝛾 * max_a Q) as state of prize is the same for each episode
        # += as value will be add on to that when a new action
        self.q_vals[old_state_index, action] += self.alpha * (reward - self.q_vals[old_state_index, action])
    
    def adjust_learning_rate(self, episode):
        """
        Adjusts the learning rate linearly between alpha_i and alpha_f based on the current episode.
        """
        fraction = min(episode / self.num_ep, 1)
        self.alpha = self.alpha_i + fraction * (self.alpha_f - self.alpha_i)


class Receiver:
    """
    A Q-learning agent that receives a message from a Sender and then navigates a grid

    """
    def __init__(self, num_sym:int, grid_rows:int, grid_cols:int, alpha_i:float, alpha_f:float, num_ep:int, epsilon:float, discount:float):
        """
        Initializes this agent with a state, set of possible actions, and a means of storing Q-values

        :param num_sym: The number of arbitrary symbols available for sending
        :type num_sym: int
        :param grid_rows: The number of rows in the grid
        :type grid_rows: int
        :param grid_cols: The number of columns in the grid
        :type grid_cols: int
        :param alpha_i: The initial learning rate
        :type alpha: float
        :param alpha_f: The final learning rate
        :type alpha: float
        :param num_ep: The total number of episodes
        :type num_ep: int
        :param epsilon: The epsilon in epsilon-greedy exploration
        :type epsilon: float
        :param discount: The discount factor
        :type discount: float
        """
        self.actions = [0,1,2,3] # Note: these correspond to [up, down, left, right]
        self.alpha = alpha_i
        self.alpha_i = alpha_i
        self.alpha_f = alpha_f
        self.num_ep = num_ep
        self.epsilon = epsilon
        self.discount = discount
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols
        # Your code here!
        # Initialize the Q-values table with dimensions: [num_messages, grid_size, num_actions]
        self.q_vals = np.zeros((num_sym, grid_rows * grid_cols, len(self.actions)))

    def state_to_index(self, state):
        """
        Convert a state (m, x, y) to an index for accessing the Q-values table.
        This method assumes a linearized representation of the grid.
        """
        m, x, y = state
        return m, (x * self.grid_cols) + y  # Adjust the multiplier based on actual grid dimensions if different
    
    def select_action(self, state):
        """
        This function is called every time the agent must act. It produces the action that the agent will take
        based on its current state
        :param state: the state the agent is acting from, in the form (m,x,y), where m is the message received
                      and (x,y) are the board coordinates
        :type state: (int, int, int)
        :return: The direction to move, where 0 is up, 1 is down, 2 is left, and 3 is right
        :rtype: int
        """
        # Your code here!
        state_index = self.state_to_index(state)
        if np.random.rand() < self.epsilon:  # Explore
            return np.random.choice(self.actions)
        else:  # Exploit
            return np.argmax(self.q_vals[state_index])

    def update_q(self, old_state, new_state, action, reward):
        """
        This function is called after an action is resolved so that the agent can update its Q-values

        :param old_state: the state the agent was in when it acted in the form (m,x,y), where m is the message received
                          and (x,y) are the board coordinates
        :type old_state: (int, int, int)
        :param new_state: the state the agent entered after it acted
        :type new_state: (int, int, int)
        :param action: the action that was taken
        :type action: int
        :param reward: the reward that was received
        :type reward: float
        """
        # Your code here!
        old_state_index = self.state_to_index(old_state)
        new_state_index = self.state_to_index(new_state)
        # append action to the old_state_index to from 3-dimentions
        old_q_value = self.q_vals[old_state_index + (action,)]
        max_future_q = np.max(self.q_vals[new_state_index])
        self.q_vals[old_state_index + (action,)] = old_q_value + self.alpha * (reward + self.discount * max_future_q - old_q_value)
    def adjust_learning_rate(self, episode):
        """
        Adjusts the learning rate linearly between alpha_i and alpha_f based on the current episode.
        """
        fraction = min(episode / self.num_ep, 1)
        self.alpha = self.alpha_i + fraction * (self.alpha_f - self.alpha_i)

def get_grid(grid_name:str):
    """
    This function produces one of the three grids defined in the assignment as a nested list

    :param grid_name: the name of the grid. Should be one of 'fourroom', 'maze', or 'empty'
    :type grid_name: str
    :return: The corresponding grid, where True indicates a wall and False a space
    :rtype: list[list[bool]]
    """
    grid = [[False for i in range(5)] for j in range(5)] # default case is 'empty'
    if grid_name == 'fourroom':
        grid[0][2] = True
        grid[2][0] = True
        grid[2][1] = True
        grid[2][3] = True
        grid[2][4] = True
        grid[4][2] = True
    elif grid_name == 'maze':
        grid[1][1] = True
        grid[1][2] = True
        grid[1][3] = True
        grid[2][3] = True
        grid[3][1] = True
        grid[4][1] = True
        grid[4][2] = True
        grid[4][3] = True
        grid[4][4] = True
    return grid

def legal_move(posn_x:int, posn_y:int, move_id:int, grid:list[list[bool]]):
    """
    Produces the new position after a move starting from (posn_x,posn_y) if it is legal on the given grid (i.e. not
    out of bounds or into a wall)

    :param posn_x: The x position (column) from which the move originates
    :type posn_x: int
    :param posn_y: The y position (row) from which the move originates
    :type posn_y: int
    :param move_id: The direction to move, where 0 is up, 1 is down, 2 is left, and 3 is right
    :type move_id: int
    :param grid: The grid on which to move, where False indicates a space and True a wall
    :type grid: list[list[bool]]
    :return: The new (x,y) position if the move was legal, or the old position if it was not
    :rtype: (int, int)
    """
    moves = [[0,-1],[0,1],[-1,0],[1,0]]
    new_x = posn_x + moves[move_id][0]
    new_y = posn_y + moves[move_id][1]
    result = (new_x,new_y)
    if new_x < 0 or new_y < 0 or new_x >= len(grid[0]) or new_y >= len(grid):
        result = (posn_x,posn_y)
    else:
        if grid[new_y][new_x]:
            result = (posn_x,posn_y)
    return result

def run_episodes(sender:Sender, receiver:Receiver, grid:list[list[bool]], num_ep:int, delta:float):
    """
    Runs the reinforcement learning scenario for the specified number of episodes

    :param sender: The Sender agent
    :type sender: Sender
    :param receiver: The Receiver agent
    :type receiver: Receiver
    :param grid: The grid on which to move, where False indicates a space and True a wall
    :type grid: list[list[bool]]
    :param num_ep: The number of episodes
    :type num_ep: int
    :param delta: The chance of termination after every step of the receiver
    :type delta: float [0,1]
    :return: A list of the reward received by each agent at the end of every episode
    :rtype: list[float]
    """
    reward_vals = []

    # Episode loop
    for ep in range(num_ep):
        # Set receiver starting position
        receiver_x = 2
        receiver_y = 2

        # Choose prize position
        prize_x = np.random.randint(len(grid[0]))
        prize_y = np.random.randint(len(grid))
        while grid[prize_y][prize_x] or (prize_x == receiver_x and prize_y == receiver_y):
            prize_x = np.random.randint(len(grid[0]))
            prize_y = np.random.randint(len(grid))

        # Initialize new episode
        # (sender acts)
        # Your code here!
        message = sender.select_action((prize_x, prize_y))
        # Receiver loop
        # (receiver acts, check for prize, check for random termination, update receiver Q-value)
        terminate = False
        total_reward = 0
        while not terminate:
            # Your code here!
            action = receiver.select_action((message, receiver_x, receiver_y))
            new_x, new_y = legal_move(receiver_x, receiver_y, action, grid)
            
            # Check for prize
            if new_x == prize_x and new_y == prize_y:
                total_reward = 1
                terminate = True
            
            # Check for random termination
            if np.random.rand() < delta:
                terminate = True
            
            # Update receiver's Q-value
            receiver.update_q((message, receiver_x, receiver_y), (message, new_x, new_y), action, total_reward)
            receiver_x, receiver_y = new_x, new_y  # Update receiver's position

        #Finish up episode
        # (update sender Q-value, update alpha values, append reward to output list)
        # Your code here!
        # Update sender's Q-value (assuming the sender gets the same reward as the receiver)
        sender.update_q((prize_x, prize_y), message, total_reward)

        # Update learning rate for both agents
        sender.adjust_learning_rate(ep)
        receiver.adjust_learning_rate(ep)

        # Record the reward
        reward_vals.append(total_reward)

    return reward_vals
def visualize_policy(receiver, sender, grid):
    sender.epsilon = 0.0
    receiver.epsilon = 0.0
    action_symbols = ['↑', '↓', '←', '→']
    grid_size = len(grid)
    policy_grids = [[None for _ in range(grid_size)] for _ in range(grid_size)]
    sender_policy_grid = [[None for _ in range(grid_size)] for _ in range(grid_size)]

    # Visualize Receiver's policy for each message
    for message in range(len(sender.actions)):
        print(f"Receiver's Policy for message {message}:")
        for x in range(grid_size):
            for y in range(grid_size):
                if grid[x][y]:
                    policy_grids[x][y] = '█'  # Indicates a wall
                else:
                    state = (message, x, y)
                    best_action = receiver.select_action(state)
                    policy_grids[x][y] = action_symbols[best_action]
        for row in policy_grids:
            print(' '.join(row))
        print("\n")

    # Visualize Sender's policy
    print("Sender's Policy (preferred message for each prize location):")
    for x in range(grid_size):
        for y in range(grid_size):
            if grid[x][y]:
                sender_policy_grid[x][y] = '█'  # Indicates a wall
            else:
                state = (x, y)
                best_message = sender.select_action(state)
                sender_policy_grid[x][y] = str(best_message)
    for row in sender_policy_grid:
        print(' '.join(row))
    print("\n")
    
def questionb():
    epsilons = [0.01, 0.1, 0.4]
    nep_values = [10, 100, 1000, 10000, 50000, 100000]
    num_tests = 10
    num_signals = 4  # Number of symbols/messages
    discount = 0.95
    delta = 1 - discount
    alpha_init = 0.9
    alpha_final = 0.01
    test_episodes = 1000  # Episodes for testing

    # Placeholder for results
    results = {epsilon: {nep: [] for nep in nep_values} for epsilon in epsilons}

    for epsilon in epsilons:
        for nep in nep_values:
            test_rewards = []
            for test in range(num_tests):
                # Initialize agents
                sender = Sender(num_signals, 5, 5, alpha_init, alpha_final, nep, epsilon, discount)
                receiver = Receiver(num_signals, 5, 5, alpha_init, alpha_final, nep, epsilon, discount)
                
                grid = get_grid('fourroom')  # Use the four room grid
                learn_rewards = run_episodes(sender, receiver, grid, nep, delta)
                
                # Set epsilon to 0 for testing
                sender.epsilon = 0.0
                receiver.epsilon = 0.0
                
                test_rewards.extend(run_episodes(sender, receiver, grid, test_episodes, delta))
            
            # Compute average and standard deviation of test rewards
            avg_reward = np.mean(test_rewards)
            std_dev = np.std(test_rewards)
            results[epsilon][nep] = (avg_reward, std_dev)

    # Plotting
    for epsilon in epsilons:
        avg_rewards = [results[epsilon][nep][0] for nep in nep_values]
        std_devs = [results[epsilon][nep][1] for nep in nep_values]
        
        plt.errorbar(np.log10(nep_values), avg_rewards, yerr=std_devs, label=f'epsilon={epsilon}', fmt='-o', capsize=5)
        
    plt.xlabel('log(Nep)')
    plt.ylabel('Average Discounted Reward')
    plt.title('Average Discounted Reward vs. log(Nep)')
    plt.legend()
    plt.show()
    epsilon_example = 0.1
    nep_example = 100000
    sender_example = Sender(num_signals, len(grid), len(grid[0]), 0.9, 0.01, nep_example, epsilon_example, discount)
    receiver_example = Receiver(num_signals, len(grid), len(grid[0]), 0.9, 0.01, nep_example, epsilon_example, discount)

    # Assuming run_episodes function modifies the agents' Q-tables in-place
    run_episodes(sender_example, receiver_example, grid, nep_example, delta)

    # Visualize policy
    visualize_policy(receiver_example, sender_example, grid)

def questionc():
    epsilon = 0.1
    nep_values = [10, 100, 1000, 10000, 50000, 100000]
    num_tests = 10
    discount = 0.95
    delta = 1 - discount
    alpha_init = 0.9
    alpha_final = 0.01
    test_episodes = 1000  # Episodes for testing
    num_signals_values = [2, 4, 10]  # N values to test

    # Placeholder for results
    results = {num_signals: [] for num_signals in num_signals_values}

    for num_signals in num_signals_values:
        avg_rewards = []
        std_devs = []
        for nep in nep_values:
            test_rewards = []
            for test in range(num_tests):
                # Initialize agents
                sender = Sender(num_signals, 5, 5, alpha_init, alpha_final, nep, epsilon, discount)
                receiver = Receiver(num_signals, 5, 5, alpha_init, alpha_final, nep, epsilon, discount)
                
                grid = get_grid('fourroom')  # Use the four room grid
                learn_rewards = run_episodes(sender, receiver, grid, nep, delta)
                
                # Set epsilon to 0 for testing
                sender.epsilon = 0.0
                receiver.epsilon = 0.0
                
                test_rewards.extend(run_episodes(sender, receiver, grid, test_episodes, delta))
            
            # Compute average and standard deviation of test rewards
            avg_reward = np.mean(test_rewards)
            std_dev = np.std(test_rewards)
            avg_rewards.append(avg_reward)
            std_devs.append(std_dev)
        
        results[num_signals] = (avg_rewards, std_devs)

    # Plotting
    for num_signals, (avg_rewards, std_devs) in results.items():
        plt.errorbar(np.log10(nep_values), avg_rewards, yerr=std_devs, label=f'N={num_signals}', fmt='-o', capsize=5)

    plt.xlabel('log(Nep)')
    plt.ylabel('Average Discounted Reward')
    plt.title('Average Discounted Reward vs. log(Nep) for different N values')
    plt.legend()
    plt.show()
    
def questiond():
    epsilon = 0.1
    nep_values = [10, 100, 1000, 10000, 50000, 100000]
    num_tests = 10
    discount = 0.95
    delta = 1 - discount
    alpha_init = 0.9
    alpha_final = 0.01
    test_episodes = 1000  # Episodes for testing
    num_signals_values = [2, 3, 5]  # N values to test with the "maze" grid

    # Placeholder for results
    results = {num_signals: [] for num_signals in num_signals_values}

    for num_signals in num_signals_values:
        avg_rewards = []
        std_devs = []
        for nep in nep_values:
            test_rewards = []
            for test in range(num_tests):
                # Initialize agents
                sender = Sender(num_signals, 5, 5, alpha_init, alpha_final, nep, epsilon, discount)
                receiver = Receiver(num_signals, 5, 5, alpha_init, alpha_final, nep, epsilon, discount)
                
                grid = get_grid('maze')  # Use the "maze" grid
                learn_rewards = run_episodes(sender, receiver, grid, nep, delta)
                
                # Set epsilon to 0 for testing
                sender.epsilon = 0.0
                receiver.epsilon = 0.0
                
                test_rewards.extend(run_episodes(sender, receiver, grid, test_episodes, delta))
            
            # Compute average and standard deviation of test rewards
            avg_reward = np.mean(test_rewards)
            std_dev = np.std(test_rewards)
            avg_rewards.append(avg_reward)
            std_devs.append(std_dev)
        
        results[num_signals] = (avg_rewards, std_devs)

    # Plotting
    for num_signals, (avg_rewards, std_devs) in results.items():
        plt.errorbar(np.log10(nep_values), avg_rewards, yerr=std_devs, label=f'N={num_signals}', fmt='-o', capsize=5)

    plt.xlabel('log(Nep)')
    plt.ylabel('Average Discounted Reward')
    plt.title('Average Discounted Reward vs. log(Nep) for different N values on "Maze" Grid')
    plt.legend()
    plt.show()
        
def questione():
    epsilon = 0.1
    nep_values = [10, 100, 1000, 10000, 50000, 100000]
    num_tests = 10
    discount = 0.95
    delta = 1 - discount
    alpha_init = 0.9
    alpha_final = 0.01
    test_episodes = 1000  # Episodes for testing
    num_signals = 1  # Only one symbol/message

    avg_rewards = []
    std_devs = []
    for nep in nep_values:
        test_rewards = []
        for test in range(num_tests):
            # Initialize agents for the "empty" grid with N = 1
            sender = Sender(num_signals, 5, 5, alpha_init, alpha_final, nep, epsilon, discount)
            receiver = Receiver(num_signals, 5, 5, alpha_init, alpha_final, nep, epsilon, discount)

            grid = get_grid('empty')  # Use the "empty" grid
            learn_rewards = run_episodes(sender, receiver, grid, nep, delta)

            # Set epsilon to 0 for testing
            sender.epsilon = 0.0
            receiver.epsilon = 0.0

            test_rewards.extend(run_episodes(sender, receiver, grid, test_episodes, delta))

        # Compute average and standard deviation of test rewards
        avg_reward = np.mean(test_rewards)
        std_dev = np.std(test_rewards)
        avg_rewards.append(avg_reward)
        std_devs.append(std_dev)

    results = (avg_rewards, std_devs)

    # Plotting
    plt.errorbar(np.log10(nep_values), avg_rewards, yerr=std_devs, label=f'N={num_signals}', fmt='-o', capsize=5)
    plt.xlabel('log(Nep)')
    plt.ylabel('Average Discounted Reward')
    plt.title('Average Discounted Reward vs. log(Nep) for N=1 on "Empty" Grid')
    plt.legend()
    plt.show()
    
def questiontest():
    num_learn_episodes = 100000
    num_test_episodes = 1000
    grid_name = 'fourroom' # 'fourroom', 'maze', or 'empty'
    grid = get_grid(grid_name)
    num_signals = 4
    discount = 0.95
    delta = 1 - discount
    epsilon = 0.1
    alpha_init = 0.9
    alpha_final = 0.01

    # Initialize agents
    sender = Sender(num_signals, len(grid), len(grid[0]), alpha_init, alpha_final, num_learn_episodes, epsilon, discount)
    receiver = Receiver(num_signals, len(grid), len(grid[0]), alpha_init, alpha_final, num_learn_episodes, epsilon, discount)

    # Learn
    learn_rewards = run_episodes(sender, receiver, grid, num_learn_episodes, delta)

    # Test
    sender.epsilon = 0.0
    sender.alpha = 0.0
    sender.alpha_i = 0.0
    sender.alpha_f = 0.0
    receiver.epsilon = 0.0
    receiver.alpha = 0.0
    receiver.alpha_i = 0.0
    receiver.alpha_f = 0.0
    test_rewards = run_episodes(sender, receiver, grid, num_test_episodes, delta)

    # Print results
    print("Average reward during learning: " + str(np.average(learn_rewards)))
    print("Average reward during testing: " + str(np.average(test_rewards)))
    

if __name__ == "__main__":
    # You will need to edit this section to produce the plots and other output required for hand-in

    # Define parameters here
    #questionb()
    #questionc()
    #questiond()
    questione()
    # questiontest()
    
    
    
    """ accurate_corrected_steps = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24
    ]

    # Calculate discounted reward for each possible prize location with the accurate path
    accurate_discounted_rewards = [0.95 ** step for step in accurate_corrected_steps]

    # Calculate the average discounted reward with the accurate path
    accurate_average_discounted_reward = sum(accurate_discounted_rewards) / len(accurate_discounted_rewards)
    print(f"==>> accurate_average_discounted_reward: {accurate_average_discounted_reward}") """