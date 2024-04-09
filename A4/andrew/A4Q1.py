import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from multiprocessing import Pool, freeze_support

S = np.array([0,1])
F = np.array([0,1])
D = np.array([0,1])
TG = np.array([0,1])
DS = np.array([0,1,2])

prob_s_true = np.array([
    [0.05, 0.02],  # DS = None, TG = [Not Present, Present]
    [0.65, 0.1],   # DS = Mild, TG = [Not Present, Present]
    [0.8, 0.2]     # DS = Severe, TG = [Not Present, Present]
])
prob_f_true = np.array([0.1, 0.6, 0.3]) # ds = none, mild, severe
prob_d_true = np.array([0.1, 0.3, 0.6]) # ds = none, mild, severe
# Calculating P(S = False | Conditions) as 1 - P(S = True | Conditions)
prob_false = 1 - prob_s_true
prob_f_false = 1 - prob_f_true
prob_d_false = 1 - prob_d_true

S_given_DSTG = np.dstack((prob_false, prob_s_true)) # [ds][tg][S]
# [ds][f or d -> false true]
F_given_DS = np.column_stack((prob_f_false,prob_f_true))  
D_given_DS = np.column_stack((prob_d_false,prob_d_true))

TG_probs = np.array([0.9, 0.1]) # not present, present
DS_probs = np.array([0.5, 0.25, 0.25])  # None, Mild, Severe


def add_noise_and_normalize(probabilities, delta, seed):
    np.random.seed(seed)
    noise = np.random.uniform(0, delta, size=probabilities.shape)
    noisy_probabilities = probabilities + noise
    normalized_probabilities = noisy_probabilities / noisy_probabilities.sum(axis=-1, keepdims=True)
    return normalized_probabilities

def read_train():
        columns = ['S', 'F', 'D', 'TG', 'DS']
        dataset = pd.read_csv('dataset/traindata.txt', sep=" ", names=columns)
        dataset.reset_index(drop=True, inplace=True)
        return dataset
def read_test():
        columns = ['S', 'F', 'D', 'TG', 'DS']
        dataset = pd.read_csv('dataset/testdata.txt', sep=" ", names=columns)
        dataset.reset_index(drop=True, inplace=True)
        return dataset
testdata = read_test()
def find_acc(df_sort):
    correct = 0
    for index,row in testdata.iterrows():
        filtered_df = df_sort[(df_sort['S'] == row['S']) & (df_sort['F'] == row['F']) & (df_sort['D'] == row['D']) & (df_sort['TG'] == row['TG'])]
        predict = df_sort.loc[filtered_df['P(DS | S,F,D,TG)'].idxmax(), 'DS']
        if predict == row['DS']:
            correct += 1
    return correct / testdata.shape[0]

def em(data):
    delta = data[0]
    seed = data[1]
    S_given_DSTG_N = add_noise_and_normalize(S_given_DSTG,delta, seed)
    F_given_DS_N = add_noise_and_normalize(F_given_DS,delta, seed)
    D_given_DS_N = add_noise_and_normalize(D_given_DS,delta, seed)
    TG_probs_N = add_noise_and_normalize(TG_probs,delta, seed)
    DS_probs_N = add_noise_and_normalize(DS_probs,delta, seed)

    def logic_S(row):
        return S_given_DSTG_N[int(row['DS'])][int(row['TG'])][int(row['S'])]

    def logic_F(row):
        return F_given_DS_N[int(row['DS'])][int(row['F'])]

    def logic_D(row):
        return D_given_DS_N[int(row['DS'])][int(row['D'])]

    def logic_TG(row):
        return TG_probs_N[int(row['TG'])]
        
    def logic_DS(row):
        return DS_probs_N[int(row['DS'])]

    mesh = np.array(np.meshgrid(S,F,D,TG,DS)).T.reshape(-1, 5)

    def alland(row):
        return row["P(S | DS,TG)"] * row["P(F | DS)"]* row["P(D | DS)"] * row["P(TG)"]* row["P(DS)"]
    
    result = float('inf')
    traindata = read_train()
    iter = 0
    while True:
        print(f"==>> iteration: {iter}")
        iter += 1
        df_sort = pd.DataFrame(mesh[np.lexsort(np.rot90(mesh))], columns = [ "S", "F", "D", "TG", "DS"])
        df_sort["P(S | DS,TG)"] = df_sort.apply(logic_S, axis=1)
        df_sort["P(F | DS)"] = df_sort.apply(logic_F, axis=1)
        df_sort["P(D | DS)"] = df_sort.apply(logic_D, axis=1)
        df_sort["P(TG)"] = df_sort.apply(logic_TG, axis=1)
        df_sort["P(DS)"] = df_sort.apply(logic_DS, axis=1)
        df_sort["P(S,F,D,TG,DS)"] = df_sort.apply(alland, axis=1)
        normalize = df_sort["P(S,F,D,TG,DS)"].rolling(3).sum().shift(-2)[::3].to_numpy()
        def logic_Norm(row):
            return row["P(S,F,D,TG,DS)"] / normalize[row.name // 3]
            
        df_sort["P(DS | S,F,D,TG)"] = df_sort.apply(logic_Norm, axis=1)
        if iter == 1:
            accuracy_before = find_acc(df_sort)
        data_table = pd.DataFrame(columns = ['S', 'F', 'D', 'TG', 'DS', 'P(S,F,D,TG,DS)', 'P(DS | S,F,D,TG)'])
        for index,row in traindata.iterrows():
            if row['DS'] != -1:
                row['P(S,F,D,TG,DS)'] = 1
                row['P(DS | S,F,D,TG)'] = 1
                data_table.loc[len(data_table)] = row
                for val in [n for n in [0,1,2] if n != row['DS']]:
                    new_row = row.copy()
                    new_row['DS'] = val
                    new_row['P(S,F,D,TG,DS)'] = 0
                    new_row['P(DS | S,F,D,TG)'] = 0
                    data_table.loc[len(data_table)] = new_row
            else:
                for val in [0,1,2]:
                    new_row = row.copy()
                    new_row['DS'] = val
                    matched_row = df_sort[(df_sort['S'] == new_row['S']) & (df_sort['F'] == new_row['F']) & (df_sort['D'] == new_row['D']) & (df_sort['TG'] == new_row['TG']) & (df_sort['DS'] == new_row['DS'])]
                    new_row['P(S,F,D,TG,DS)'] = matched_row['P(S,F,D,TG,DS)'].iloc[0]
                    new_row['P(DS | S,F,D,TG)'] = matched_row['P(DS | S,F,D,TG)'].iloc[0]
                    data_table.loc[len(data_table)] = new_row
        newresult = data_table['P(S,F,D,TG,DS)'].sum()
        print(f"==>> newresult: {newresult}")
        print(f"==>> result: {result}")
        if abs(newresult-result) <= 0.01:
            break
        else:
            result = newresult
        # update DS
        for i in range(3):
            DS_probs_N[i] = data_table.loc[data_table['DS'] == i, 'P(DS | S,F,D,TG)'].sum() / data_table['P(DS | S,F,D,TG)'].sum()
        # update TG
        for i in range(2):
            TG_probs_N[i] = data_table.loc[data_table['TG'] == i, 'P(DS | S,F,D,TG)'].sum() / data_table['P(DS | S,F,D,TG)'].sum()
        # update D
        for i in range(3):
            for j in range(2):
                D_given_DS_N[i][j] = data_table.loc[(data_table['D'] == j)& (data_table['DS'] == i), 'P(DS | S,F,D,TG)'].sum() / data_table.loc[data_table['DS'] == i, 'P(DS | S,F,D,TG)'].sum()
        for i in range(3):
            for j in range(2):
                F_given_DS_N[i][j] = data_table.loc[(data_table['F'] == j)& (data_table['DS'] == i), 'P(DS | S,F,D,TG)'].sum() / data_table.loc[data_table['DS'] == i, 'P(DS | S,F,D,TG)'].sum()
        for i in range(3):
            for j in range(2):
                for k in range(2):
                    S_given_DSTG_N[i][j][k] = data_table.loc[(data_table['S'] == k)&(data_table['TG'] == j)&(data_table['DS'] == i), 'P(DS | S,F,D,TG)'].sum() / data_table.loc[(data_table['TG'] == j)& (data_table['DS'] == i), 'P(DS | S,F,D,TG)'].sum()
    # end while
    accuracy_after = find_acc(df_sort)
    return accuracy_before, accuracy_after

if __name__ == '__main__':
    freeze_support()
    # Main experiment loop
    delta_values = np.linspace(0, 4, 5, endpoint=False)
    num_trials = 3
    args_list = [(delta_values[delta_idx], trial_idx) for delta_idx in range(len(delta_values)) for trial_idx in range(num_trials)]
    # Use multiprocessing Pool to apply function in parallel
    with Pool(processes=1) as pool:  # Adjust 'processes' based on your CPU; too high can decrease efficiency
        results = pool.map(em, args_list)
    acc_before, acc_after = zip(*results)
    # Convert results back to a 2D array
    acc_before_grid = np.array(acc_before).reshape(len(delta_values), num_trials)
    acc_after_grid = np.array(acc_after).reshape(len(delta_values), num_trials)
    print(acc_before_grid)
    arr_before_mean = np.mean(acc_before_grid,axis=1)
    arr_before_std = np.std(acc_before_grid,axis=1)
    acc_after_mean = np.mean(acc_after_grid,axis=1)
    acc_after_std = np.std(acc_after_grid,axis=1)
    plt.errorbar(delta_values, arr_before_mean, yerr=arr_before_std, label='before', fmt='-o', capsize=5)
    plt.errorbar(delta_values, acc_after_mean, yerr=acc_after_std, label='after', fmt='-x', capsize=5)

    plt.xlabel('Delta')
    plt.ylabel('Metrics')
    plt.title('Prediction Metrics vs. Delta')
    plt.legend()
    plt.grid(True)
    plt.show()