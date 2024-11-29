import numpy as np
from nn_autotrain_base import *
today = np.datetime64('today', 'D') 


data = np.load(f'2024-08-24_Trained_All_4.npz', allow_pickle=True)

success = data['total_success']
fail = data['total_fail']
print(f'Successful Initial Weights: {len(success)/8}, Failures: {len(fail)/8}')



results = np.load('2024-08-25_Trained_PlusDim_4-5.npz', allow_pickle=True)
f_s = results['fail_success']
f_f = results['fail_fail']
s_s = results['success_success']
s_f = results['success_fail']
print(len(f_f), len(f_s), len(s_f), len(s_s))

print(fail[0], f_s[0])

'''
#finding double gates from fail fail and og fail
matches = np.empty((0,7,1))
for item in f_f: 
    for item_og in fail: 
        if (np.round(item[2][0][:-1], 2) == np.round(item_og[5][0][-1],2)).all() and item[0][0] == item_og[0][0]:
            print("\n\n\nMatch Found.")
            print(item[2][0][:-1], item_og[5][0][-1]) 
            print(item[0][0], item_og[0][0])
            
            matches = np.vstack([matches, [item_og]])

for item in matches:
    print(item[2])
'''

'''

# Separating the Failed Gates from Successes
fail_instance = np.empty((0,7,1))
success_instance = np.empty((0,7,1))

fail_data = fail

for i in range(0, len(fail_data)):
    if not (fail_data[i][6]):
        fail_instance = np.vstack([fail_instance, [fail_data[i]]], dtype=object)
    else: 
        success_instance = np.vstack([success_instance, [fail_data[i]]], dtype=object)

print(fail_instance)

# Adding a Dimension and Retraining Failures and Successes
dimf_success = np.empty((0,7,1))
dimf_fail = np.empty((0,7,1))

for item in range(len(fail_instance)):
    item = fail_instance[item] 
    #add zero row to w1 and 1 row to w2
    updim_w1 = np.zeros((item[3][0].shape[0]+1, item[3][0].shape[1]-1))
    updim_w2 = np.zeros((np.delete(item[3][0], [0,1], 1).T.shape[0], np.delete(item[3][0], [0,1], 1).T.shape[1]+1))
    updim_w1[:-1, ] = np.delete(item[3][0], -1 , 1)
    updim_w2[0, :-1] = np.delete(item[3][0], [0, 1], 1).T
   

    
    #updim_w1[-1, :] = np.array([[0, -0.5]]) #1, 2, 3, 13 = slope
    #updim_w1[-1, :] = np.array([[0, 0.5]]) #4, 5, 6, 7, 11, 12 = saddle
    #updim_w1[-1, :] = np.array([[0.5, 0]]) # conclusion: mixed slope and saddle points
    #updim_w1[-1, :] = np.array([[-0.5, 0]]) 
    
    worked, instance = main('2', item[0], (updim_w1, updim_w2), 0.37, 10000, updim_w1.shape[0])

    if worked: 
        dimf_success = np.vstack([dimf_success, [instance]], dtype=object)
    else:
        dimf_fail = np.vstack([dimf_fail, [instance]], dtype=object)

print(dimf_success)
print()
print(dimf_fail)


dims_success = np.empty((0,7,1))
dims_fail = np.empty((0,7,1))
for item in success_instance: 
    #add zero row to w1 and 1 row to w2
    updim_w1 = np.zeros((item[3][0].shape[0]+1, item[3][0].shape[1]-1))
    updim_w2 = np.zeros((np.delete(item[3][0], [0,1], 1).T.shape[0], np.delete(item[3][0], [0,1], 1).T.shape[1]+1))
    updim_w1[:-1, ] = np.delete(item[3][0], -1 , 1)
    updim_w2[0, :-1] = np.delete(item[3][0], [0,1], 1).T

    worked, instance = main('2', item[0], (updim_w1, updim_w2), 0.37, 10000, updim_w1.shape[0])

    if worked: 
        dims_success = np.vstack([dims_success, [instance]], dtype=object)
    else:
        dims_fail = np.vstack([dims_fail, [instance]], dtype=object)


for item in success: 
    #add zero row to w1 and 1 row to w2
    updim_w1 = np.zeros((item[3][0].shape[0]+1, item[3][0].shape[1]-1))
    updim_w2 = np.zeros((np.delete(item[3][0], [0,1], 1).T.shape[0], np.delete(item[3][0], [0,1], 1).T.shape[1]+1))
    updim_w1[:-1, ] = np.delete(item[3][0], -1 , 1)
    updim_w2[0, :-1] = np.delete(item[3][0], [0,1], 1).T

    worked, instance = main('2', item[0], (updim_w1, updim_w2), 0.37, 10000, updim_w1.shape[0])

    if worked: 
        dims_success = np.vstack([dims_success, [instance]], dtype=object)
    else:
        dims_fail = np.vstack([dims_fail, [instance]], dtype=object)

today = np.datetime64('today', 'D') 
np.savez(f'{today}_Trained_PlusDim_{updim_w1.shape[0]-1}-{updim_w1.shape[0]}.npz', fail_success=dimf_success, fail_fail=dimf_fail, success_success=dims_success, success_fail=dims_fail)
'''
'''
# Calculating the Number of Gates that Fail per Initial Set of Weights
fail_gates = np.empty((0,1))

for i in range(0, len(fail), 8): 
    num_gates = 0
    for x in range(i, i+7):
        if fail[x][-1] == False: 
            num_gates +=1
    fail_gates = np.append(fail_gates, num_gates)

print(fail_gates)
x0, x1, x2, x3, x4, x5, x6, x7, x8 = [0, 0, 0, 0, 0, 0, 0, 0, 0]
for item in fail_gates: 
    if item == 0: x0+=1
    if item == 1: x1+=1
    if item == 2: x2+=1
    if item == 3: x3+=1
    if item == 4: x4+=1
    if item == 5: x5+=1
    if item == 6: x6+=1
    if item == 7: x7+=1
    if item == 8: x8+=1
    
print(f'0 Gates: {x0}, 1 Gate: {x1}, 2 Gates: {x2}, 3 Gates: {x3}, 4 Gates: {x4}, 5 Gates: {x5}, 6 Gates: {x6}, 7 Gates {x7}, 8 gates: {x8}')
'''