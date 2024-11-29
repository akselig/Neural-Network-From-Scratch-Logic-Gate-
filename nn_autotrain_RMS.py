import numpy as np
import matplotlib.pyplot as plt
import math


data = np.load(f'2024-08-25_Trained_PlusDim_4-5.npz', allow_pickle=True)
success_data = data['success_success']


data = np.load(f'2024-08-24_Trained_All_4.npz', allow_pickle=True)
og_data = data['total_success']

success_data = success_data[-len(og_data):]
print(len(success_data), len(og_data))
'''
RMS_weights = np.empty((0,20000))

#print(len(success_data, fail_data))

def RMS_calculation(data1, data2): 
    return math.sqrt(np.square(np.sum(np.subtract(data1, data2))))/(math.sqrt(data1.shape[0]*data1.shape[1]))


for sesh in range(len(success_data)):
    calculated_val = np.empty((0,1))
    for weightindx in range(10000):  
        sesh_dim = og_data[sesh][5][0][weightindx]
        sesh_og = og_data[sesh][2][0]
        calculated_val = np.append(calculated_val, RMS_calculation(sesh_dim, sesh_og))

    for weightindx in range(10000): 
        sesh_dim =  success_data[sesh][5][0][weightindx]
        sesh_og = og_data[sesh][2][0]
        calculated_val = np.append(calculated_val, RMS_calculation(sesh_dim[:-1], sesh_og))


    print(f'{sesh}/{len(success_data)}')
    RMS_weights = np.vstack((RMS_weights, calculated_val), dtype=object) 


print(f'\n{RMS_weights}')
today = np.datetime64('today', 'D') 
np.save(f'{today}_Calculated_RMS_Success_PlusDim_4-5.npy', RMS_weights)
'''
RMS_data = np.load('2024-08-25_Calculated_RMS_Success_PlusDim_4-5.npy', allow_pickle=True)

# graphing results
print(len(RMS_data[0]))
for x in range(len(RMS_data)): 
    item = RMS_data[x]
    print(success_data[x+len(success_data)-len(og_data)])
    print(f'Item: {x}')
    plt.plot(item)
    plt.show()
