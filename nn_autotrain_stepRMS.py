
import numpy as np
import matplotlib.pyplot as plt
import math

# To Train & Save the Data
from nn_autotrain_base import *

success = np.empty((0,1))
fail = np.empty((0,1))

w1 = np.array([[-5.60730243, -7.34356735],
                    [6.4687279, -3.73997026],
                    [-7.34536971, 4.043939],
                    [4.72441624, -8.42228823]])
w2 = np.array([[-20.27903413, -7.16617662, 5.4589956, 12.49345734]])


w1_f = np.array([[-5.5695976,  -7.43578953],
 [ 6.70804928, -3.87688996],
 [-7.25226627,  4.02258366],
 [ 4.72933506, -8.40240076]])
w2_f = np.array([[-20.24311041,  -7.14527508,   5.43220685,  12.4713398 ]])

success = main('2', 'XNOR', (w1, w2), 0.37, 20000)[1]
fail = main('2', 'XNOR', (w1_f, w2_f), 0.37, 20000)[1]

'''
#finding close average
worked_s=True
while(worked_s):
    w1 = w1_s
    w2 = w2_s
    w1_s = (w1_s+w1_f)/2
    w2_s = (w2_s+w2_f)/2
    worked_s, success = main('2', 'XNOR', (w1, w2))
    print(success[2][0])
    print(f'\n\n\n{w1_s}\n{w2_s}\n\n\n\n')
    
'''
np.savez('Trained_CloseRMS.npz', success=success, fail=fail)


data = np.load('Trained_CloseRMS.npz', allow_pickle=True)
success = data['success']
fail = data['fail']
print(fail)

RMS_steps = np.empty([0,1])
fail_RMS_steps = np.empty([0,1])
success_RMS_steps = np.empty([0,1])

def calc_RMS(storage_arr, val1, val2):
    return np.vstack([storage_arr, math.sqrt(np.sum(np.square(np.subtract(val1, val2))))/math.sqrt(len(np.squeeze(val1)[0])*len(np.squeeze(val1)[1]))])


for i in range(len(fail[5][0])-1):
    fail_RMS_steps = calc_RMS(fail_RMS_steps, fail[5][0][i+1], fail[5][0][i])

for i in range(len(success[5][0])-1):
    success_RMS_steps = calc_RMS(success_RMS_steps, success[5][0][i+1], success[5][0][i])

for step in range(len(success[5][0])): 
    RMS_steps = calc_RMS(RMS_steps, success[5][0][step], fail[5][0][step])
    
print(f'\n{RMS_steps}, \n\n\n{success_RMS_steps}\n\n\n{fail_RMS_steps}')
np.savez('Calculated_RMS_steps.npz', RMS_steps = RMS_steps, fail_RMS_steps = fail_RMS_steps, success_RMS_steps = success_RMS_steps)

RMS_data = np.load('Calculated_RMS_steps.npz')
steps = RMS_data['RMS_steps']
fail = RMS_data['fail_RMS_steps']
success = RMS_data['success_RMS_steps']

print(f'Both Comparison: {steps}\nFail Steps: {fail}\nSuccess Steps: {success}')

# graphing results
fig, axs = plt.subplots(1,2, figsize=(20, 10))

axs[0].plot(steps)
axs[0].set_title('Compared Step Distance Between Success and Failure')
axs[1].plot(success, color='maroon', label='Distance between Success Steps')
axs[1].plot(fail, color='teal', label='Distance between Fail Steps')
axs[1].set_title('Fail Step Distance and Success Step Distance')
axs[1].legend()
plt.show()
