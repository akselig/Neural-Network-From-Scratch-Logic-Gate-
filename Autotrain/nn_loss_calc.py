import numpy as np

s_arr = np.load('Trained_Success.npy', allow_pickle=True)
f_arr = np.load('Trained_Fail.npy', allow_pickle=True)

dist_s = np.empty([0,3])
dist_f = np.empty([0,3])

print(len(s_arr))
print(len(f_arr))

for session in s_arr:
    sums = 0
    for index in range(len(session[4][0])-1):
        sums += abs(session[4][0][index]-session[4][0][index+1])
    displacement = abs((session[4][0][-1]-session[4][0][0]))
    dist_s = np.vstack([dist_s, np.array([sums, displacement, round(sums,7)==round(displacement,7)])])

for session in f_arr: 
    sums = 0
    for index in range(len(session[4][0])-1):
        sums += abs(session[4][0][index]-session[4][0][index+1])
    displacement = abs((session[4][0][-1]-session[4][0][0]))
    dist_f = np.vstack([dist_f, np.array([sums, displacement, round(sums,7)==round(displacement,7)])])


np.save('Calculated_Loss_Sums.npy',np.array([[dist_s],[dist_f]], dtype=object))

sf_arr = np.load('Calculated_Loss_Sums.npy', allow_pickle=True)
print(sf_arr)
