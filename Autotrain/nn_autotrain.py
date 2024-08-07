from nn_autotrain_base import * 
import time
import winsound

worked = True
myiter = 0

gate = ['AND','OR','NAND','NOR','XOR','XNOR']
success = np.empty((0,1))
total_success = np.empty((0,5,1))
fail = np.empty((0,1))
total_fail = np.empty((0,5,1))

for i in range(100):
    params = initialize_params(2, 4, 1) 
    block = np.empty((0,5,1))
    success_measure = np.empty((0,1))

    for i in range(6): 
        worked, newitem = main('2', gate[i], params)
        print(np.array([newitem]))
        success_measure = np.append(success_measure, worked)
        block = np.vstack([block, np.array([newitem])])

    if success_measure.all(): 
        success = np.vstack([success, np.array([newitem[2]])])
        total_success = np.vstack([total_success, block])
    else: 
        fail = np.vstack([fail, np.array([newitem[2]])])
        total_fail = np.vstack([total_fail, block])
    
        
print(f'{success}\n\n{fail}')
np.save('Trained_Success.npy', success)
np.save('Trained_Fail.npy', fail)
np.savez('Trained_All.npz', total_success=total_success, total_fail=total_fail)


# Play the system sound for an error (10x) once fails
#myiter=1
#while(myiter<=10):
#    myiter+=1
#    winsound.PlaySound("SystemExclamation", winsound.SND_ALIAS)
#    time.sleep(120)
