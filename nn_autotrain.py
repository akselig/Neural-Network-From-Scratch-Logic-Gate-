from nn_autotrain_base import * 
import os.path
import time
import winsound

worked = True
gate = ['AND','OR','NAND','NOR','XOR','XNOR','All_True','All_False']
today = np.datetime64('today', 'D') 
hidden_layer = 4   

if (os.path.isfile(f'{today}_Trained_All_{hidden_layer}.npz')): 
    print("Loading Old File...")
    data = np.load(f'{today}_Trained_All_{hidden_layer}.npz', allow_pickle=True)

else: 
    print("Creating New File...")
    np.savez(f'{today}_Trained_All_{hidden_layer}.npz', total_success = np.empty((0,7,1)), total_fail = np.empty((0,7,1)))
    data = np.load(f'{today}_Trained_All_{hidden_layer}.npz', allow_pickle=True)

total_success = data['total_success']
total_fail = data['total_fail']

fail = np.empty((0,1))
success = np.empty((0,1))

for i in range(100): #Training until finds 100 instances where neural network fails to learn intitial random starting weights
    success_measure = np.array([True])
    while(success_measure.all()):
        params = initialize_params(2, hidden_layer, 1) 
        block = np.empty((0,7,1))

        for i in range(8): 
            worked, newitem = main('2', gate[i], params, 0.37, 10000, hidden_layer)
            print(np.array([newitem]))
            success_measure = np.append(success_measure, worked)
            block = np.vstack([block, np.array([newitem])])

        if success_measure.all(): 
            success = np.vstack([success, np.array([newitem[2]])])
            total_success = np.vstack([total_success, block])
        else: 
            fail = np.vstack([fail, np.array([newitem[2]])])
            total_fail = np.vstack([total_fail, block])
        
        np.savez(f'{today}_Trained_All_{hidden_layer}.npz', total_success=total_success, total_fail=total_fail)
        
    print(f'Successes: {total_success}, Failures:{total_fail}')
    print(len(total_success/8))
    print(len(total_fail/8))



np.savez(f'{today}_Trained_All_{hidden_layer}.npz', total_success=total_success, total_fail=total_fail)
#np.save('Trained_Success.npy', success)
#np.save('Trained_Fail.npy', fail)

# Play the system sound for an error (10x) once complete
#myiter=1
#while(myiter<=10):
#    myiter+=1
#    winsound.PlaySound("SystemExclamation", winsound.SND_ALIAS)
#    time.sleep(120)