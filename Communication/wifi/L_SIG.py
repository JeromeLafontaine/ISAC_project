import numpy as np

def L_SIG(B : int , data_carrier : np.ndarray , pilot_carrier : np.ndarray , pilot_value : int , unused_carrier : np.ndarray, rate : int , length : int ) -> np.ndarray:
    Rate = np.array([int(x) for x in list('{0:04b}'.format(rate))])
    R = [0]
    #create array for the length from the LSB to the MSB
    length = np.array([int(x) for x in list('{0:08b}'.format(length))])
    length = np.flip(length)
    tail_sig = [0,0,0,0,0,0]
    Parity = [1]
    
    L_SIG_sequence = np.concatenate([Rate,R,length,tail_sig,Parity])
    return L_SIG_sequence


