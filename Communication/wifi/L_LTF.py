
import numpy as np


def L_LTF(B : int , data_carrier : np.ndarray , pilot_carrier : np.ndarray , pilot_value : int , unused_carrier : np.ndarray) -> np.ndarray:
    """
    L-LTF field generation
    The sequence used for the L-LTF in wifi 
    Args:
        B (int): Bandwidth
        data_carrier (np.ndarray): Data carriers
        pilot_carrier (np.ndarray): Pilot carriers
        pilot_value (int): Pilot value
        unused_carrier (np.ndarray): Unused carriers

    Returns:
        np.ndarray: L-LTF field
    """
    carrier_used_L_LTF = np.linspace(-26, 26, 53) + 64/2
    carrier_used_L_LTF = np.delete(carrier_used_L_LTF, 26) #remove the DC carrier

    L_LTF_sequence_part1 = [1,1,-1,-1,1,1,-1,1,-1,1,1,1,1,1,1,-1,-1,1,1,-1,1,-1,1,1,1,1]
    L_LTF_sequence_part2 = [1,-1,-1,1,1,-1,1,-1,1,-1,-1,-1,-1,-1,1,1,-1,-1,1,-1,1,-1,1,1,1,1]
    L_LTF_sequence = np.concatenate([L_LTF_sequence_part1,L_LTF_sequence_part2])

    L_LTF = np.zeros(64, dtype=complex)
    L_LTF[carrier_used_L_LTF.astype(int)] = L_LTF_sequence
    L_LTF_OFDM_symbols_time = np.fft.ifft(L_LTF,64 , norm="ortho")

    #create CP with half of the L-LTF
    CP = L_LTF_OFDM_symbols_time[0:32]

    L_LTF_time = np.concatenate([CP,L_LTF_OFDM_symbols_time, L_LTF_OFDM_symbols_time])

    return L_LTF_time

