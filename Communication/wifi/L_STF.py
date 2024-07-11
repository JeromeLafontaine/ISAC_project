import numpy as np


def L_STF(B : int , data_carrier : np.ndarray , pilot_carrier : np.ndarray , pilot_value : int , unused_carrier : np.ndarray) -> np.ndarray:
    """
    L-STF field generation
    The sequence used for the L-STF in wifi 
    Args:
        B (int): Bandwidth
        data_carrier (np.ndarray): Data carriers
        pilot_carrier (np.ndarray): Pilot carriers
        pilot_value (int): Pilot value
        unused_carrier (np.ndarray): Unused carriers

    Returns:
        np.ndarray: L-STF field
    """
    carrier_used_L_STF = np.array([ -24, -20, -16, -12, -8, -4, 4, 8, 12, 16, 20, 24]) + 64/2
    L_STF_sequence = np.array([1+1j , -1-1j ,1+1j , -1-1j ,-1-1j , 1+1j ,-1-1j , -1-1j ,1+1j , 1+1j ,1+1j , 1+1j])
    L_STF = np.zeros(64, dtype=complex)

    L_STF[carrier_used_L_STF.astype(int)] = L_STF_sequence

    L_STF_OFDM_symbols_time = np.fft.ifft(L_STF,64 , norm="ortho")[0:8]

    L_STF_OFDM_symbols_time = np.tile(L_STF_OFDM_symbols_time, 10)

    return np.concatenate([L_STF_OFDM_symbols_time])

