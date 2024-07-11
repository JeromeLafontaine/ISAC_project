#%matplotlib qt
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from time import time
import math
#from scipy.special import erfc
#import scipy.interpolate
import scipy
from scipy.interpolate import interp1d
#from scipy import signal


sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from Communication.mapping import bin_to_mod
from Communication.mapping import plot_constellation
from Communication.mapping import mod_to_bin

sys.path.append(os.path.join(os.path.dirname(__file__), 'wifi'))

from Communication.wifi.L_STF import L_STF
from Communication.wifi.L_LTF import L_LTF
from Communication.wifi.L_SIG import L_SIG

from Communication.channel import add_signals
from Communication.channel import SISO_channel

# from Communication.Synchronization import FFT_CFO_estimation
#from Communication.Synchronization import Packet_detector
from Communication.generate_preamble import generate_preamble


testing = False

#----------------------------------------------------------------------------------------#
#                                     Transmission                                       #
#----------------------------------------------------------------------------------------#


def s_to_p(data : np.ndarray, N_data_carrier : int, unused_carrier = []) -> np.ndarray:
    """
    Serial to parallel conversion
    N = number of subcarriers allocated for the data
    example:
    X_S=  [1 2 3 4 5 6 7 8 9]
    X_P=  [[1. 5. 9.]
            [2. 6. 0.]
            [3. 7. 0.]
            [4. 8. 0.]]
    """
    number_of_OFDM_symbols = int(np.ceil(len(data)/N_data_carrier))
    data_pad = np.zeros(number_of_OFDM_symbols*N_data_carrier, dtype=complex)
    data_pad[:len(data)] = data.reshape(-1,1).T[0]
    s2p = np.reshape(data_pad, (N_data_carrier, -1), order='F')

    for i in range(len(unused_carrier)):
        s2p = np.insert(s2p, unused_carrier[i], 0, axis=0)
    return s2p

def OFDM_symbol(data : np.ndarray, pilot : int, data_carrier : np.ndarray, pilot_carrier : np.ndarray, N : int , pilot_freq = 1 , Pilote_rate = 0, all_carrier = 0, unused_carrier = []) -> np.ndarray:
    """
    Creation of 1 OFDM symbol from the data and pilot
    data = data to be transmitted
    pilot = pilot to be transmitted
    data_carrier = indices of the data carriers
    pilot_carrier = indices of the pilot carriers
    N = number of subcarriers
    pilot_freq = 1 if the pilots are in every subcarrier contained in pilot_carrier, 0 if they are spaced by Pilote_rate and present over a whole OFDM symbol

    for example:
    data =      [[1. 5. 9.]
                [2. 6. 0.]
                [3. 7. 0.]
                [4. 8. 0.]]
    pilot = 0.5
    N = 6
    data_carrier = np.array([0,1,2,3])
    pilot_carrier = np.array([4,5])
    OFDM symbol =  [[1. +0.j 5. +0.j 9. +0.j]
                    [2. +0.j 6. +0.j 0. +0.j]
                    [3. +0.j 7. +0.j 0. +0.j]
                    [4. +0.j 8. +0.j 0. +0.j]
                    [0.5+0.j 0.5+0.j 0.5+0.j]
                    [0.5+0.j 0.5+0.j 0.5+0.j]]
    
                    where each column is a OFDM symbol (before IFFT) and each row is a subcarrier
    """
    symbol = np.zeros((N,len(data.T)), dtype=complex)

    if(pilot_freq == 1):
        if(len(data_carrier!= 0)):
            symbol[data_carrier.astype(int)] = data
        if(len(pilot_carrier) != 0):
            pilot = np.tile(pilot, (len(data.T), 1)).T
            symbol[pilot_carrier.astype(int)] = pilot
        return symbol
    
    if(pilot_freq == 0):
        symbol_Pilots = np.zeros((N), dtype=complex)
        symbol = np.zeros((len(data.T), len(data)), dtype=complex)
        symbol = data.T
        symbol_Pilots[pilot_carrier] = pilot 
        time_values_for_pilots = np.arange(0, len(data.T)*Pilote_rate, Pilote_rate)
        #if values in time_values_for_pilots are above the number of columns of symbol, we remove them
        time_values_for_pilots = time_values_for_pilots[time_values_for_pilots < len(data.T)*2]
        Pos = 0
        not_the_end = 1
        i = 0
        while(not_the_end == 1):
            if(i == 0 or i%(Pilote_rate) == 0):
                position = time_values_for_pilots[i%(Pilote_rate) + Pos]
                symbol = np.insert(symbol, position, symbol_Pilots, axis=0)
                if(time_values_for_pilots[i%(Pilote_rate) + Pos] + Pilote_rate >= len(symbol)):
                    not_the_end = 0
                    return symbol
                Pos += 1                
            i += 1
    return symbol


def IFFT(OFDM_symbols : np.ndarray , Oversampling_factor : int) -> np.ndarray:
    """
    IFFT
    example:
    OFDM_symbols = [[1. +0.j 5. +0.j 9. +0.j]
                    [2. +0.j 6. +0.j 0. +0.j]
                    [3. +0.j 7. +0.j 0. +0.j]
                    [4. +0.j 8. +0.j 0. +0.j]
                    [0.5+0.j 0.5+0.j 0.5+0.j]
                    [0.5+0.j 0.5+0.j 0.5+0.j]]
    IFFT =  [[ 1.83333333+0.00000000e+00j  4.5       +0.00000000e+00j  1.66666667+0.00000000e+00j]
            [-0.58333333+5.77350269e-01j -0.58333333+1.73205081e+00j   1.5       -1.44337567e-01j]
            [ 0.33333333-1.44337567e-01j  1.        -1.44337567e-01j   1.41666667+0.00000000e+00j]
            [-0.33333333-7.40148683e-17j -0.33333333-1.48029737e-16j   1.5       +9.25185854e-18j]
            [ 0.33333333+1.44337567e-01j  1.        +1.44337567e-01j   1.41666667+0.00000000e+00j]
            [-0.58333333-5.77350269e-01j -0.58333333-1.73205081e+00j   1.5       +1.44337567e-01j]]
    where each column is a OFDM symbol in time domain
    """
    # ifft_not_centered = np.fft.ifft(OFDM_symbols, n = len(OFDM_symbols)*Oversampling_factor , axis=0, norm="ortho")
    # ifft_centered = np.zeros((len(ifft_not_centered), len(ifft_not_centered.T)), dtype=complex)
    # ifft_centered[:len(ifft_not_centered)//2] = ifft_not_centered[len(ifft_not_centered)//2:]
    # ifft_centered[len(ifft_not_centered)//2:] = ifft_not_centered[:len(ifft_not_centered)//2]
    # return ifft_centered
    return np.fft.ifft(OFDM_symbols, n = len(OFDM_symbols)*Oversampling_factor , axis=0, norm="ortho")


def add_cp(OFDM_symbols_time : np.ndarray, cp : int , Oversampling_factor : int) -> np.ndarray:
    """
    Add cyclic prefix
    example:
    OFDM_symbols_time =  [[ 1.83333333+0.00000000e+00j  4.5       +0.00000000e+00j  1.66666667+0.00000000e+00j]
            [-0.58333333+5.77350269e-01j -0.58333333+1.73205081e+00j   1.5       -1.44337567e-01j]
            [ 0.33333333-1.44337567e-01j  1.        -1.44337567e-01j   1.41666667+0.00000000e+00j]
            [-0.33333333-7.40148683e-17j -0.33333333-1.48029737e-16j   1.5       +9.25185854e-18j]
            [ 0.33333333+1.44337567e-01j  1.        +1.44337567e-01j   1.41666667+0.00000000e+00j]
            [-0.58333333-5.77350269e-01j -0.58333333-1.73205081e+00j   1.5       +1.44337567e-01j]]
    cp = 2
    result = [[ 0.33333333+1.44337567e-01j  1.        +1.44337567e-01j1.41666667+0.00000000e+00j]
            [-0.58333333-5.77350269e-01j -0.58333333-1.73205081e+00j1.5       +1.44337567e-01j]
            [ 1.83333333+0.00000000e+00j  4.5       +0.00000000e+00j1.66666667+0.00000000e+00j]
            [-0.58333333+5.77350269e-01j -0.58333333+1.73205081e+00j1.5       -1.44337567e-01j]
            [ 0.33333333-1.44337567e-01j  1.        -1.44337567e-01j1.41666667+0.00000000e+00j]
            [-0.33333333-7.40148683e-17j -0.33333333-1.48029737e-16j1.5       +9.25185854e-18j]
            [ 0.33333333+1.44337567e-01j  1.        +1.44337567e-01j1.41666667+0.00000000e+00j]
            [-0.58333333-5.77350269e-01j -0.58333333-1.73205081e+00j1.5       +1.44337567e-01j]]
    """
    cyclic_prefix = OFDM_symbols_time[-cp*Oversampling_factor:]
    return np.concatenate((cyclic_prefix, OFDM_symbols_time), axis=0)

def center_around_0Hz(x : np.ndarray, Oversampling_factor : int, N:int) -> np.ndarray:
    """
    Center the signal around 0Hz
    """
    l = np.tile(np.arange(0, len(x)), (x.shape[1],1)).T
    x *= np.exp(-1j*2*np.pi/(N*Oversampling_factor) * ((N-1)/Oversampling_factor) * l * Oversampling_factor/2)
    return x



#----------------------------------------------------------------------------------------#
#                                      Reception                                         #
#----------------------------------------------------------------------------------------#





def equalize(rx_sig_est : np.ndarray , pilot_carrier : np.ndarray , pilotValue : np.ndarray, L_LTF_est_seq = [0], L_LTF_seq = [0], pilot_freq = 1 , Pilote_rate = 0, All_carriers = 0) -> np.ndarray:
    """DOC STRING GENERATED AFTER THE SUBMISSION DEADLINE WITH THE HELP OF Gemini
    Equalizes a received signal based on pilot carriers.

    This function estimates the channel frequency response using pilot carriers and
    interpolates it across all subcarriers. It then equalizes the received signal by dividing
    it by the estimated channel response.

    Args:
        rx_sig_est (np.ndarray): The received signal to be equalized. 2D array of shape (num_subcarriers, num_samples)
        pilot_carrier (np.ndarray): Indices of the pilot subcarriers.
        pilotValue (np.ndarray): Known values of the pilots (typically from a reference signal).
        L_LTF_est_seq (list, optional):  List containing indices of estimated L-LTF sequences. Defaults to [0].
        L_LTF_seq (list, optional): List containing indices of L-LTF sequences. Defaults to [0].
        pilot_freq (int, optional): Indicates pilot arrangement frequency.
            - 1: Pilots are in every subcarrier.
            - 0: Pilots are spaced by `Pilote_rate`. Defaults to 1.
        Pilote_rate (int, optional): Spacing between pilots (used when `pilot_freq` is 0). Defaults to 0.
        All_carriers (int, optional): Indicates if all carriers should be considered. Defaults to 0.

    Returns:
        np.ndarray: The equalized signal, a complex array with the same shape as `rx_sig_est`.
    """
    if(pilot_freq == 1): 
        equalized_sig = np.zeros((len(rx_sig_est), len(rx_sig_est.T)), dtype=complex)
        H_matrix = np.zeros((len(rx_sig_est.T), len(rx_sig_est)), dtype=complex)
        for i in range(len(rx_sig_est.T)):
            plt.figure(figsize = (10,5))
            pilots_est = Extract_pilots(rx_sig_est.T[i], pilot_carrier)
            H = pilots_est/pilotValue
            plt.scatter(np.real(H), np.imag(H), color = "blue", label = "pilots estimation", marker = "x")
            f = scipy.interpolate.interp1d(pilot_carrier, H, kind='slinear',fill_value='extrapolate')
            values = np.arange(0 , len(rx_sig_est.T[i]), 1)
            H_matrix[i] = f(values)
            plt.scatter(np.real(H_matrix[i]),np.imag(H_matrix[i]), color = "red", label = "interpolation")
        for i in range(len(rx_sig_est.T)):
                equalized_sig.T[i] = rx_sig_est.T[i]/H_matrix
        return equalized_sig
    if(pilot_freq == 0):
        #The first column is made of pilots and every Pilote_rate column after that
        pilot_times = np.arange(0 , rx_sig_est.shape[1], Pilote_rate)
        equalized_sig = np.zeros((rx_sig_est.shape[0], rx_sig_est.shape[1]), dtype=complex)
        H_est_matrix = np.zeros((len(pilot_carrier), len(pilot_times)), dtype=complex)

        Pos = 0 ; not_the_end = 1 ;i = 0
        while(not_the_end == 1):
            if(i == 0 or i%(Pilote_rate) == 0):
                position = pilot_times[i%(Pilote_rate) + Pos]
                pilots_est = rx_sig_est[pilot_carrier,position]
                H_est_matrix.T[i//(Pilote_rate)] = pilots_est/pilotValue
                if(pilot_times[i%(Pilote_rate) + Pos] + Pilote_rate >= rx_sig_est.shape[1]):
                    not_the_end = 0
                Pos += 1                
            i += 1

        f_real = scipy.interpolate.RectBivariateSpline(pilot_carrier,pilot_times, np.real(H_est_matrix), kx=1, ky=1)
        f_imag = scipy.interpolate.RectBivariateSpline(pilot_carrier,pilot_times, np.imag(H_est_matrix), kx=1, ky=1)
    
        times = np.arange(0 , rx_sig_est.shape[1], 1)

        H_matrix_real = f_real(pilot_carrier, times)
        H_matrix_imag = f_imag(pilot_carrier, times)
        H_matrix = H_matrix_real + 1j*H_matrix_imag

        equalized_sig = rx_sig_est[pilot_carrier, :]/H_matrix
        return equalized_sig
        



def DFT(x : np.ndarray, Oversampling_factor :int) -> np.ndarray:
    """
    Discrete Fourier Transform
    """
    return np.fft.fft(x, axis = 0,n = len(x), norm="ortho")[:len(x)//Oversampling_factor]


def remove_cp(x : np.ndarray, cp : int, N : int, oversamping_factor) -> np.ndarray:
    """
    Remove cyclic prefix
    """
    return x[cp*oversamping_factor:(cp+N)*oversamping_factor]

def decenter_around_0Hz(x : np.ndarray, Oversampling_factor : int, N:int) -> np.ndarray:
    """
    Decenter the signal around 0Hz
    """
    l = np.tile(np.arange(0, len(x)), (x.shape[1],1)).T
    x *= np.exp(1j*2*np.pi/(N*Oversampling_factor) * ((N-1)/Oversampling_factor) * l * Oversampling_factor/2)
    return x

def Extract_data(x : np.ndarray, data_carrier : np.ndarray , pilot_freq = 1 , Pilote_rate = 0) -> np.ndarray:
    """
    Extract data from OFDM symbol
    """
    if(pilot_freq == 1):
        data = np.zeros((len(data_carrier), len(x.T)), dtype=complex)
        data = x[data_carrier.astype(int)]
        return data
    if(pilot_freq == 0):
        time_Pilots = np.arange(0 , len(x.T), Pilote_rate)
        data = np.delete(x, time_Pilots.astype(int), axis = 1)
        data = data[data_carrier.astype(int)] 

        return data

def Extract_pilots(x : np.ndarray, pilot_carrier : np.ndarray) -> np.ndarray:
    pilot = np.zeros((len(pilot_carrier), len(x.T)), dtype=complex)
    pilot = x[pilot_carrier.astype(int)]
    return pilot

def p_to_s(x : np.ndarray) -> np.ndarray:
    """
    Parallel to serial conversion
    example : 
    parallele =  [[1 2 3]
                    [4 5 6]
                    [7 8 9]]
    serial =  [1 2 3 4 5 6 7 8 9]
    """
    return np.reshape(x, (-1, 1), order="F").T[0]


#----------------------------------------------------------------------------------------#
#                                        Metrics                                         #
#----------------------------------------------------------------------------------------#

def BER(x : np.ndarray, y : np.ndarray) -> float:
    """
    Bit Error Rate
    """
    count = 0
    for i in range(x.shape[0]):
        if(x[i] != y[i]):
            count += 1
    return count/x.shape[0]

#----------------------------------------------------------------------------------------#
#                                    non idealities                                      #
#----------------------------------------------------------------------------------------#

def CFO(x : np.ndarray, CFO_max : int , fs : int) -> np.ndarray:
    """
    Carrier Frequency Offset, this function will take a random value between -CFO_max and CFO_max and multiply the signal by exp(1j*2*pi*CFO/fs*i)
    """
    #take a random value between -ppm_max and ppm_max the value can be a float
    CFO = np.random.uniform(-CFO_max, CFO_max)

    for i in range(len(x)):
        x[i] *= np.exp(1j*2*np.pi*CFO/fs*i)
    return x

def CFO_Corrector(x : np.ndarray , CFO_est : int , fs : int) -> np.ndarray:
    """
    Correct the CFO, this function will multiply the signal by exp(1j*2*pi*CFO_est/fs*i) to correct the CFO
    """
    for i in range(len(x)):
        x[i] *= np.exp(1j*2*np.pi*CFO_est/fs*i)
    return x

