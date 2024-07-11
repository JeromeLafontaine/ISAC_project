#DOC STRING GENERATED AFTER THE SUBMISSION DEADLINE WITH THE HELP OF Gemini

import matplotlib.pyplot as plt
from time import time
import numpy as np
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from Communication.channel import SISO_channel


import matplotlib.pyplot as plt


from Communication.OFDM import bin_to_mod
from Communication.OFDM import s_to_p
from Communication.OFDM import IFFT
from Communication.OFDM import add_cp
from Communication.OFDM import OFDM_symbol
from Communication.OFDM import remove_cp
from Communication.OFDM import DFT
from Communication.OFDM import equalize
from Communication.OFDM import Extract_data
from Communication.OFDM import plot_constellation
from Communication.OFDM import BER
from Communication.OFDM import mod_to_bin
from Communication.OFDM import p_to_s
from Communication.OFDM import CFO_Corrector
from Communication.OFDM import center_around_0Hz
from Communication.OFDM import decenter_around_0Hz



sys.path.append(os.path.join(os.path.dirname(__file__), 'wifi'))
from Communication.wifi.L_LTF import L_LTF

from Communication.generate_preamble import generate_preamble







def FFT_CFO_estimation(rx_sig: np.ndarray, N: int, T: float, N_cp: int, M: int) -> float:
    """Estimates the Carrier Frequency Offset (CFO) using FFT-based method.
    ATTENTION: Need to add the pilot sequence over the 1 subcarrier of the OFDM symbols !!! (it is the first subcarrier of the OFDM symbols that has been selected)

    This function calculates the CFO from a received OFDM signal (`rx_sig`) by leveraging the 
    phase rotation between consecutive OFDM symbols. It performs the following steps:

    1. Reshapes the received signal into parallel streams.
    2. Removes cyclic prefix (CP) from the signal.
    3. Recenters the signal around 0 Hz to isolate the CFO's effect.
    4. Performs a Discrete Fourier Transform (DFT) on one OFDM symbol.
    5. Applies zero-padding to increase frequency resolution.
    6. Computes the FFT of the product of the signal and its complex conjugate.
    7. Determines the CFO estimate as the frequency corresponding to the peak magnitude in the FFT.

    Args:
        rx_sig (np.ndarray): Received OFDM signal as a 1D complex array.
        N (int): Number of subcarriers per OFDM symbol.
        T (float): Sampling period.
        N_cp (int): Length of the cyclic prefix.
        M (int): Number of OFDM symbols.

    Returns:
        float: Estimated CFO in Hz.
    """

    Zero_padding_factor = 100

    rx_sig_parallel = s_to_p(rx_sig, (N + N_cp)*M)
    P = rx_sig_parallel.shape[1] #the number of pulses
    rx_sig_no_CP = remove_cp(rx_sig_parallel, N_cp, N, M)
    rx_sig_no_CP_decentered = decenter_around_0Hz(rx_sig_no_CP, M, N)
    rx_sig_est = DFT(rx_sig_no_CP_decentered, M)
    rx_sig_est = rx_sig_est[0,:] #take only the first OFDM symbol
    fft_len = P*(Zero_padding_factor+1)
    fft = np.fft.fftshift( np.fft.fft( (rx_sig_est * np.conj(1+1j)) / (abs(1+1j)**2) , n = fft_len) )
    T_PRI = T * (N + N_cp) * M
    f = np.linspace(-1/(2*T_PRI), 1/(2*T_PRI), len(fft))
    Highest_peak_arg = np.argmax(abs(fft))
    CFO_est = f[Highest_peak_arg]

    return CFO_est

def Moose_2_CFO_estimation(rx_sig: np.ndarray, N: int, T: float, N_cp: int, M: int) -> float:
    """Estimates Carrier Frequency Offset (CFO) using the MOOSE 2 algorithm.

    This function implements the Modified Overlap-Save (MOOSE 2) algorithm to estimate
    the CFO in an OFDM signal. It works by calculating the phase difference between 
    overlapping segments of the signal, which is indicative of the CFO. The specific
    implementation assumes a repeating sequence within the OFDM signal.

    ATTENTION : The CFO is calculated over the first subcarrier of the OFDM symbols => Needs to be changed if the CFO is not on the first subcarrier of the OFDM symbols 
                                                                                       Needs to be changed if the length of the repeating sequence changes
                                                                                       The sequence used must be added on the first subcarrier !! 

    Args:
        rx_sig (np.ndarray): The received OFDM signal as a 1D complex array.
        N (int): Number of subcarriers per OFDM symbol.
        T (float): Sampling period.
        N_cp (int): Length of the cyclic prefix.
        M (int): Number of OFDM symbols.

    Returns:
        float: The estimated CFO in Hz.
    """
    rx_sig_parallel = s_to_p(rx_sig, (N + N_cp)*M)
    P = rx_sig_parallel.shape[1] #the number of pulses
    rx_sig_no_CP = remove_cp(rx_sig_parallel, N_cp, N, M)
    rx_sig_no_CP_decentered = decenter_around_0Hz(rx_sig_no_CP, M, N)
    rx_sig_est = DFT(rx_sig_no_CP_decentered, M)
    rx_sig_est = rx_sig_est[0,:] 
    y = rx_sig_est
    y_shifted = np.zeros(len(rx_sig_est)+ N//2, dtype=np.complex64)
    y_shifted[N//2:] = rx_sig_est
    P = np.zeros(len(y), dtype=np.complex64)
    R1 = np.zeros(len(y), dtype=np.complex64)
    R2 = np.zeros(len(y), dtype=np.complex64)

    for n in range(len(P)):
        for m in range(N): #on itère sur tout les valeurs de la taille de notre sequence de répétition
            if(n+m < len(y)):
                P[n] += np.conj(y[n+m])*y_shifted[n+m]
                R1[n] += np.abs(y[n+m])**2
                R2[n] += np.abs(y_shifted[n+m])**2

    Delta_f = np.angle(P[64]) / (np.pi * (T * (N + N_cp))*N) #during the test we look at the 64th value of the P vector, this needs to be changed if the length of the repetition sequence changes (64=length of 1 instance of the repeating sequence)
    return Delta_f
    




def Schmidl_and_Cox(y : np.ndarray , N : int , T : int):
    """Implements the Schmidl and Cox algorithm for timing and frequency synchronization.

    This function applies the Schmidl and Cox algorithm to a received OFDM signal (`y`). It 
    estimates the starting point of the frame (`n_hat`) and the carrier frequency offset
    (`Delta_f`) based on the correlation between a known repeating sequence in the signal 
    and a shifted version of itself.

    Args:
        y (np.ndarray): The received OFDM signal as a 1D complex array.
        N (int): Length of the known repeating sequence within the OFDM symbol.
        T (float): Sampling period of the signal.

    Returns:
        tuple: A tuple containing three values:
            - n_hat (int): Estimated start index of the frame within the signal.
            - Delta_f (float): Estimated carrier frequency offset in Hz.
            - n_M (int): Index of the maximum peak in the correlation metric.
    """
    #N = N//2
    y_shifted = np.zeros(len(y)+ N, dtype=np.complex64)
    y_shifted[N:] = y
    P = np.zeros(len(y), dtype=np.complex64)
    R1 = np.zeros(len(y), dtype=np.complex64)
    R2 = np.zeros(len(y), dtype=np.complex64)

    for n in range(len(P)):
        for m in range(N): #on itère sur tout les valeurs de la taille de notre sequence de répétition
            if(n+m < len(y)):
                P[n] += np.conj(y[n+m])*y_shifted[n+m]
                R1[n] += np.abs(y[n+m])**2
                R2[n] += np.abs(y_shifted[n+m])**2

    metric = np.abs(P[1:])**2 /(R1[1:] * R2[1:])
    metric = np.nan_to_num(metric)

    metric = metric[:-100] #there is ALWAYS a peak at the end of the metric, we remove it WHY :'(
    n_M = np.argmax(metric)
    metric = metric/metric[n_M]
    n_L = 0
    n_U = 0
    for i in range(n_M, len(metric)):
        if(metric[i] < 0.9):
            n_U = i
            break
    for i in range(n_M, 0, -1):
        if(metric[i] < 0.9):
            n_L = i
            break

    n_hat = (n_L + n_U)/2

    if(0): #for visualisation purposes the user can set this to 1

        print("n_M = ",n_M)
        print("n_L = ",n_L)
        print("n_U = ",n_U)
        print("n_hat = ",n_hat)
        plt.figure( figsize=(10, 5))
        plt.subplot(1, 1, 1).set_xlim(268, 348)
        plt.subplot(1, 1, 1).set_ylim(0, 1.2)
        plt.plot(np.abs(y), color = "blue", label = "y") 
        plt.plot(metric, color = "red",label = "metric S&C")
        plt.axvline(x=n_M, color = "green", label = "n_M")
        plt.axvline(x=n_L, color = "orange", label = "n_L")
        plt.axvline(x=n_U, color = "orange", label = "n_U")
        plt.axvline(x=n_hat, color = "black", label = "n_hat")
        plt.legend()
        plt.title("Schmidl and Cox algorithm")
        plt.xlabel("n [-]")
        plt.ylabel("Amplitude [-]")
        plt.savefig("Schmidl_and_Cox.svg")
        plt.show()

    phase = np.angle(P)
    Delta_f = phase/(np.pi * T*N)
    
    return n_hat , Delta_f, n_M #need to add n_M for the test file !! 

def Packet_detector(y: np.ndarray, N: int, T: float, Algo: str = "correlation") -> tuple[np.ndarray, float, np.ndarray, int]:
    """Detects OFDM packets and estimates CFO using various algorithms.

    This function takes a received signal (`y`), the length of a known repeating sequence (`N`),
    and the sampling period (`T`). It detects the start of an OFDM packet and estimates the
    carrier frequency offset (CFO) using the specified algorithm (`Algo`). 

    Args:
        y (np.ndarray): The received signal as a 1D complex array.
        N (int): Length of the known repeating sequence (e.g., the L-LTF in WiFi).
        T (float): Sampling period of the signal.
        Algo (str, optional): Algorithm to use for packet detection and CFO estimation.
            Supported options are:
                - "S&C": Schmidl & Cox algorithm
                - "Moose": Modified Overlap-Save (MOOSE) algorithm
                - "correlation": Simple correlation-based method
                - "new_correlation": An alternative correlation method
                - "new_new_correlation": Another correlation method (using HE-LTF)
            Defaults to "correlation".

    Returns:
        tuple: A tuple containing:
            - y_reshifted (np.ndarray): The received signal shifted to align with the start of the packet.
            - CFO (float): The estimated carrier frequency offset in Hz. #a dummy value is returned if the CFO is not estimated
            - L_LTF_est_seq (np.ndarray): The estimated L-LTF sequence. #may not be used after
            - n_hat (int): The estimated start index of the packet.

    Raises:
        ValueError: If an invalid `Algo` is provided.

    Notes:
        - Different algorithms may have different performance and assumptions about the signal structure.
        - The "new_new_correlation" algorithm is the currently selected method, which uses the HE-LTF for correlation.
        - For visualization, you can set the debug flag to '1' within the Schmidl & Cox section (currently '0').
    """
    n_M = 0
    if(Algo == "S&C"):
        n_hat, Delta_f, n_M = Schmidl_and_Cox(y, N , T)
        CFO = Delta_f[int(n_hat +1)]
        y_reshifted = np.roll(y, -(int(n_hat) + 64 + 20 + 1)) #the additions of the terms is performed in this way for the comparisons between the methods  #64 = length of the L-LTF, 20 = length of the cyclic prefix as this methods is not selected, it has been kept as such

        if(0): #for visualisation purposes the user can set this to 1
            plt.figure( figsize=(10, 5))
            plt.subplot(1, 4, 1)
            plt.plot(np.abs(y))
            plt.title("y")
            plt.subplot(1, 4, 2)
            plt.plot(np.abs(y_reshifted))
            plt.title("y_reshifted")
            #add a vertical line on y_shifted at 880
            plt.axvline(x=880, color = "green", label = "where end packet")
            #make a zoom in a 3rd plot on the right of the 2 first plots 
            plt.subplot(1, 4, 3).set_xlim(len(y_reshifted) - 40, len(y_reshifted)+1)
            plt.plot(np.abs(y_reshifted))
            plt.axvline(x=880, color = "green", label = "where end packet")
            plt.title("y_reshifted")
            plt.subplot(1, 4, 4).set_xlim(0, 20)
            plt.plot(np.abs(y_reshifted))
            plt.show()
            plt.figure(figsize = (10,5))
            plt.subplot(1,1,1).set_xlim(860 , 900)
            plt.plot(np.abs(y_reshifted))
            plt.axvline(x=880, color = "green", label = "where end packet")
            plt.show()

        L_LTF_est_seq = np.roll(y , -(int(n_hat)-N - N//2 +1))[:N+N//2]

    if(Algo == "Moose"):
        L_LTF_sig = L_LTF(2*N , 0 , 0 , 0 , 0)
        corr_y = np.correlate(y , L_LTF_sig , mode = "full")
        n_hat = np.argmax(np.abs(corr_y))
        print(n_hat)
        _,Delta_f,_ = Schmidl_and_Cox(y, N , T)
        CFO = Delta_f[int(n_hat +1 - 64)] #we add 64 because we need to have the miidle of the L-LTF, used as such for the comparison between the methods
        L_LTF_est_seq = 0
        y_reshifted = np.roll(y , -(n_hat - 160 - 80 + 1 + 160 + 80 + 20 +0))
    #########################################################################################################
    if(Algo == "correlation"):
        L_LTF_sig = L_LTF(2*N , 0 , 0 , 0 , 0)
        corr_y = np.correlate(y , L_LTF_sig , mode = "full")
        n_hat = np.argmax(np.abs(corr_y))
        CFO = 9999999 #this is a dummy value, it is not used in this method
        y_reshifted = np.roll(y , -(n_hat - 160 - 80 + 1 + 160 + 80 + 20 +0))
        L_LTF_est_seq = np.roll(y , -(n_hat - 160 - 80 + 1 + 160 + 80 - 64*2))[:2*64]
    if(Algo == "new_correlation"):
        L_LTF_sig = L_LTF(2*N , 0 , 0 , 0 , 0)
        corr_y = np.correlate(y , L_LTF_sig )
        n_hat = np.argmax(np.abs(corr_y))
        peak = n_hat + 30 + 160 + 1
        y_reshifted = np.roll(y , -(peak))
        CFO = 9999999 #this is a dummy value, it is not used in this method
        L_LTF_est_seq = 0
    #########################################################################################################
    elif(Algo == "new_new_correlation"): #it is the method that has been selected  
        HELTF = np.squeeze(generate_preamble("Preamble")[6])
        corr_y = np.correlate(y, HELTF)
        n_hat = np.argmax(np.abs(corr_y))
        peak = n_hat + len(HELTF) + 1
        y_reshifted = np.roll(y, -(peak))
        HE_LTF_seq = y[n_hat:n_hat+len(HELTF)]

        CFO = 9999999 #this is a dummy value, it is not used in this method
        L_LTF_est_seq = HE_LTF_seq

    return y_reshifted , CFO , L_LTF_est_seq , n_hat
