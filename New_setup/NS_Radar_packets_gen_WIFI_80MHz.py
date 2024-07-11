#%matplotlib qt

import numpy as np
import sys
import os
import pandas as pd
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import matplotlib.pyplot as plt

from Communication.mapping import bin_to_mod
from Communication.mapping import plot_constellation
from Communication.mapping import mod_to_bin


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
from Communication.Synchronization import Packet_detector
from Communication.OFDM import p_to_s
from Communication.OFDM import CFO_Corrector
from Communication.OFDM import center_around_0Hz
from Communication.OFDM import decenter_around_0Hz

from Communication.channel import SISO_channel

from Communication.Synchronization import FFT_CFO_estimation


sys.path.append(os.path.join(os.path.dirname(__file__), 'wifi'))

from Communication.wifi.L_STF import L_STF
from Communication.wifi.L_LTF import L_LTF
from Communication.wifi.L_SIG import L_SIG

from Communication.generate_preamble import generate_preamble



import Communication.OFDM as ofdm

from Communication.radcom_OFDM import plot_Range_Doppler_Map, Radar_Channel_state_estimation, Radar_Range_Doppler_processing, OFDM_radar_range_1D




SIMU = False
path = path = os.path.join(os.path.dirname(__file__))+"/"
FILE = path + "tests/WIFI_Files_80MHz/"








def Simu():
    #true value of the speed of light
    c = 299792458
    fc = 3.2e9                      #carrier frequency
    lamb = c/fc

    m_wanted = 10 #go from 0 to m_wanted
    v_wanted = 10 #go from -v_wanted to v_wanted
    N_ticks_X = 5
    N_ticks_Y = 21
    interp = False

    ########################
    ### Reference signal ###  Used in the simu if we generate rx signal
    ########################
    dist_ref = 0
    additionnal_delay = 100
    tau_ref = (2*dist_ref + additionnal_delay)/c
    f_D_ref = 0

    ########################
    ######## Cible #########  Used in the simu if we generate rx signal
    ########################

    tau1 = (2*5 + additionnal_delay)/c
    v1 = 0
    f_D1 = 2*fc*v1/c

    tau2 = (2*10 + additionnal_delay)/c
    v2 = 5
    f_D2 = 2*fc*v2/c

    tau3 = (2*3 + additionnal_delay)/c
    v3 = 0
    f_D3 = 2*fc*v3/c

    real_range = [1, 5,10,3]



    number_of_bits = 1024 * 1519 * 2                    #Number of bits to send
    subcarrier_spacing = 78.125e3                       #Subcarrier spacing
    Bandwidth_USRP = 100e6                              #Bandwidth of the USRP
    Numb_subcarrier = Bandwidth_USRP//subcarrier_spacing
    print("Numb_subcarrier = ",Numb_subcarrier)
    Number_carrier_to_kill = int(20e6//subcarrier_spacing)
    print("Number_carrier_to_kill = ",Number_carrier_to_kill)
    N = 1280                                            #Number of subcarriers in order to have at 100MHz sampling rate a subcarrier spacing of 78.125kHz
    All_carriers = np.arange(0 , N , 1)
    unused_carrier = All_carriers[-Number_carrier_to_kill:] #The last 20MHz are not used
    data_carrier = np.delete(All_carriers, unused_carrier)
    pilot_carrier = data_carrier                        #We put the pilot on all the data carriers
    N_data_carrier = len(data_carrier)                  #Number of data carriers


    pilotValue = np.array([1+1j , 1-1j , -1+1j , -1-1j])            #The 4-QAM pilot values
    pilotValue = np.random.choice(pilotValue, len(pilot_carrier))   #Random pilot values
    pilotValue[0] = 1+1j                                            #The first pilot is always 1+1j for the CFO estimation

    cp = 4                     #Cyclic prefix length -> here we take 1/4 of the OFDM symbol
    d = 1                       #distance in the constellation
    map = "4-QAM"               #mapping type


    pilot_rate = 4           #Pilot rate
    pilot_freq = 0 

    #####
    M = 2 #over sampling factor
    L = 2 #over sampling factor
    ####
    tx_samp_rate = 100e6*M # TX sample rate [Hz]
    rx_samp_rate = 100e6*L # RX sample rate [Hz]
    fs = tx_samp_rate
    T = M*N/fs #T = M*N/fs ATTENTION
    T = M/fs
    N_cp = N//cp




    ###############################################
    #                Signal creation              #
    ###############################################

    #generate OFDM signal
    x = np.random.randint(2, size=number_of_bits)
    x_mod = bin_to_mod(x, d, map)
    x_parallel = s_to_p(x_mod, N_data_carrier, unused_carrier) 

    x_parallel = OFDM_symbol(x_parallel, pilotValue, data_carrier, pilot_carrier, N, pilot_freq= pilot_freq, Pilote_rate= pilot_rate , all_carrier= All_carriers, unused_carrier = unused_carrier).T
    x_parallel[0,:] = (1+1j)
    x_time = IFFT(x_parallel , L)

    x_time_centered = center_around_0Hz(x_time, L, N)
    x_time_cp = add_cp(x_time_centered, N_cp, L)
    payload = p_to_s(x_time_cp)



    #generate the preamble
    preamble = generate_preamble("Preamble") 
    LSTF = np.squeeze(preamble[0])
    LSIG = np.squeeze(preamble[2])
    LLTF = np.squeeze(preamble[1])
    RLSIG = np.squeeze(preamble[3])
    HELTF = np.squeeze(preamble[6])
    HESIGA =np.squeeze(preamble[4])
    HESTF = np.squeeze(preamble[5])

    #concatenate the preamble and the payload
    tx_signal = np.concatenate((LSTF , LLTF, LSIG, RLSIG, HESIGA, HESTF,HELTF, payload))
    len_preamble = len(LSTF) + len(LLTF) + len(LSIG) + len(RLSIG) + len(HESIGA) + len(HESTF) + len(HELTF)



    #save tx_signal in a txt file
    tx_sig_real = np.real(tx_signal)
    tx_sig_imag = np.imag(tx_signal)

    #Need to do this to interface with the C++ code
    vectorIn = np.zeros((len(tx_signal)*2))
    vectorIn[::2] = tx_sig_real
    vectorIn[1::2] = tx_sig_imag

    #variable to use in the C++ code when calling the function
    print("------------------------------------------")
    print("SIG LEN", len(vectorIn)//2)
    print("------------------------------------------")

    index = "_1" #for flexibility
    #We save everything in files to reuse them at the receiver
    np.savetxt(FILE+"New_test_files/tx_signal_"+str(number_of_bits)+index+".txt", vectorIn/np.max(vectorIn)*0.7)
    np.save(FILE+"New_test_files/tx_signal_"+str(number_of_bits)+index+".npy", vectorIn/np.max(vectorIn)*0.7)
    np.save(FILE+"New_test_files/tx_signal_TRUE_"+str(number_of_bits)+index+".npy", tx_signal)
    np.save(FILE+"New_test_files/pilotValue_"+str(number_of_bits)+index+".npy", pilotValue)
    np.save(FILE+"New_test_files/payload_"+str(number_of_bits)+index+".npy", payload)
    np.save(FILE+"New_test_files/x_parallel"+str(number_of_bits)+index+".npy", x_parallel)
    np.save(FILE+"New_test_files/x"+str(number_of_bits)+index+".npy", x)


    ###############################################
    #                     Channel                 #
    ###############################################
    
    #if we want to simulate the reception of the signal we need to generate the channel
    if(SIMU):
        h = [[1 , f_D_ref , tau_ref], [0 , f_D1 , tau1], [0, f_D2, tau2], [0,f_D3,tau3]]         #generate the channel
        var_noise = 0                                                                            #noise variance
        rx_sample = SISO_channel(tx_signal, h, var_noise, fs_tx = fs, fs_rx = fs)                #simulate the reception of the signal
        np.save(FILE+"New_test_files/rx_sample"+str(number_of_bits)+index+".npy", rx_sample)     #save the received signal in a file

        

        ###############################################
        #               Radar Processing             #
        ###############################################

        rx_sample_com = rx_sample
        rx_sig_corrrected, _, HELTF_est ,_ = Packet_detector(rx_sample_com, N, T, Algo="new_new_correlation")
        rx_sig_corr = rx_sig_corrrected[0: len(payload)]

        n_hat_rad = np.argmax(np.abs(np.correlate(rx_sample, tx_signal)))
        rx_sig_corr_RAD = rx_sample[n_hat_rad+len_preamble:n_hat_rad+ len(tx_signal)]
        rx_sig_corr_COM = rx_sig_corrrected[0: len(payload)]

        ##### OFDM demodulation ######
        rx_sig_parallel = s_to_p(rx_sig_corr_RAD, (N + N_cp)*M)
        rx_sig_no_CP = remove_cp(rx_sig_parallel, N_cp, N,M)
        rx_sig_no_CP_decentered = decenter_around_0Hz(rx_sig_no_CP, M, N)
        rx_sig_no_CP = rx_sig_no_CP_decentered
        rx_sig_est = DFT(rx_sig_no_CP, M)


        D = Radar_Channel_state_estimation(rx_sig_est, x_parallel, unused_carrier)


        ###############################################
        #                 Load DC                     #
        ###############################################
        D_cleaned = np.copy(D)# - np.mean(D_DC)

        #here only the values at 0m/s are removed -> need to do the same processing as the one done in the receiver to increase the performance
        for i in range(D.shape[0]):
            D[i,:] = D[i,:] #- np.mean(D[i,:])
            D_cleaned[i,:] = D_cleaned[i,:] - np.mean(D[i,:])
        


        ZP = 5
        Z = Radar_Range_Doppler_processing(D_cleaned, ZP)

        N_tot = N + N_cp
        T_PRI = N_tot/fs #Temps d'un symbole OFDM
        B = fs/M
        max_speed = c/(4*fc * T_PRI)
        max_range = N*c/(2*B)

        plot_Range_Doppler_Map(Z, m_wanted , v_wanted , N_ticks_X , N_ticks_Y , interp , max_range , max_speed, dist_ref)


        rx_sig_corr = rx_sig_corr_COM
        rx_sig_parallel = s_to_p(rx_sig_corr, (N + N_cp)*M)
        rx_sig_no_CP = remove_cp(rx_sig_parallel, N_cp, N, M)
        rx_sig_no_CP_decentered = decenter_around_0Hz(rx_sig_no_CP, M, N)
        rx_sig_est = DFT(rx_sig_no_CP_decentered, M)

        CFO_FFT = FFT_CFO_estimation(rx_sig_corr, N, T/M, N_cp, M) 
        print("CFO_FFT = ",CFO_FFT)

        rx_sig_est_equalized = equalize(rx_sig_est, pilot_carrier, pilotValue, pilot_freq = 0, Pilote_rate= pilot_rate,All_carriers = All_carriers)
        data_est = Extract_data(rx_sig_est_equalized, data_carrier,pilot_freq=pilot_freq , Pilote_rate = pilot_rate)
        data_est = p_to_s(data_est)


        # demodulate x_parallel
        x = mod_to_bin(p_to_s(Extract_data(x_parallel, data_carrier, pilot_freq=pilot_freq, Pilote_rate=pilot_rate)), d, map)
        y = data_est[:number_of_bits//2] 
        print(number_of_bits)
        BER_non_eq = BER(x, mod_to_bin(p_to_s(Extract_data(rx_sig_est, data_carrier,pilot_freq=pilot_freq , Pilote_rate = pilot_rate)), d, map)[:number_of_bits])
        print("BER non equalized = ",BER_non_eq)

        BER_eq = BER(x, mod_to_bin(data_est, d, map)[:number_of_bits])
        print("BER equalized = ",BER_eq)



Simu()