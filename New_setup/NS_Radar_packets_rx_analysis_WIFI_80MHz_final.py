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




len_measurement = 19474560
ZP = 5 #zero padding for the Range Doppler processing
save = False #save the figures



path = os.path.join(os.path.dirname(__file__))+"/"


FILE = path+"tests/WIFI_Files_80MHz/"

folder = FILE+"New_test_files/RX_files/To_process/"

files_in_folder = os.listdir(folder)
print("files_in_folder", files_in_folder)

for filename in files_in_folder:
    print("FILENAME = ",filename)
    filename = folder +"/"+ filename


    with open(filename, 'rb') as f:
        data = np.fromfile(f, dtype=np.float32)
        data = data.astype(np.complex64)
        v = data[::2] + 1j * data[1::2]
        v = v.reshape(-1, 1)  # Column vector0
    rx_sample = np.squeeze(v)


    Num_meas = rx_sample.shape[0]//len_measurement
    print("Num_meas = ",Num_meas)


    #true value of the speed of light
    c = 299792458
    fc = 3.2e9                      #carrier frequency

    lamb = c/fc

    m_wanted = 35 #go from 0 to m_wanted
    v_wanted = 10 #go from -v_wanted to v_wanted
    N_ticks_X = 5
    N_ticks_Y = 21
    interp = False

    ########################
    ### Reference signal ###
    ########################
    dist_ref = 0.5 #distance of the reference point in meters (in general 2 times the distances between the antennas works well)

    ########################
    ######## Cible #########
    ########################

    real_range = [0] #may be used in a future version to add the positions of the targets

    number_of_bits =  1024 * 1519*2 #number of bits to send
    subcarrier_spacing = 78.125e3   #Subcarrier spacing
    Bandwidth_USRP = 100e6          #Bandwidth of the USRP
    Numb_subcarrier = Bandwidth_USRP//subcarrier_spacing
    Number_carrier_to_kill = int(20e6//subcarrier_spacing) #Number of subcarriers to kill to have a bandwidth of 80MHz
    N = 1280 #Number of subcarriers in order to have at 100MHz sampling rate a subcarrier spacing of 78.125kHz
    All_carriers = np.arange(0 , N , 1)
    unused_carrier = All_carriers[-Number_carrier_to_kill:] #The last subcarriers are not used
    data_carrier = np.delete(All_carriers, unused_carrier)  #Data carriers
    pilot_carrier = data_carrier #Pilot carriers are the same as the data carriers because we sent whole OFDM symbols of pilots 
    N_data_carrier = len(data_carrier)


    pilotValue = np.array([1+1j , 1-1j , -1+1j , -1-1j])                    #possible values for the pilots
    pilotValue = np.random.choice(pilotValue, len(pilot_carrier))           #random choice of the pilots
    pilotValue[0] = 1+1j                                                    #The first pilot is always 1+1j for the CFO estimation (it is the reference)

    cp = 4                      #cyclic prefix length (1/4 of the OFDM symbol length)
    d = 1                       #distance in the constellation
    map = "4-QAM"               #mapping type


    pilot_rate = 4              #Pilot rate (1 pilot OFDM symbol every pilot_rate OFDM symbols)
    pilot_freq = 0              #used to indicate that the pilots are on whole OFDM symbol

    carrier_freq = fc # Carrier frequency [Hz]
    #####
    M = 2 #over sampling factor at the transmitter
    L = 2 #over sampling factor at the receiver
    ####
    tx_samp_rate = 100e6*M # TX sample rate [Hz]
    rx_samp_rate = 100e6*L # RX sample rate [Hz]
    fs = tx_samp_rate
    T = M/fs
    N_cp = N//cp            #Cyclic prefix length in symbols
    len_preamble =  8320    #length of the preamble of the wifi packet





    ###############################################
    #                Signal loading               #
    ###############################################





    index = "_1"
    tx_signal = np.load(FILE+"New_test_files/tx_signal_TRUE_"+str(number_of_bits)+index+".npy")
    pilotValue = np.load(FILE+"New_test_files/pilotValue_"+str(number_of_bits)+index+".npy")
    payload = np.load(FILE+"New_test_files/payload_"+str(number_of_bits)+index+".npy")
    x_parallel = np.load(FILE+"New_test_files/x_parallel"+str(number_of_bits)+index+".npy")
    x = np.load(FILE+"New_test_files/x"+str(number_of_bits)+index+".npy")
    print("FILES LOADED")
    print("len payload", len(payload))
    print("number of bits", number_of_bits)


    ###############################################
    #               Radar Procerssing             #
    ###############################################

    #Preamble generation for the correlation
    preamble = generate_preamble("Preamble")
    LSTF = np.squeeze(preamble[0])
    LSIG = np.squeeze(preamble[2])
    LLTF = np.squeeze(preamble[1])
    RLSIG = np.squeeze(preamble[3])
    HELTF = np.squeeze(preamble[6])
    HESIGA =np.squeeze(preamble[4])
    HESTF = np.squeeze(preamble[5])

    preamble_for_corr = np.concatenate((LSTF , LLTF, LSIG, RLSIG, HESIGA, HESTF,HELTF))

    for l in range(Num_meas):
        print("Doing measurement:",l+1)

        fig_save_path_clean =  filename[:-4]+"CLEAN_iter_num_"+str(l+1)+".svg"
        fig_save_path =  filename[:-4]+"_iter_num_"+str(l+1)+".svg"
        rx_samples_l = rx_sample[l*len_measurement:(l+1)*len_measurement]
        rx_sample_com = rx_sample[l*len_measurement:(l+1)*len_measurement]
        rx_sig_corrrected, _, HELTF_est ,_ = Packet_detector(rx_sample_com, N, T, Algo="new_new_correlation")
        rx_sig_corr = rx_sig_corrrected[0: len(payload)]
        n_hat_rad = np.argmax(np.abs(np.correlate(rx_samples_l, preamble_for_corr)[:int(1.5*len(tx_signal))]))
        rx_sig_corr_RAD = rx_samples_l[n_hat_rad+len_preamble:n_hat_rad+ len(tx_signal)]
        rx_sig_corr_COM = rx_sig_corrrected[0: len(payload)]

        print("CORRELATION DONE")

        ##### OFDM demodulation ######
        rx_sig_parallel = s_to_p(rx_sig_corr_RAD, (N + N_cp)*M)
        rx_sig_no_CP = remove_cp(rx_sig_parallel, N_cp, N,M)
        rx_sig_no_CP_decentered = decenter_around_0Hz(rx_sig_no_CP, M, N)
        rx_sig_no_CP = rx_sig_no_CP_decentered
        rx_sig_est = DFT(rx_sig_no_CP, M)




        D = Radar_Channel_state_estimation(rx_sig_est, x_parallel, unused_carrier)
        print("D MATRIX COMPUTED")

        ######################################################################
        #                 Cleaning of the channel matrix                     #
        ######################################################################

        D_cleaned = np.copy(D)#  
        D_cleaned_FFT = np.fft.fftshift(np.fft.fft2(D_cleaned))
        #remove the column of the DC and make it equal to 0 -> remove all the targets at 0 m/s
        D_cleaned_FFT[:,D_cleaned_FFT.shape[1]//2] = 0
        #remove the line of the DC and make it equal to 0  -> remove all the targets at 0m, and the next bins (the user may want to keep the targets at 0m, remove more or less bins), it is hardcoded here for the moment
        D_cleaned_FFT[D_cleaned_FFT.shape[0]//2,:] = 0
        D_cleaned_FFT[D_cleaned_FFT.shape[0]//2-1,:] = 0
        D_cleaned_FFT[D_cleaned_FFT.shape[0]//2+1,:] = 0
        D_cleaned_FFT[D_cleaned_FFT.shape[0]//2-2,:] = 0
        D_cleaned_FFT[D_cleaned_FFT.shape[0]//2+2,:] = 0
        
        #the distance-speed map is cleaned and we can go back to the time domain to get the cleaned channel matrix
        D_cleaned = np.fft.ifft2(np.fft.ifftshift(D_cleaned_FFT))

        #compute some useful parameters for the plots
        N_tot = N + N_cp
        T_PRI = N_tot/fs #Temps d'un symbole OFDM
        B = fs/M
        max_speed = c/(4*fc * T_PRI)
        max_range = N*c/(2*B)


        num_Symb2take = D_cleaned.shape[1]


        Z_cleaned = Radar_Range_Doppler_processing(D_cleaned, ZP)
        Z = Radar_Range_Doppler_processing(D, ZP)

        N_tot = N + N_cp
        T_PRI = N_tot/fs #Temps d'un symbole OFDM
        B = fs/M
        max_speed = c/(4*fc * T_PRI)
        max_range = N*c/(2*B)


        plot_Range_Doppler_Map(Z_cleaned, m_wanted , v_wanted , N_ticks_X , N_ticks_Y , interp , max_range , max_speed, dist_ref, path=fig_save_path_clean)
        plot_Range_Doppler_Map(Z, m_wanted , v_wanted , N_ticks_X , N_ticks_Y , interp , max_range , max_speed, dist_ref, path=fig_save_path)




############################################################################################################ 
#                                     Communication processing                                             #
############################################################################################################
        
        rx_sig_corr = rx_sig_corr_COM                                          #The signal is the one after the correlation
        rx_sig_parallel = s_to_p(rx_sig_corr, (N + N_cp)*M)                    #Serial to parallel
        rx_sig_no_CP = remove_cp(rx_sig_parallel, N_cp, N, M)                  #Remove the cyclic prefix
        rx_sig_no_CP_decentered = decenter_around_0Hz(rx_sig_no_CP, M, N)      #Decenter the signal around 0Hz
        rx_sig_est = DFT(rx_sig_no_CP_decentered, M)                           #DFT of the signal


        CFO_FFT = FFT_CFO_estimation(rx_sig_corr, N, T/M, N_cp, M)          #CFO estimation
        print("CFO_FFT = ",CFO_FFT)
        #ATTENTION: The CFO is estimated but not corrected here as the Octoclock is used for the USRP and the CFO is very small
        #           If the Octoclock is not used, the CFO should be corrected here

        rx_sig_est_equalized = equalize(rx_sig_est, pilot_carrier, pilotValue, pilot_freq = 0, Pilote_rate= pilot_rate,All_carriers = All_carriers) #Equalization of the signal using the pilots
        data_est = Extract_data(rx_sig_est_equalized, data_carrier,pilot_freq=pilot_freq , Pilote_rate = pilot_rate) 
        data_est = p_to_s(data_est)
        x = mod_to_bin(p_to_s(Extract_data(x_parallel, data_carrier, pilot_freq=pilot_freq, Pilote_rate=pilot_rate)), d, map)
        y = data_est[:number_of_bits//2] 

        BER_non_eq = BER(x, mod_to_bin(p_to_s(Extract_data(rx_sig_est, data_carrier,pilot_freq=pilot_freq , Pilote_rate = pilot_rate)), d, map)[:number_of_bits])
        print("BER non equalized = ",BER_non_eq)

        BER_eq = BER(x, mod_to_bin(data_est, d, map)[:number_of_bits])
        print("BER equalized = ",BER_eq)



 