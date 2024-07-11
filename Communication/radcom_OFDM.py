#%matplotlib qt
#DOC STRING GENERATED AFTER THE SUBMISSION DEADLINE WITH THE HELP OF Gemini

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



import Communication.OFDM as ofdm

c = 299792458





def Radar_Channel_state_estimation(rx_sig_est, symbols, unused_carrier) -> np.ndarray:
    """Estimates the channel state for a radar system.

    This function takes a received signal (`rx_sig_est`), the transmitted symbols (`symbols`), and
    a list of unused subcarriers (`unused_carrier`). It removes the unused subcarriers and any
    trailing zero-padding from both the received signal and the symbols. Then, it reshapes the
    received signal to match the shape of the symbols and calculates the channel state estimate
    by dividing the received signal by the transmitted symbols.

    Args:
        rx_sig_est (np.ndarray): The received signal, a complex array with shape
                                (num_subcarriers, num_samples).
        symbols (np.ndarray): The transmitted symbols, a complex array with shape
                              (num_subcarriers, num_samples).
        unused_carrier (np.ndarray): An array of indices indicating the unused subcarriers.

    Returns:
        np.ndarray: The estimated channel state, a complex array with the same shape as 
                    `rx_sig_est` and `symbols` after removing unused carriers.
    """

    #Removal of the unused carriers
    rx_sig_est = np.delete(rx_sig_est, unused_carrier, axis = 0)
    symbols = np.delete(symbols, unused_carrier, axis = 0)
    
    #the last column of the matrix sometimes contains zeros to zero pad the signal
    symbols = np.delete(symbols, -1, axis = 1)
    rx_sig_est = np.delete(rx_sig_est, -1, axis = 1)

    #reshape the matrix to have the same shape as the symbols matrix
    rx_sig_est = rx_sig_est.reshape(symbols.shape)
    D = rx_sig_est/symbols    
    return D

def Radar_Range_Doppler_processing(D, ZP):
    """Performs Range-Doppler processing on a radar channel state matrix.

    This function takes a channel state matrix (`D`) and applies zero-padding (`ZP`) before
    performing a 2D Fast Fourier Transform (FFT) to obtain the Range-Doppler map. The FFT is
    computed along the range (axis 0) and Doppler (axis 1) dimensions. The output is shifted
    to center the zero-frequency component for better visualization.

    Args:
        D (np.ndarray): A complex array representing the channel state matrix with shape
                       (num_range_bins, num_pulses).
        ZP (int): Zero-padding factor, the number of times to repeat the matrix in each dimension.

    Returns:
        np.ndarray: A complex array representing the Range-Doppler map, with the same 
                    shape as the zero-padded `D` matrix.
    """

    D = np.pad(D, ((0,ZP*D.shape[0]),(0,ZP*D.shape[1])), 'constant', constant_values = 0)
    Z = np.fft.fftshift(np.fft.ifft(np.fft.fft(D,axis=0), axis=1), axes=(0)) #axis=0 is the range axis and axis=1 is the speed axis
    return Z




def plot_Range_Doppler_Map(Z, m_wanted , v_wanted , N_ticks_X , N_ticks_Y , interp, max_range , max_speed, dist_ref, ZP = 0, path= "non"):
    """Plots and analyzes a Range-Doppler map.

    This function takes a Range-Doppler map (`Z`) and performs the following:

    1. Converts the map to decibels (dB).
    2. Rolls and flips the map for proper visualization.
    3. Zooms in to a specified range and velocity region.
    4. Optionally applies Gaussian interpolation for smoother visualization.
    5. Adjusts for a reference distance.
    6. Plots the Range-Doppler map using a 'jet' colormap.
    7. Calculates and prints estimated range and speed of the maximum peak.
    8. Saves the plot (if `path` is provided) and the results in a CSV file.

    Args:
        Z (np.ndarray): The Range-Doppler map, a complex array.
        m_wanted (float): Desired maximum range for plotting (in meters).
        v_wanted (float): Desired maximum speed for plotting (in meters per second).
        N_ticks_X (int): Number of ticks on the x-axis (velocity).
        N_ticks_Y (int): Number of ticks on the y-axis (range).
        interp (bool): If True, applies Gaussian interpolation to the map.
        max_range (float): The maximum unambiguous range of the radar (in meters).
        max_speed (float): The maximum unambiguous speed of the radar (in meters per second).
        dist_ref (float): The reference distance for calibration (in meters).
        ZP (int, optional): Zero-padding factor used during Range-Doppler processing. Defaults to 0.
        path (str, optional): Path to save the plot. If "non", the plot is not saved. Defaults to "non".
    """
    cmap = plt.cm.get_cmap('jet')    
    Z_LOG = 10*np.log10(np.abs(Z)**2)


    pixel_to_m = max_range/Z_LOG.shape[0]
    pixel_to_v = max_speed/Z_LOG.shape[1]
    pixel_to_keep_m = int(np.ceil(m_wanted/pixel_to_m))
    pixel_to_keep_v = int(np.ceil(v_wanted/pixel_to_v))

    Z_LOG = np.roll(Z_LOG, -(Z_LOG.shape[0]+(ZP+1))//2, axis = 0) #For the range

    Z_LOG = np.roll(Z_LOG, (Z_LOG.shape[1]-1)//2, axis = 1) #For the speed

    Z_LOG = np.flip(Z_LOG, axis = 1) #For the speed

    Z_LOG_Zoomed_meter = Z_LOG[Z_LOG.shape[0]-pixel_to_keep_m:Z_LOG.shape[0],:]
    Z_LOG_Zoomed = Z_LOG_Zoomed_meter[:,Z_LOG.shape[1]//2-pixel_to_keep_v:Z_LOG.shape[1]//2+pixel_to_keep_v+1] #I center the image around 0m/s so I need to add 1

    ###############################################
    #         roll for Reference correction       #
    ###############################################
    num_pixel_to_callibrate = dist_ref / pixel_to_m  #number of pixel to callibrate
    Z_LOG_Zoomed = np.roll(Z_LOG_Zoomed, -int(num_pixel_to_callibrate), axis = 0)

    max_pos = np.unravel_index(np.argmax(Z_LOG, axis=None), Z_LOG.shape)

    plt.figure()
    if(interp):
        plt.imshow(Z_LOG_Zoomed, cmap=cmap, aspect='auto',interpolation='gaussian') #interpolation='gaussian',
    else:
        #ATTENTION : The user needs to change the vmin and vmax to have a better contrast on the image depending on the context - ATTENTION 
        Z_min = -45 
        Z_max = 35
        plt.imshow(Z_LOG_Zoomed, cmap=cmap, aspect='auto', vmin=Z_min, vmax=Z_max)




    ticks_x = np.linspace(0, Z_LOG_Zoomed.shape[1]-1, N_ticks_X)
    new_ticks_x = np.linspace(-v_wanted, v_wanted, N_ticks_X)
    new_ticks_x = np.around(new_ticks_x, decimals=1)
    plt.xticks(ticks_x , new_ticks_x, rotation=60)
    ticks_y = np.linspace(0, Z_LOG_Zoomed.shape[0]-1, N_ticks_Y)
    new_ticks_y = np.linspace(m_wanted, 0, N_ticks_Y)
    new_ticks_y = np.around(new_ticks_y, decimals=1)
    plt.yticks(ticks_y , new_ticks_y)
    plt.xlabel("Speed [m/s]")
    plt.ylabel("Distance [m]")
    plt.title("Distance-Doppler map")
    plt.colorbar()
    plt.savefig(path)


    max_pos = np.unravel_index(np.argmax(Z_LOG_Zoomed, axis=None), Z_LOG_Zoomed.shape)
    print("---------------------------------- RESULT ----------------------------------")
    print("max_pos = ",max_pos)
    print("Estimated speed is ", -v_wanted + max_pos[1]*2*v_wanted/(Z_LOG_Zoomed.shape[1]-1), "m/s with a precision of ", 2*v_wanted/(Z_LOG_Zoomed.shape[1]-1))
    print("Estimated range is ", m_wanted - max_pos[0]*m_wanted/(Z_LOG_Zoomed.shape[0]-1), "m with a precision of ", m_wanted/(Z_LOG_Zoomed.shape[0]-1))
    df = pd.DataFrame({"Estimated_speed" : [-v_wanted + max_pos[1]*2*v_wanted/(Z_LOG_Zoomed.shape[1]-1)], "Estimated_range" : [m_wanted - max_pos[0]*m_wanted/(Z_LOG_Zoomed.shape[0]-1)]})
    df.to_csv(path[:-4] + ".csv", mode='a', header=True)

    print("----------------------------------------------------------------------------")





def OFDM_radar_range_1D(D, m_wanted, dist_ref,num_Symb2take, ZP, B, save , path, real_range = [0], D_cleaned = "non", clutter = "non"):
    """ ATTENTION : This function is an old version and may not work properly anymore. Please use plot_Range_Doppler_Map which gives a 2D representation of the range and speed. 
                    It is given here for reference only and may be used as a starting point for a new implementation or to give ideas :-)

    Processes OFDM radar data to estimate range and plot results.

    This function takes complex OFDM data (`D`), processes it to estimate range, and performs the following:
        - OFDM demodulation: calculates the average and removes unused symbols.
        - Zero-padding: Applies zero-padding for finer FFT resolution.
        - FFT Calculation: Computes the FFT of the zero-padded data.
        - Logarithmic scaling: Converts the FFT output to decibels (dB).
        - Rolling & Flipping: Adjusts the range axis for correct visualization.
        - Calibration: Shifts the range axis to account for a reference distance.
        - Zooming: Focuses the plot on a desired range.
        - Peak detection: Finds peaks in the averaged, zoomed signal and estimates their ranges.
        - Plotting: Creates a plot of the processed data (optional).
        - Cleaning (optional): If `D_cleaned` is provided, performs the same processing on cleaned data.
        - Clutter estimation (optional): If `clutter` is provided, performs processing to estimate clutter.
        - Saving (optional): Saves the plot and estimation results to files (if `save` is True).

    Args:
        D (np.ndarray): Complex OFDM data as a 2D array (num_subcarriers, num_samples).
        m_wanted (float): Desired maximum range for plotting (in meters).
        dist_ref (float): Reference distance for calibration (in meters).
        num_Symb2take (int): Number of symbols to use for processing.
        ZP (int): Zero-padding factor.
        B (float): Bandwidth of the OFDM signal.
        save (bool): If True, saves the plot and results to files.
        path (str): Path to save the plot (only relevant if `save` is True).
        real_range (list, optional): List of real range values. Defaults to [0].
        D_cleaned (np.ndarray or str, optional): Cleaned OFDM data (if available). Defaults to "non".
        clutter (np.ndarray or str, optional): Clutter data (if available). Defaults to "non".

    Returns:
        None: The function displays the plot (if enabled) and saves results, but doesn't return data.

    """
     
    N = D.shape[0]
    #Take number_of_Symb2take symbols
    D = D[: , :num_Symb2take]
    D_average = np.mean(D, axis = 1)
    max_range = N*c/(2*B)

    #Zero padding
    D = np.pad(D, ((0, ZP*N),(0,0)), 'constant', constant_values = 0)
    D_average = np.pad(D_average, (0, ZP*N), 'constant', constant_values = 0)
    Z = np.fft.fftshift(np.fft.fft(D, axis = 0))
    Z_average = np.fft.fftshift(np.fft.fft(D_average, axis = 0))

    Z_LOG = 10*np.log10(np.abs(Z))
    Z_LOG = Z
    calibration_shift = 0

    pixel_to_m = max_range/len(Z_LOG)
    num_pixel_to_callibrate = dist_ref / pixel_to_m    #number of pixel to callibrate
    Z_LOG = np.roll(Z_LOG, -(Z_LOG.shape[0]+(ZP+1))//2 - (1), axis = 0)
    Z_LOG = np.roll(Z_LOG, -int(num_pixel_to_callibrate), axis = 0)
    Z_LOG = np.flip(Z_LOG)


    Z_LOG_average = Z_average
    Z_LOG_average = np.roll(Z_LOG_average, -(Z_LOG_average.shape[0]+(ZP+1))//2 - (1), axis = 0)

    Z_LOG_average = np.roll(Z_LOG_average, -int(num_pixel_to_callibrate), axis = 0)
    Z_LOG_average = np.flip(Z_LOG_average)



    range_array = np.linspace(0, max_range, N*(ZP+1))   

    index_to_keep = int(m_wanted*len(Z_LOG)/max_range)
    Z_LOG_Zoomed = Z_LOG[:index_to_keep]
    Z_LOG_average_Zoomed = Z_LOG_average[:index_to_keep]
    range_array_Zoomed = range_array[:index_to_keep]

    Z_LOG_average_Zoomed_processing = 10*np.log10(np.real(Z_LOG_average_Zoomed)**2+ np.imag(Z_LOG_average_Zoomed)**2)
    
    number_of_max = 2
    range_est_no_CR = []
    max_pos_arr_no_CR = []
    for i in range(number_of_max):
        #find the maximum value
        max_pos = np.unravel_index(np.argmax(Z_LOG_average_Zoomed_processing, axis=None), Z_LOG_average_Zoomed_processing.shape)
        max_pos_arr_no_CR.append(max_pos[0])
        range_est_no_CR.append(range_array_Zoomed[max_pos[0]])
        #remove all the values of Z_LOG_average_Zoomed_processing around the maximum where Z_LOG_average_Zoomed_processing_diff2 is positive
        j = max_pos[0]-1
        while((Z_LOG_average_Zoomed_processing[j] - Z_LOG_average_Zoomed_processing[j-1]) > 0):
            if(j == -1):
                break
            Z_LOG_average_Zoomed_processing[j] = -100
            j = j-1

        j = max_pos[0]
        if(j < len(Z_LOG_average_Zoomed_processing)-1):
            Z_LOG_average_Zoomed_processing[j] = -100
            break

        while((Z_LOG_average_Zoomed_processing[j] - Z_LOG_average_Zoomed_processing[j+1]) > 0):
            Z_LOG_average_Zoomed_processing[j] = -100
            j = j+1
            if(j == len(Z_LOG_average_Zoomed_processing)-1):
                break
        plt.figure()
        plt.plot(Z_LOG_average_Zoomed_processing)
        plt.show()

    if(type(D_cleaned) != str):
        N = D_cleaned.shape[0]
        #Take number_of_Symb2take symbols
        D_cleaned = D_cleaned[: , :num_Symb2take]
        D_cleaned_average = np.mean(D_cleaned, axis = 1)
        max_range = N*c/(2*B)

        #Zero padding
        D_cleaned = np.pad(D_cleaned, ((0, ZP*N),(0,0)), 'constant', constant_values = 0)
        D_cleaned_average = np.pad(D_cleaned_average, (0, ZP*N), 'constant', constant_values = 0)
        Z_cleaned = np.fft.fftshift(np.fft.fft(D_cleaned, axis = 0))
        Z_cleaned_average = np.fft.fftshift(np.fft.fft(D_cleaned_average, axis = 0))

        Z_cleaned_LOG = 10*np.log10(np.abs(Z_cleaned))
        Z_cleaned_LOG = Z_cleaned
        calibration_shift = 0

        pixel_to_m = max_range/len(Z_cleaned_LOG)
        num_pixel_to_callibrate = dist_ref / pixel_to_m    #number of pixel to callibrate
        Z_cleaned_LOG = np.roll(Z_cleaned_LOG, -(Z_cleaned_LOG.shape[0]+(ZP+1))//2 - (1), axis = 0)
        Z_cleaned_LOG = np.roll(Z_cleaned_LOG, -int(num_pixel_to_callibrate), axis = 0)
        Z_cleaned_LOG = np.flip(Z_cleaned_LOG)


        Z_cleaned_LOG_average = Z_cleaned_average
        Z_cleaned_LOG_average = np.roll(Z_cleaned_LOG_average, -(Z_cleaned_LOG_average.shape[0]+(ZP+1))//2 - (1), axis = 0)

        Z_cleaned_LOG_average = np.roll(Z_cleaned_LOG_average, -int(num_pixel_to_callibrate), axis = 0)
        Z_cleaned_LOG_average = np.flip(Z_cleaned_LOG_average)



        range_array = np.linspace(0, max_range, N*(ZP+1))   

        index_to_keep = int(m_wanted*len(Z_cleaned_LOG)/max_range)
        Z_cleaned_LOG_Zoomed = Z_cleaned_LOG[:index_to_keep]
        Z_cleaned_LOG_average_Zoomed = Z_cleaned_LOG_average[:index_to_keep]
        range_array_Zoomed = range_array[:index_to_keep]


        Z_cleaned_LOG_average_Zoomed_processing = 10*np.log10(np.real(Z_cleaned_LOG_average_Zoomed)**2+ np.imag(Z_cleaned_LOG_average_Zoomed)**2)
        number_of_max = 3
        range_est = []
        max_pos_arr = []

        for i in range(number_of_max):
            #find the maximum value
            max_pos = np.unravel_index(np.argmax(Z_cleaned_LOG_average_Zoomed_processing, axis=None), Z_cleaned_LOG_average_Zoomed_processing.shape)
            max_pos_arr.append(max_pos[0])
            range_est.append(range_array_Zoomed[max_pos[0]])
            #remove all the values of Z_LOG_average_Zoomed_processing around the maximum where Z_LOG_average_Zoomed_processing_diff2 is positive
            j = max_pos[0]-1
            while((Z_cleaned_LOG_average_Zoomed_processing[j] - Z_cleaned_LOG_average_Zoomed_processing[j-1]) > 0):
                if(j == -1):
                    break
                Z_cleaned_LOG_average_Zoomed_processing[j] = -100
                j = j-1

            j = max_pos[0]
            if(j < len(Z_cleaned_LOG_average_Zoomed_processing)-1):
                while((Z_cleaned_LOG_average_Zoomed_processing[j] - Z_cleaned_LOG_average_Zoomed_processing[j+1]) > 0):
                    Z_cleaned_LOG_average_Zoomed_processing[j] = -100
                    j = j+1
                    if(j == len(Z_cleaned_LOG_average_Zoomed_processing)-1):
                        break



    if(type(clutter) != str):
        N = clutter.shape[0]
        #Take number_of_Symb2take symbols
        clutter = clutter[: , :num_Symb2take]
        clutter_average = np.mean(clutter, axis = 1)
        max_range = N*c/(2*B)

        #Zero padding
        clutter = np.pad(clutter, ((0, ZP*N),(0,0)), 'constant', constant_values = 0)
        clutter_average = np.pad(clutter_average, (0, ZP*N), 'constant', constant_values = 0)
        Z_clutter_average = np.fft.fftshift(np.fft.fft(clutter_average, axis = 0))


        pixel_to_m = max_range/len(Z_clutter_average)
        num_pixel_to_callibrate = dist_ref / pixel_to_m    #number of pixel to callibrate

        Z_clutter_LOG_average = Z_clutter_average
        Z_clutter_LOG_average = np.roll(Z_clutter_LOG_average, -(Z_clutter_LOG_average.shape[0]+(ZP+1))//2 - (1), axis = 0)
        Z_clutter_LOG_average = np.roll(Z_clutter_LOG_average, -int(num_pixel_to_callibrate), axis = 0)
        Z_clutter_LOG_average = np.flip(Z_clutter_LOG_average)

        Z_clutter_LOG_average_Zoomed = Z_clutter_LOG_average[:index_to_keep]


    plt.figure( figsize=(10,6))
    plt.plot(range_array_Zoomed, 10*np.log10(np.real(Z_LOG_average_Zoomed)**2+ np.imag(Z_LOG_average_Zoomed)**2), linewidth = 2, color = "red",label = "Average power signal without CR")
    
    if(type(D_cleaned) != str):
        plt.plot(range_array_Zoomed, 10*np.log10(np.real(Z_cleaned_LOG_average_Zoomed)**2+ np.imag(Z_cleaned_LOG_average_Zoomed)**2), linewidth = 2, color = "blue",label = "Average power clutter removed (CR)")
    
    #add stars for the estimated range
    if(type(D_cleaned) != str):
        plt.plot(range_est[0], 10*np.log10(np.real(Z_cleaned_LOG_average_Zoomed[max_pos_arr[0]])**2+ np.imag(Z_cleaned_LOG_average_Zoomed[max_pos_arr[0]])**2), "x",markersize = 10, color = "green", label = "Estimated range")
        #plt.axvline(x=range_est[0], color = "green", linestyle = "--", label = "Estimated range")
        if(number_of_max > 1):
            for i in range(number_of_max-1):
                plt.plot(range_est[i+1], 10*np.log10(np.real(Z_cleaned_LOG_average_Zoomed[max_pos_arr[i+1]])**2+ np.imag(Z_cleaned_LOG_average_Zoomed[max_pos_arr[i+1]])**2), "x",markersize = 10, color = "green")
                #plt.axvline(x=range_est[i+1], color = "green", linestyle = "--")

    plt.plot(range_est_no_CR[0], 10*np.log10(np.real(Z_LOG_average_Zoomed[max_pos_arr_no_CR[0]])**2+ np.imag(Z_LOG_average_Zoomed[max_pos_arr_no_CR[0]])**2), "x",markersize = 10, color = "red", label = "Estimated range")
    # plt.axvline(x=range_est[0], color = "green", linestyle = "--", label = "Estimated range") 
    if(number_of_max > 1):
        for i in range(len(max_pos_arr_no_CR)-1):

            plt.plot(range_est_no_CR[i+1], 10*np.log10(np.real(Z_LOG_average_Zoomed[max_pos_arr_no_CR[i+1]])**2+ np.imag(Z_LOG_average_Zoomed[max_pos_arr_no_CR[i+1]])**2), "x",markersize = 10, color = "red")
            #  plt.axvline(x=range_est[i+1], color = "green", linestyle = "--")

    if(type(clutter) != str):
        plt.plot(range_array_Zoomed, 10*np.log10(np.real(Z_clutter_LOG_average_Zoomed)**2+ np.imag(Z_clutter_LOG_average_Zoomed)**2), linewidth = 1, color = "black",label = "Average power clutter",alpha = 0.3 )

        power_Z_LOG_average_Zoomed = np.real(Z_LOG_average_Zoomed)**2+ np.imag(Z_LOG_average_Zoomed)**2
        power_Z_clutter_LOG_average_Zoomed = np.real(Z_clutter_LOG_average_Zoomed)**2+ np.imag(Z_clutter_LOG_average_Zoomed)**2
        other_CR = np.abs(power_Z_LOG_average_Zoomed - power_Z_clutter_LOG_average_Zoomed)
        other_CR = 10*np.log10(other_CR)
        #plt.plot(range_array_Zoomed, other_CR, linewidth = 2, color = "green",label = "Average power CR method 2",alpha = 0.3, linestyle = "--" )
    # #add vertical lines for the real range
    
    if(real_range != [0]):
        plt.axvline(x=real_range[0], color = "black", label = "Real range")
        i_range = np.linspace(1, len(real_range)-1, len(real_range)-1)
        print("i_range = ",i_range)
        if(len(real_range) > 1):
            for i in (i_range):
                i = int(i)
                plt.axvline(x=real_range[i], color = "black")
            
    plt.title("Range profile dB")
    plt.xlabel("Range (m)")
    plt.ylabel("Amplitude (dB)")
    plt.legend()
    plt.xticks(np.arange(0, range_array_Zoomed[-1], 1))
    plt.grid()
    plt.savefig("Range_profile.svg")
    plt.show()

    if(save == True and real_range != [0]):
        #closest to real range
        range_est_closest_index = np.argmin(np.abs(np.array(range_est) - real_range[0]))
        range_est_closest = range_est[range_est_closest_index]
        range_est_no_CR_closest_index = np.argmin(np.abs(np.array(range_est_no_CR) - real_range[0]))
        range_est_no_CR_closest = range_est_no_CR[range_est_no_CR_closest_index]
        Error_with_CR_closest = np.abs(range_est_closest - real_range[0])
        Error_no_CR_closest = np.abs( range_est_no_CR_closest - real_range[0])    
        Error_with_CR = np.abs(range_est[0] - real_range[0])
        Error_no_CR = np.abs(range_est_no_CR[0] - real_range[0])
        df = pd.DataFrame({"Error_with_CR_closest" : [Error_with_CR_closest], "Error_no_CR_closest" : [Error_no_CR_closest], "Error_with_CR" : [Error_with_CR], "Error_no_CR" : [Error_no_CR], "real_range" : [real_range[0]], "range_est_with_CR" : [range_est[0]], "range_est_no_CR" : [range_est_no_CR[0]], "range_est_with_CR_closest" : [range_est[range_est_closest_index]], "range_est_no_CR_closest" : [range_est_no_CR[range_est_no_CR_closest_index]]})
        #add the other obtained df to the csv
        df.to_csv(path + "/Error_range_estimation.csv", mode='a', header=False)

    return 

