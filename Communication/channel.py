"""
SISO and MIMO channel models.
"""

__all__ = ["add_signals", \
           "SISO_channel", \
           "MIMO_channel"]

### Imports
from numpy import array,arange,zeros
from numpy import concatenate,pad
from numpy import ceil,floor,exp,sqrt,conj
from numpy import pi,complex64

from numpy.random import normal

from scipy.interpolate import interp1d

### Exceptions

### Functions
def add_signals(*sigs):
    """
    Adds multiple signals with different lengths together 

    Parameters
    ----------
    *sigs : List of Numpy 1D array
        signals to add.

    Returns
    -------
    out : Numpy 1D array
        Output signal.
    """

    out = array([],dtype=complex64)
    
    for sig in sigs:
        max_length = max((len(out),len(sig)))
        out  = pad(out, (0,max_length - len(out)), mode='constant',constant_values=0) \
         + pad(sig, (0,max_length - len(sig)), mode='constant',constant_values=0)
    
    return out

def SISO_channel(tx_sig,path_params,var_noise,fs_tx,fs_rx):
    """
    Computes the received signal when the emitted signal <tx_sig> passes trough
    a Single Input Single Output (SISO) multi-path communication channel with
    time-variant impulse response
    
    h(t,tau) = sum_k alpha_k * exp(1j*2*pi*f_D_k*t) * delta(tau-tau_k)

    where k=0,...,K-1 with K the number of paths, and alpha_k, f_D_k and tau_k
    are respectively the complex attenuations, Doppler frequencies and delays
    of each path.
    
    An Additive White Gaussian Noise (AWGN) also corrupts the output signal.    

    Parameters
    ----------
    tx_sig : Numpy 1D array
        Input signal.
    path_params : List
        List of K elements containing the paths parameters. Each element is
        another list containing the complex attenuations, Doppler frequencies 
        and delays of the paths. Here is an example for a three-path channel:
            
        [[alpha_0,f_D_0,tau_0], [alpha_1,f_D_1,tau_1], [alpha_2,f_D_2,tau_2]]
        
    var_noise : Float
        Complex noise power.
    fs_tx : Float
        Input sample rate, supposed to be a multiple of <fs_rx>.
    fs_rx : Float
        Output sample rate, supposed to be lower than <fs_tx>.

    Returns
    -------
    out : Numpy 1D array
        Output signal.
    """
    
    N_paths = len(path_params)
    R = fs_tx/fs_rx
    
    rx_sigs = []
    for i in range(N_paths):
        alpha,f_D,tau = path_params[i][:3]
    
        n_delay = int(ceil(fs_rx*tau))
        eps = fs_tx*tau - floor(fs_rx*tau) * R

        f = interp1d(arange(len(tx_sig)),tx_sig,kind="linear",bounds_error=False,fill_value=0)
        tx_interp = f(arange(((R-eps) % R),len(tx_sig),int(R)))
        
        rx_sig = concatenate((zeros((n_delay,),dtype=complex64),tx_interp))
        rx_sig *= alpha * exp(1j*2*pi*f_D*arange(len(rx_sig))/fs_rx)
        
        rx_sigs.append(rx_sig)
    
    out = add_signals(*rx_sigs)
    out += normal(0,sqrt(var_noise/2),len(out)) + 1j*normal(0,sqrt(var_noise/2),len(out))
    return out

def MIMO_channel(tx_sig,path_params,a_t,a_r,var_noise,fs_tx,fs_rx):
    """
     Computes the received signal when the emitted signal <tx_sig> passes trough
    a Multiple Input Multiple Output (MIMO) multi-path communication channel with
    time-variant impulse response between TX antenna i and RX antenna j
    
    h_ij(t,tau,theta_t,theta_r) = sum_k alpha_k * exp(1j*2*pi*fD_k*t) 
     * conj(a_t_i(theta_t)) * conj(a_r_j(theta_r)) * delta(tau-tau_k) 
     * delta(theta_t-theta_t_k) * delta(theta_r-theta_r_k)

    where k=0,...,K-1 with K the number of paths, and alpha_k, fD_k, tau_k, 
    theta_t_k, theta_r_k are respectively the complex attenuations, Doppler 
    frequencies, delays, angles of departure and angles of arrival of each path.
    The steering coefficient for TX antenna i is a_t_i, and the steering 
    coefficient for RX antenna j is a_r_j.
    
    An Additive White Gaussian Noise (AWGN) also corrupts the output signal.   

    Parameters
    ----------
    tx_sig : Numpy 2D array
        Input signal. Each line corresponds to the signal emitted by an antenna.
    path_params :  List
        List of K elements containing the paths parameters. Each element is
        another list containing the complex attenuations, Doppler
        frequencies and delays of the paths. Here is an example for a two-path
        channel:
            
        [[alpha_0,f_D_0,tau_0,theta_t_0,theta_r_0],[alpha_1,f_D_1,tau_1,theta_t_1,theta_r_1]]
    a_t : List of functions
        Steering vector of the transmitter. Each element of the list is a function
        of the angle of departure.
    a_r : List of functions
        Steering vector of the receiver. Each element of the list is a function
        of the angle of arrival.
    var_noise : float
        Noise power.
    fs_tx : Float
        Input sample rate, supposed to be a multiple of <fs_rx>.
    fs_rx : Float
        Output sample rate, supposed to be lower than <fs_tx>.

    Returns
    -------
    List of Numpy 1D array
        Output signal for each antenna.
    """
    
    N_TX = len(a_t)
    N_RX = len(a_r)
    N_paths = len(path_params[0])
    R = fs_tx/fs_rx
    
    out = []
    for j in range(N_RX):
        rx_sigs = []
        for i in range(N_TX):
            tx_sig_i = tx_sig[i]
            for k in range(N_paths):
                alpha,tau,f_D,theta_t,theta_r = path_params[k]
            
                n_delay = int(ceil(fs_rx*tau))
                eps = fs_tx*tau - floor(fs_rx*tau) * R
        
                f = interp1d(arange(len(tx_sig_i)),tx_sig_i,kind="linear",bounds_error=False,fill_value=0)
                tx_interp = f(arange(((R-eps) % R),len(tx_sig_i),int(R)))
                
                rx_sig_i = concatenate((zeros((n_delay,),dtype=complex64),tx_interp))
                rx_sig_i *= alpha * exp(1j*2*pi*f_D*arange(len(rx_sig_i))/fs_rx)
                rx_sig_i *= conj(a_t(theta_t)[i]) * conj(a_r(theta_r)[j])
                
                rx_sigs.append(rx_sig_i)
    
        rx_sig = add_signals(*rx_sigs)
        rx_sig += normal(0,sqrt(var_noise/2),len(rx_sig)) + 1j*normal(0,sqrt(var_noise/2),len(rx_sig))
        out.append(rx_sig)
    return out