#DOC STRING GENERATED AFTER THE SUBMISSION DEADLINE WITH THE HELP OF Gemini

import numpy as np
import matplotlib.pyplot as plt


def bin_to_mod(x : np.array , d : int , map: str) -> np.array:
    """Converts binary input to modulated symbols for QAM or BPSK schemes.

    This function takes binary input data (`x`), a scaling factor (`d`), and a modulation
    scheme (`map`) and converts it to modulated complex symbols. It supports two modulation
    schemes: 4-QAM and BPSK.

    Args:
        x (np.array): A NumPy array containing the binary input data (0s and 1s).
        d (int): The scaling factor to apply to the modulated symbols.
        map (str): The modulation scheme to use. Must be either "4-QAM" or "BPSK".

    Returns:
        np.array: A NumPy array containing the modulated complex symbols.
    """ 
    if map == "4-QAM":
        u = x.reshape(-1,2)
        reel = 2*u[:,0]-1
        img = 2*u[:,1]-1
        u = d*(reel+1j*img)
        return u
    if map == "BPSK":
        u = x.reshape(-1,1)
        u = 2*u - 1
        return u*d

def mod_to_bin(x : np.array , d : int , map: str) -> np.array:
    """Converts modulated symbols back to binary data for QAM or BPSK schemes.

    This function takes modulated complex symbols (`x`), a scaling factor (`d`), and a
    modulation scheme (`map`) and converts them back to binary data (0s and 1s). It
    supports two modulation schemes: 4-QAM and BPSK.

    Args:
        x (np.array): A NumPy array containing the modulated complex symbols.
        d (int): The scaling factor used during modulation (not directly used in this function).
        map (str): The modulation scheme used. Must be either "4-QAM" or "BPSK".

    Returns:
        np.array: A NumPy array containing the demodulated binary data (0s and 1s).
    """
    if map == "4-QAM":
        u = np.zeros((x.shape[0],2))
        for i in range(x.shape[0]):
            if(np.real(x[i]) > 0):
                u[i,0] = 1
            else:
                u[i,0] = 0
            if(np.imag(x[i]) > 0):
                u[i,1] = 1
            else:
                u[i,1] = 0
        u = u.reshape(-1,1)
        return u
    if map == "BPSK":
        u = np.zeros((x.shape[0],1))
        for i in range(x.shape[0]):
            if(np.real(x[i]) > 0):
                u[i,0] = 1
            else:
                u[i,0] = 0
        return u

def plot_constellation(x : np.array, title  : str ,const = "4-QAM") -> None:
    """Plots a constellation diagram for modulated symbols.

    This function creates a scatter plot of modulated symbols represented by complex
    numbers. It supports two constellation types: 4-QAM and a generic constellation 
    where the specific constellation points are not drawn (assuming the caller has already
    modulated the data appropriately). For 4-QAM, the function also draws the
    quadrant boundaries and ideal constellation points.

    Args:
        x (np.array): A NumPy array of complex numbers representing the modulated symbols.
        title (str): The title to display above the plot.
        const (str, optional): The type of constellation. Must be either "4-QAM" 
                               (for 4-Quadrature Amplitude Modulation) or any other
                               string for a generic constellation. Defaults to "4-QAM".

    Returns:
        None: The function displays the plot but doesn't return any values.
    """
    plt.scatter(x.real,x.imag)
    if(const == "4-QAM"):
        plt.axvline(x=0, color='k')
        plt.axhline(y=0, color='k')
        plt.scatter([1,-1,-1,1],[1,1,-1,-1],color='g', marker='x')
    plt.grid()
    plt.xlabel('Reel')
    plt.ylabel('Imaginary')   
    plt.title(title) 
    plt.show()



