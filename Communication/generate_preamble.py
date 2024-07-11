#DOC STRING GENERATED AFTER THE SUBMISSION DEADLINE WITH THE HELP OF Gemini

import matplotlib.pyplot as plt
import numpy as np
import scipy.io
import os

path = os.path.dirname(os.path.abspath(__file__)) + "/preamble_files/"

def generate_preamble(file, plot=False):
    """Loads and processes preamble data from a MATLAB file.

    This function reads preamble information from a MATLAB file (`.mat`), extracting 
    and organizing data for multiple signals: LSTF, LLTF, LSIG, RLSIG, HESIGA, HESTF, and HELTF.

    Args:
        file (str): The path to the MATLAB file containing the preamble data.
        plot (bool, optional): If True, plots the extracted signals on a single graph. Defaults to False.

    Returns:
        list: A list containing the extracted preamble data arrays in the following order:
            - LSTF
            - LLTF
            - LSIG
            - RLSIG
            - HESIGA
            - HESTF
            - HELTF

    Raises:
        FileNotFoundError: If the specified `file` does not exist.
        KeyError: If the MATLAB file does not contain the expected signal keys.
        ImportError: If the `scipy.io` module is not available.

    Example:
        ```python
        preamble_data = generate_preamble('preamble_data.mat', plot=True)
        ```
    """

    preamble = scipy.io.loadmat(path + file)

    # Extract signal data
    LSTF = preamble['LSTF']
    LSIG = preamble['LSIG']
    LLTF = preamble['LLTF']
    RLSIG = preamble['RLSIG']
    HELTF = preamble['HELTF']
    HESIGA = preamble['HESIGA']
    HESTF = preamble['HESTF']

    preamble = [LSTF, LLTF, LSIG, RLSIG, HESIGA, HESTF, HELTF]

    if plot:
        plt.figure()
        plt.plot(LSTF, label='LSTF')
        plt.plot(np.arange(len(LSTF), len(LSTF)+len(LLTF)), LLTF, label='LLTF')
        plt.plot(np.arange(len(LSTF)+len(LLTF), len(LSTF)+len(LLTF)+len(LSIG)), LSIG, label='LSIG')
        plt.plot(np.arange(len(LSTF)+len(LLTF)+len(LSIG), len(LSTF)+len(LLTF)+len(LSIG)+len(RLSIG)), RLSIG, label='RLSIG')
        plt.plot(np.arange(len(LSTF)+len(LLTF)+len(LSIG)+len(RLSIG), len(LSTF)+len(LLTF)+len(LSIG)+len(RLSIG)+len(HESIGA)), HESIGA, label='HESIGA')
        plt.plot(np.arange(len(LSTF)+len(LLTF)+len(LSIG)+len(RLSIG)+len(HESIGA), len(LSTF)+len(LLTF)+len(LSIG)+len(RLSIG)+len(HESIGA)+len(HESTF)), HESTF, label='HESTF')
        plt.plot(np.arange(len(LSTF)+len(LLTF)+len(LSIG)+len(RLSIG)+len(HESIGA)+len(HESTF), len(LSTF)+len(LLTF)+len(LSIG)+len(RLSIG)+len(HESIGA)+len(HESTF)+len(HELTF)), HELTF, label='HELTF')
        plt.legend()
        plt.show()

    return preamble

