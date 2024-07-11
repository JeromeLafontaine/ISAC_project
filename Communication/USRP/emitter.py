import numpy as np
import uhd
from uhd import check_if_locked
from uhd import wait_for_pps
import matplotlib.pyplot as plt
import threading
from threading import ReturnThread


### PARAMETERS ###
CLOCK_TIMEOUT = 1000 
STARTING_TIME_DELAY = 3.
ADDITIONNAL_DELAY_TX = 0.0005


MASTER_CLOCK_RATE = 100e6  # Master clock rate in Hz

RX_DC_OFFSET_CORRECTION = True
RX_IQ_BALANCE = True

carrier_freq = 1e6  # Carrier frequency in Hz

tx_samp_rate = 10e6  # TX sample rate in Hz
rx_samp_rate = 10e6  # RX sample rate in Hz

rx_bandwidth = 160e6  # RX bandwidth in Hz

tx_gain = 0  # TX gain in dB
rx_gain = 0  # RX gain in dB

num_samples = 10000  # Number of samples to receive



### USRP Configuration ###
usrp = uhd.usrp.MultiUSRP("resource0=RIO0, resource1=RIO1, master_clock_rate={}".format(MASTER_CLOCK_RATE))

usrp.set_tx_subdev_spec(uhd.usrp.SubdevSpec("A:0"), 0)
usrp.set_tx_antenna("TX/RX", 0)
usrp.set_rx_subdev_spec(uhd.usrp.SubdevSpec(""), 0)

usrp.set_tx_subdev_spec(uhd.usrp.SubdevSpec(""), 1)
usrp.set_rx_subdev_spec(uhd.usrp.SubdevSpec("A:0"), 1)
usrp.set_rx_antenna("TX/RX", 1)

usrp.set_clock_source("external")
usrp.set_time_source("external")

check_if_locked(usrp)

# Time reset
wait_for_pps(usrp)
usrp.set_time_next_pps(uhd.time_spec(0.0))

# Frequency tuning
wait_for_pps(usrp)
usrp.set_command_time(usrp.get_time_last_pps() + uhd.types.TimeSpec(1.))
usrp.set_tx_freq(uhd.types.TuneRequest(carrier_freq), 0)
usrp.set_rx_freq(uhd.types.TuneRequest(carrier_freq), 1)
usrp.clear_command_time()

### TX Setting ###
print("TX setting...")
usrp.set_tx_rate(tx_samp_rate)
usrp.set_tx_gain(tx_gain)

tx_st_args = uhd.usrp.StreamArgs("fc32", "sc16")
tx_st_args.channel = [0]
tx_streamer = usrp.get_tx_stream(tx_st_args)

### RX Setting ###

print("RX setting...")
usrp.set_rx_rate(rx_samp_rate)
usrp.set_rx_gain(rx_gain)
usrp.set_rx_bandwidth(rx_bandwidth)

usrp.set_rx_dc_offset(RX_DC_OFFSET_CORRECTION)
usrp.set_rx_iq_balance(RX_IQ_BALANCE)

rx_st_args = uhd.usrp.StreamArgs("fc32", "sc16")
rx_st_args.channels = [0]
rx_streamer = usrp.get_rx_stream(rx_st_args)

## Transmission and reception 
wait_for_pps(usrp)
print("USRP (TX) ready at {}s".format(usrp.get_time_now(0).get_real_secs()))

#Start at a given time
absolute_starting_time_tx = usrp.get_time_last_pps(0) + uhd.types.TimeSpec(STARTING_TIME_DELAY + ADDITIONNAL_DELAY_TX)
absolute_starting_time_rx = usrp.get_time_last_pps(0) + uhd.types.TimeSpec(STARTING_TIME_DELAY)
print("Transmitting at {}s".format(absolute_starting_time_tx.get_real_secs()))
print("Receiving at {}s".format(absolute_starting_time_rx.get_real_secs()))

tx_signal = np.exp(1j*2*np.pi*1e5*np.arange(1000)/tx_samp_rate)

tx_thread = threading.Thread(target=transmit, args=(usrp, tx_streamer, tx_signal, absolute_starting_time_tx))
rx_thread = ReturnThread(target=receive, args=(usrp, rx_streamer, num_samples, absolute_starting_time_rx))

tx_thread.start()
rx_thread.start()

print("Threads started...")

rx_samples = rx_thread.join()
