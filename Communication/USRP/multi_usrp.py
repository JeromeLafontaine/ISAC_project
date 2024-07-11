# https://files.ettus.com/manual/index.html 
# https://kb.ettus.com/UHD_Python_API
# https://pysdr.org/content/usrp.html 
# https://github.com/EttusResearch/uhd/blob/master/host/examples/python/benchmark_rate.py

# https://www.ni.com/docs/fr-FR/bundle/usrp-2944-feature/page/block-diagram.html

%matplotlib qt
# %matplotlib inline

import matplotlib.pyplot as plt
import numpy as np
import uhd 
import sys
import threading

import time
from datetime import datetime, timedelta

class ReturnThread(threading.Thread):
    def __init__(self, group=None, target=None, name=None,
                 args=(), kwargs={}, Verbose=None):
        threading.Thread.__init__(self, group, target, name, args, kwargs)
        
        self._return = None

    def run(self):
        if self._target is not None:
            self._return = self._target(*self._args,**self._kwargs)
    def join(self, *args):
        threading.Thread.join(self, *args)
        return self._return

### PARAMETERS
CLOCK_TIMEOUT = 1000  # Timeout for external clock locking [ms]
STARTING_TIME_DELAY = 3.  # Initial delay before transmit [s]
ADDITIONAL_DELAY_TX = 0.001

SOB_TIME_DURATION = 2e-5 # Start of burst time duration [s]
EOB_TIME_DURATION = 2e-5 # End of burst time duration [s]

MASTER_CLOCK_RATE = 2e8

RX_DC_OFFSET_CORRECTION = True
RX_IQ_BALANCE = True

carrier_freq = 2e9 # Carrier frequency [Hz]

tx_samp_rate = 100e6 # TX sample rate [Hz]
rx_samp_rate = 100e6 # RX sample rate [Hz]

rx_bandwidth = 160e6 # RX bandwidth on the frontend [Hz]

tx_gain = 0 # TX gain [dB]
rx_gain = 0 # RX gain [dB]

tx_signal_duration = 0.001 # Duration of the transmission [s]
rx_time_duration = 0.003 # Duration of the measurement [s]

SOB_num_samples = int(np.round(SOB_TIME_DURATION*tx_samp_rate))
EOB_num_samples = int(np.round(EOB_TIME_DURATION*tx_samp_rate))
tx_num_samples = int(np.round(tx_signal_duration*tx_samp_rate))
rx_num_samples = int(np.round(rx_time_duration*rx_samp_rate))

def wait_for_pps(usrp):
    time_last_pps = usrp.get_time_last_pps(0).get_real_secs()
    while time_last_pps == usrp.get_time_last_pps(0).get_real_secs():
        time.sleep(CLOCK_TIMEOUT/1e6)
    return

def check_if_locked(usrp):
    for i in range(usrp.get_num_mboards()):
        # PPS frequency = 1Hz => wait at least the duration of one clock edge to see if the USRP is locked
        end_time = datetime.now() + timedelta(milliseconds=CLOCK_TIMEOUT)
        is_locked = usrp.get_mboard_sensor("ref_locked", i)
        while (not is_locked) and (datetime.now() < end_time):
            time.sleep(CLOCK_TIMEOUT/1e6)
            is_locked = usrp.get_mboard_sensor("ref_locked", i)
        if not is_locked:
            print("[ERROR] Unable to confirm clock signal locked on USRP {}".format(i))
            sys.exit()
        else:
            print("USRP n°{} locked.".format(i))
    return
    
def transmit(usrp,tx_streamer,tx_signal,absolute_starting_time):
    tx_buffer_size = tx_streamer.get_max_num_samps()
    print("TX buffer size:", tx_buffer_size)
    tx_metadata = uhd.types.TXMetadata()
    
    tx_metadata.has_time_spec = True 
    tx_metadata.time_spec = absolute_starting_time
    tx_metadata.end_of_burst = False

    # SOB packet
    tx_metadata.start_of_burst = True
    SOB_packet = np.zeros((1,SOB_num_samples), dtype=np.complex64)
    SOB_packet[0,:SOB_num_samples//2] = 1
    print('SOB ',tx_streamer.send(SOB_packet, tx_metadata))
    tx_metadata.start_of_burst = False
    tx_metadata.has_time_spec = False

    num_transmitted_samples = 0
    while num_transmitted_samples < len(tx_signal):
        if num_transmitted_samples + tx_buffer_size > len(tx_signal):
            tx_signal_idx = len(tx_signal)
            buffer_length = len(tx_signal) - num_transmitted_samples
            tx_buffer = np.zeros((1, buffer_length), dtype=np.complex64)
        else:
            tx_signal_idx = num_transmitted_samples + tx_buffer_size
            tx_buffer = np.zeros((1, tx_buffer_size), dtype=np.complex64)
        
        tx_buffer[0,:] = tx_signal[num_transmitted_samples:tx_signal_idx]
        num_transmitted_samples += tx_streamer.send(tx_buffer, tx_metadata)

    # EOB packet
    tx_metadata.end_of_burst = True
    EOB_packet = np.zeros((1,EOB_num_samples), dtype=np.complex64)
    EOB_packet[0,EOB_num_samples//2:] = 1
    print('EOB ',tx_streamer.send(EOB_packet, tx_metadata))

    print("Transmission completed!")

def receive(usrp,rx_streamer,num_samples,absolute_starting_time):
    rx_samples = np.zeros(num_samples, dtype=np.complex64)

    rx_buffer_size = np.min([rx_streamer.get_max_num_samps(),num_samples])
    rx_buffer = np.empty((1, rx_buffer_size), dtype=np.complex64)

    rx_metadata = uhd.types.RXMetadata()

    rx_stream_cmd = uhd.types.StreamCMD(uhd.types.StreamMode.num_done) # only recover a finite number of samples
    rx_stream_cmd.num_samps = num_samples

    # start when wanted    
    rx_stream_cmd.stream_now = False
    rx_stream_cmd.time_spec = absolute_starting_time

    rx_streamer.issue_stream_cmd(rx_stream_cmd)

    waiting_to_start = True
    n_samps = 0
    i = 0
    while n_samps != 0 or waiting_to_start:
        n_samps = rx_streamer.recv(rx_buffer, rx_metadata)
        if n_samps != 0 and waiting_to_start:
            rx_last_pps_time = usrp.get_time_last_pps(1).get_real_secs()
            rx_absolute_time = usrp.get_time_now(1).get_real_secs()
            waiting_to_start = False
        elif n_samps != 0:
            rx_samples[i:i+n_samps] = rx_buffer[0][:n_samps]
        i += n_samps

        if rx_metadata.error_code in (uhd.types.RXMetadataErrorCode.none,uhd.types.RXMetadataErrorCode.timeout):
            pass
        elif rx_metadata.error_code == uhd.types.RXMetadataErrorCode.overflow:
            print("[WARNING RX] Overflow")
        else:
            print("[WARNING RX] ",rx_metadata.error_code)

    print("Reception completed!")
    return rx_samples 

def handle_async_tx_metadata(tx_streamer):
    async_metadata = uhd.types.TXAsyncMetadata()
    while True:
        if not tx_streamer.recv_async_msg(async_metadata,0.1):
            continue

        if async_metadata.event_code == uhd.types.TXMetadataEventCode.burst_ack:
            return
        elif async_metadata.event_code == uhd.types.TXMetadataEventCode.underflow:
            print("[WARNING TX] Underflow")
        elif async_metadata.event_code == uhd.types.TXMetadataEventCode.underflow_in_packet:
            print("[WARNING TX] Underflow in packet")
        elif async_metadata.event_code == uhd.types.TXMetadataEventCode.seq_error:
            print("[WARNING TX] Packet lost between host and device")
        elif async_metadata.event_code == uhd.types.TXMetadataEventCode.seq_error_in_packet:
            print("[WARNING TX] Sequence error in packet")
        else:
            print("[WARNING TX] ", async_metadata.event_code)

### USRP Configuration
usrp = uhd.usrp.MultiUSRP("resource0=RIO0,resource1=RIO1,master_clock_rate={}".format(MASTER_CLOCK_RATE))

print("Number of USRPs: ", usrp.get_num_mboards())
for i in range(usrp.get_num_mboards()):
    print("URSP n°{}: {}, master clock rate = {}MHz".format(i,usrp.get_mboard_name(i),usrp.get_master_clock_rate(i)*1e-6))

### USRP Antennas/Channels Setting
print("TX antennas/channels setting...") # Only 1 TX antenna/channel, check commented lines for 2 TX antennas/channels

# usrp.set_tx_subdev_spec(uhd.usrp.SubdevSpec("A:0 B:0"),0)
usrp.set_tx_subdev_spec(uhd.usrp.SubdevSpec("A:0"),0)
usrp.set_tx_antenna('TX/RX',0)
# usrp.set_tx_antenna('TX/RX',1)
usrp.set_rx_subdev_spec(uhd.usrp.SubdevSpec(""),0)

print("RX antennas/channels setting...") # Only 1 RX antenna/channel, check commented lines for 2 RX antennas/channels

usrp.set_tx_subdev_spec(uhd.usrp.SubdevSpec(""),1)
# usrp.set_rx_subdev_spec(uhd.usrp.SubdevSpec("A:0 B:0"),1)
usrp.set_rx_subdev_spec(uhd.usrp.SubdevSpec("A:0"),1)
usrp.set_rx_antenna('TX/RX',0)
# usrp.set_rx_antenna('TX/RX',1)

print("Channels summary:")
print(usrp.get_pp_string())

### USRP Synchronisation Setting
print("Synchronisation settings...")

usrp.set_clock_source("external")
usrp.set_time_source("external")

check_if_locked(usrp) # Check if USRPS are locked

# Time reset
wait_for_pps(usrp)
usrp.set_time_next_pps(uhd.types.TimeSpec(0.0))

time.sleep(CLOCK_TIMEOUT/1e3)

time_last_pps = usrp.get_time_last_pps(0).get_real_secs()
while time_last_pps == usrp.get_time_last_pps(0).get_real_secs():
    time.sleep(CLOCK_TIMEOUT/1e6)

tx_time = usrp.get_time_now(0).get_real_secs(); 
rx_time = usrp.get_time_now(0).get_real_secs()
print("TX: last pps={}s, now={}s".format(usrp.get_time_last_pps(0).get_real_secs(),tx_time))
print("RX: last pps={}s, now={}s".format(usrp.get_time_last_pps(1).get_real_secs(),rx_time))

## does not work owing to delay between multiple command lines => use C++ ?
# if usrp.get_time_synchronized():
#     print("Time synchronized.")
# else:
#     print("[ERROR] Unable to synchronize timers.")
#     sys.exit()

# Frequency tuning
wait_for_pps(usrp)
usrp.set_command_time(usrp.get_time_last_pps(0) + uhd.types.TimeSpec(1.)) 
usrp.set_tx_freq(uhd.types.TuneRequest(carrier_freq),0)
usrp.set_rx_freq(uhd.types.TuneRequest(carrier_freq),0)
usrp.clear_command_time()

### TX Setting

print("TX setting...")
usrp.set_tx_rate(tx_samp_rate)
usrp.set_tx_gain(tx_gain)

tx_st_args = uhd.usrp.StreamArgs("fc32", "sc16") # CPU and OTW data format, modify this can help increase the bandwidth
tx_st_args.channels = [0]
tx_streamer = usrp.get_tx_stream(tx_st_args)

print("Actual TX sample rate: {}MHz".format(usrp.get_tx_rate()*1e-6))
print("Actual TX gain: {}dB".format(usrp.get_tx_gain()))

### RX Setting

print("RX setting...")

usrp.set_rx_rate(rx_samp_rate)
usrp.set_rx_gain(rx_gain)
usrp.set_rx_bandwidth(rx_bandwidth)

usrp.set_rx_dc_offset(RX_DC_OFFSET_CORRECTION)
usrp.set_rx_iq_balance(RX_IQ_BALANCE)

rx_st_args = uhd.usrp.StreamArgs("fc32", "sc16") # CPU and OTW data format, modify this can help increase the bandwidth
rx_st_args.channels = [0]
rx_streamer = usrp.get_rx_stream(rx_st_args)

print("Actual RX sample rate: {}MHz".format(usrp.get_rx_rate()*1e-6))
print("Actual RX gain: {}dB".format(usrp.get_rx_gain()))

## Transmission and reception

wait_for_pps(usrp)
print("USRP (TX) ready at {}s".format(usrp.get_time_now(0).get_real_secs()))

# Start at a given time
absolute_starting_time_tx = usrp.get_time_last_pps(0) + uhd.types.TimeSpec(STARTING_TIME_DELAY + ADDITIONAL_DELAY_TX)
absolute_starting_time_rx = usrp.get_time_last_pps(0) + uhd.types.TimeSpec(STARTING_TIME_DELAY)
print("Transmitting at {}s".format(absolute_starting_time_tx.get_real_secs()))
print("Receiving at {}s".format(absolute_starting_time_rx.get_real_secs()))

tx_signal = np.exp(1j*2*np.pi*1e5*np.arange(tx_num_samples)/tx_samp_rate)

tx_thread = threading.Thread(target=transmit,args=(usrp,tx_streamer,tx_signal,absolute_starting_time_tx))
tx_async_metadata_thread = threading.Thread(target=handle_async_tx_metadata,args=(tx_streamer,))
rx_thread = ReturnThread(target=receive,args=(usrp,rx_streamer,rx_num_samples,absolute_starting_time_rx))

print("Threads start...")
tx_async_metadata_thread.start()
tx_thread.start()
rx_thread.start()

rx_samples = rx_thread.join()

### PLOT
plt.figure()

t = absolute_starting_time_rx.get_real_secs() + np.arange(rx_num_samples) / rx_samp_rate

plt.plot(t,np.real(rx_samples))
plt.plot(t,np.imag(rx_samples))

plt.xlabel("Time (s)")