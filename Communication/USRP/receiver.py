import numpy as np
import uhd
import matplotlib.pyplot as plt

# Receiver USRP parameters
receiver_args = "addr=192.168.10.2"
receiver_freq = 1e6  # Receiver frequency in Hz

# Create receiver USRP instance
receiver_usrp = uhd.usrp.MultiUSRP(receiver_args)
receiver_usrp.set_clock_source("external")
receiver_usrp.set_time_source("external")
receiver_usrp.set_time_next_pps(uhd.time_spec(0.0))
receiver_usrp.clear_command_time()

# Create the receiver streamer
rx_stream_args = uhd.stream_args("fc32")
rx_stream_args.channels = [0]  # Use channel 0
rx_streamer = receiver_usrp.get_rx_stream(rx_stream_args)
rx_streamer.issue_stream_cmd(uhd.stream_cmd(uhd.stream_cmd.STREAM_MODE_START_CONTINUOUS))

# Receive and process the signal
num_samples = 10000
received_samples = np.empty(num_samples, dtype=np.complex64)

for i in range(num_samples):
    rx_streamer.recv(received_samples[i:i + 1], num_samples=1)

# Close the USRP device
receiver_usrp.close()

# Plot the received samples
plt.figure()

plt.title("Received Signal")
plt.plot(np.real(received_samples), label="I")
plt.plot(np.imag(received_samples), label="Q")
plt.legend()
plt.xlabel("Sample Index")
plt.ylabel("Amplitude")
plt.grid()
plt.show()
