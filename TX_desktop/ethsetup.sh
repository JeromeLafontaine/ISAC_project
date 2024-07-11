ifconfig enp1s0f0 up
ip addr add 192.168.40.1/24 dev enp1s0f0
ifconfig enp1s0f0 mtu 9000

ifconfig enp1s0f1 up
ip addr add 192.168.50.1/24 dev enp1s0f1
ifconfig enp1s0f1 mtu 9000

ifconfig enp2s0f0 up
ip addr add 192.168.60.1/24 dev enp2s0f0
ifconfig enp2s0f0 mtu 9000

ifconfig enp2s0f1 up
ip addr add 192.168.70.1/24 dev enp2s0f1
ifconfig enp2s0f1 mtu 9000

ifconfig -a
uhd_find_devices
