from scapy.all import Ether, IP, TCP, Raw, wrpcap

packets = []

client = "192.0.2.10"
server = "198.51.100.20"

sport = 51000
dport = 45678

seq_client = 1000
seq_server = 2000

packets.append(
    Ether()/IP(src=client, dst=server)/
    TCP(sport=sport, dport=dport, flags="S", seq=seq_client)
)

packets.append(
    Ether()/IP(src=server, dst=client)/
    TCP(sport=dport, dport=sport, flags="SA",
        seq=seq_server, ack=seq_client + 1)
)

packets.append(
    Ether()/IP(src=client, dst=server)/
    TCP(sport=sport, dport=dport, flags="A",
        seq=seq_client + 1, ack=seq_server + 1)
)

payload = b"Test traffic for Random Forest classification"

packets.append(
    Ether()/IP(src=client, dst=server)/
    TCP(sport=sport, dport=dport, flags="PA",
        seq=seq_client + 1, ack=seq_server + 1)/
    Raw(load=payload)
)

wrpcap("test_unknown_network.pcap", packets)

print("Created test_unknown_network.pcap")
print(f"Total packets: {len(packets)}")
