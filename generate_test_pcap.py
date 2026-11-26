from scapy.all import Ether, IP, TCP, Raw, wrpcap

packets = []

client = "192.0.2.10"
server = "198.51.100.20"

sport = 50000
dport = 80

seq_client = 1000
seq_server = 2000

# TCP handshake
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

# HTTP request
request = b"GET / HTTP/1.1\r\nHost: example.com\r\n\r\n"

packets.append(
    Ether()/IP(src=client, dst=server)/
    TCP(sport=sport, dport=dport, flags="PA",
        seq=seq_client + 1, ack=seq_server + 1)/
    Raw(load=request)
)

# HTTP response
response = b"HTTP/1.1 200 OK\r\nContent-Length: 5\r\n\r\nHello"

packets.append(
    Ether()/IP(src=server, dst=client)/
    TCP(sport=dport, dport=sport, flags="PA",
        seq=seq_server + 1,
        ack=seq_client + 1 + len(request))/
    Raw(load=response)
)

wrpcap("test_network.pcap", packets)

print("Created test_network.pcap")
print(f"Total packets: {len(packets)}")
