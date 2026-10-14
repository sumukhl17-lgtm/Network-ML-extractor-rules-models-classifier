import os, random, socket, threading, time

def one_flow():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    dst = (f"203.0.113.{random.randint(1, 254)}", random.randint(20000, 60000))
    for _ in range(random.randint(10, 40)):
        s.sendto(os.urandom(random.randint(100, 900)), dst)
        time.sleep(random.uniform(0.3, 1.5))
    s.close()

threads = [threading.Thread(target=one_flow) for _ in range(120)]
for t in threads:
    t.start()
    time.sleep(0.05)
for t in threads:
    t.join()
print("done")
