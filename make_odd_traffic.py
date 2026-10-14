import os, random, socket, time

for _ in range(150):                       # 150 separate flows
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    dst = (f"203.0.113.{random.randint(1, 254)}", random.randint(20000, 60000))
    for _ in range(random.randint(1, 40)):  # 1-40 packets per flow
        size = random.choice([random.randint(20, 120), random.randint(600, 1400)])
        s.sendto(os.urandom(size), dst)
        time.sleep(random.uniform(0.005, 0.15))
    s.close()
print("done")
