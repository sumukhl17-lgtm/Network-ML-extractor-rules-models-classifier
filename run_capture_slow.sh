#!/bin/bash
# usage: ./run_capture.sh capture3.pcap
OUT=${1:-capture3.pcap}
sudo -v
sudo tcpdump -i en0 -w "$OUT" &
sleep 3
python3 make_odd_traffic_slow.py &
ODD=$!
for round in 1 2 3 4 5 6; do
  for site in https://www.wikipedia.org https://www.python.org https://github.com \
              https://www.bbc.com https://news.ycombinator.com http://neverssl.com \
              https://stackoverflow.com https://www.apple.com; do
    curl -s -o /dev/null -m 10 "$site"
    dig +short example.com @8.8.8.8 > /dev/null
  done
done
wait $ODD
sleep 3
sudo pkill -INT tcpdump
sleep 2
sudo chown $USER "$OUT"
echo "saved $OUT"
