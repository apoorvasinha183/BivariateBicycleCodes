#!/bin/bash

set_terminal_title() {
  echo -ne "\033]0;$1\007"
}


set_terminal_title "Decoding OnlyZ"
# Define the start and end values for the error rate
start=0.001
end=0.007
increment=0.001

# Loop through the range of error rates

current=$start
while (( $(echo "$current <= $end" | bc -l) )); do
  echo "Sampling with -err $current"
  python decoder_aggregate_run.py -err $current  -sit onlyz
  current=$(echo "$current + $increment" | bc -l)
done
#python circuit_distance.py -sit onlyz