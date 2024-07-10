#!/bin/bash

set_terminal_title() {
  echo -ne "\033]0;$1\007"
}


set_terminal_title "Decoding All"
# Define the start and end values for the error rate
start=0.001
end=0.007
increment=0.001

# Loop through the range of error rates

current=$start
while (( $(echo "$current <= $end" | bc -l) )); do
  echo "Sampling with -err $current"
  python decoder_aggregate_run_special.py -err $current  -sit allcommon -style zx
  current=$(echo "$current + $increment" | bc -l)
done