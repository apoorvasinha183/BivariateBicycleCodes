#!/bin/bash

set_terminal_title() {
  echo -ne "\033]0;$1\007"
}


set_terminal_title "Sampling Distant"
# Define the start and end values for the error rate
start=0.001
end=0.007
increment=0.001

# Loop through the range of error rates

current=$start
while (( $(echo "$current <= $end" | bc -l) )); do
  echo "Sampling with -err $current"
  python defect_aggereate_setup.py -err $current -repair asym -arg 0 -sit allcommon
  current=$(echo "$current + $increment" | bc -l)
done
python circuit_distance.py -sit allcommon -style zx