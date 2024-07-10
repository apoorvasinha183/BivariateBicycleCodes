#!/bin/bash
# Define the start and end values for the error rate
start=0.001
end=0.007
increment=0.001

# Loop through the range of error rates
>debug1.txt
current=$start
while (( $(echo "$current <= $end" | bc -l) )); do
  echo "Sampling with -err $current"
  python defect_based_decoder_setup.py -err $current -repair asym -arg 0 >> debug1.txt
  current=$(echo "$current + $increment" | bc -l)
done
current=$start
while (( $(echo "$current <= $end" | bc -l) )); do
  echo "Decoding with -err $current"
  python decoder_run.py -err $current >> debug1.txt
  current=$(echo "$current + $increment" | bc -l)
done
