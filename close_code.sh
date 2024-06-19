#!/bin/bash
> valid_code.txt
for run in {1..100}; do
  python code_capacity_multiple.py -type x -dmg 63 >> valid_code.txt
done
