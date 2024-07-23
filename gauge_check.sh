#!/bin/bash
> gauge_trail.txt
for i in {1..50}
do
    # Replace 'your_command_here' with the command you want to run
    python explicit_distance.py >>trial.txt
done